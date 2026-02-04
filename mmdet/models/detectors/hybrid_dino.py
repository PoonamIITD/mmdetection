# Copyright (c) OpenMMLab. All rights reserved.
# Hybrid model: GDINO Encoder + DINO Decoder
import copy
import torch
import os
import json
import cv2
import numpy as np

import h5py
from filelock import FileLock

import pandas as pd
import torch.nn as nn
from torch import Tensor
from collections import defaultdict
from pycocotools.coco import COCO
import re
from typing import Dict, Tuple, Optional, Union, List
import warnings
from mmengine.runner.amp import autocast

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoder
from .deformable_detr import DeformableDETR
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerEncoder
)
from ..layers.transformer.dino_layers import (
    DinoTransformerDecoder
)
from .dino import DINO
from .glip import create_positive_map, create_positive_map_label_to_token, run_ner

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

@MODELS.register_module()
class HybridDINO(DINO):
    """Hybrid model: GroundingDINO Encoder + DINO Decoder.

    Encoder: multimodal fusion from GroundingDINO
    Decoder: vanilla DINO decoder
    """

    def __init__(self,
                 language_model,
                 text_encoder,
                 dino_ratio=0.0,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:
        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.text_encoder = text_encoder
        super().__init__(*args, **kwargs)
        # gdino_embed_dir = 'gdino_embeddings' 
        # self.gdino_cache = {} 

        # import os 
        # for fname in os.listdir(gdino_embed_dir): 
        #     if fname.endswith(".pt"): 
        #         key = os.path.splitext(fname)[0] # filename without extension 
        #         self.gdino_cache[key] = torch.load( os.path.join(gdino_embed_dir, fname), map_location="cpu") 
                
                
        # import logging 
        # logging.basicConfig(level=logging.INFO) 
        # logger = logging.getLogger("Loading GDINO embeddings") 
        # logger.info(f"[HybridDINO] Loaded {len(self.gdino_cache)} GDINO embeddings into cache")  

    def _init_layers(self) -> None:
        """Use GroundingDINO encoder and DINO decoder."""
        # Positional encoding
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        # Encoder from GroundingDINO
        self.text_encoder = GroundingDinoTransformerEncoder(**self.text_encoder)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)

        # self.encoder = GroundingDinoTransformerEncoder(**self.encoder)

        # Decoder from DINO
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims

        # Query embedding as in DINO
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be 2x num_feats. Found {self.embed_dims} vs {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # for param in self.neck.parameters():
        #     param.requires_grad = False

        # for param in self.language_model.parameters():
        #     param.requires_grad = False

        # for param in self.text_feat_map.parameters():
        #     param.requires_grad = False
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.text_layers.parameters():
            param.requires_grad = False

        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.dino_encoder.parameters():
        #     param.requires_grad = False

        # for param in self.positional_encoding.parameters():
        #     param.requires_grad = False
        
        # self.level_embed.requires_grad = False


    def init_weights(self):
        """Initialize HybridDINO with GDINO (backbone+encoder) + DINO (decoder+head)."""
        super(DeformableDETR, self).init_weights()
        from mmengine import MMLogger

        logger = MMLogger.get_current_instance()
        # GDINO_swin-l_pretrained_rsud_best_mAP_epoch_19.pth
        gdino_ckpt = torch.load(
            '/home/poonam_rajput/scratch/mmdetection/checkpoints/GDINO_swin-l_pretrained_rsud_best_mAP_epoch_19.pth',
            map_location='cpu'
        )['state_dict']
        dino_ckpt = torch.load(
            '/home/poonam_rajput/scratch/mmdetection/checkpoints/DINO-swin-L_pretrained_rsud_best_epoch_16.pth',
            map_location='cpu'
        )['state_dict']

        # -------- GDINO: text_feat_map --------
        if hasattr(self, "text_feat_map"):
            text_feat_map_weights = {
                k.replace("text_feat_map.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("text_feat_map.")
            }

            missing, unexpected = self.text_feat_map.load_state_dict(text_feat_map_weights, strict=True)
            # print(self.backbone.state_dict().keys())
            # print([k for k in gdino_ckpt.keys() if k.startswith("backbone.")])

            logger.info(f"[GDINO] Loaded text_feat_map | missing={len(missing)}, unexpected={len(unexpected)}")

        # -------- GDINO: text_feat_map --------
        if hasattr(self, "language_model"):
            language_model_weights = {
                k.replace("language_model.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("language_model.")
            }

            missing, unexpected = self.language_model.load_state_dict(language_model_weights, strict=True)
            # print(self.backbone.state_dict().keys())
            # print([k for k in gdino_ckpt.keys() if k.startswith("backbone.")])

            logger.info(f"[GDINO] Loaded language_model | missing={len(missing)}, unexpected={len(unexpected)}")
        
        # -------- GDINO: backbone --------
        if hasattr(self, "backbone"):
            backbone_weights = {
                k.replace("backbone.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("backbone.")
            }

            missing, unexpected = self.backbone.load_state_dict(backbone_weights, strict=True)
            # print(self.backbone.state_dict().keys())
            # print([k for k in gdino_ckpt.keys() if k.startswith("backbone.")])

            logger.info(f"[GDINO] Loaded backbone | missing={len(missing)}, unexpected={len(unexpected)}")


        # -------- GDINO: neck --------
        if hasattr(self, "neck"):
            neck_weights = {
                k.replace("neck.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("neck.")
            }
            missing, unexpected = self.neck.load_state_dict(neck_weights, strict=True)
            logger.info(f"[GDINO] Loaded neck | missing={len(missing)}, unexpected={len(unexpected)}")

        # -------- GDINO: positional_encoding --------
        if hasattr(self, "positional_encoding"):
            positional_encoding_weights = {
                k.replace("positional_encoding.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("positional_encoding.")
            }

            missing, unexpected = self.positional_encoding.load_state_dict(positional_encoding_weights, strict=True)
            logger.info(f"[GDINO] Loaded positional_encoding | missing={len(missing)}, unexpected={len(unexpected)}")

        # -------- GDINO: level_embed --------
        if "level_embed" in gdino_ckpt:
            with torch.no_grad():
                self.level_embed.copy_(gdino_ckpt["level_embed"])
            logger.info("[GDINO] Loaded level_embed from checkpoint")
        else:
            logger.warning("[GDINO] level_embed not found in checkpoint — using random init")

        # -------- GDINO: text_encoder --------
        if hasattr(self, "text_encoder"):
            encoder_weights = {
                k.replace("encoder.", ""): v
                for k, v in gdino_ckpt.items() if k.startswith("encoder.")
            }
            missing, unexpected = self.text_encoder.load_state_dict(encoder_weights, strict=True)
            logger.info(f"[GDINO] Loaded text_encoder | missing={len(missing)}, unexpected={len(unexpected)}")


        # -------- DINO: encoder --------
        if hasattr(self, "encoder"):
            encoder_weights = {
                k.replace("encoder.", ""): v
                for k, v in dino_ckpt.items() if k.startswith("encoder.")
            }
            missing, unexpected = self.encoder.load_state_dict(encoder_weights, strict=True)
            logger.info(f"[DINO] Loaded encoder | missing={len(missing)}, unexpected={len(unexpected)}")

        # -------- DINO: decoder --------
        if hasattr(self, "decoder"):
            decoder_weights = {
                k.replace("decoder.", ""): v
                for k, v in dino_ckpt.items() if k.startswith("decoder.")
            }
            missing, unexpected = self.decoder.load_state_dict(decoder_weights, strict=True)
            logger.info(f"[DINO] Loaded decoder | missing={len(missing)}, unexpected={len(unexpected)}")

        # -------- DINO: bbox_head --------
        if hasattr(self, "bbox_head"):
            head_weights = {
                k.replace("bbox_head.", ""): v
                for k, v in dino_ckpt.items() if k.startswith("bbox_head.")
            }
            missing, unexpected = self.bbox_head.load_state_dict(head_weights, strict=False)
            logger.info(f"[DINO] Loaded bbox_head | missing={len(missing)}, unexpected={len(unexpected)}")

        if hasattr(self, "query_embedding"):
            query_embedding_weights = {
                k.replace("query_embedding.", ""): v
                for k, v in dino_ckpt.items() if k.startswith("query_embedding.")
            }

            missing, unexpected = self.query_embedding.load_state_dict(query_embedding_weights, strict=True)
            logger.info(f"[DINO] Loaded query_embedding | missing={len(missing)}, unexpected={len(unexpected)}")

        if hasattr(self, "dn_query_generator"):
            dn_query_generator_weights = {
                k.replace("dn_query_generator.", ""): v
                for k, v in dino_ckpt.items() if k.startswith("dn_query_generator.")
            }

            missing, unexpected = self.dn_query_generator.load_state_dict(dn_query_generator_weights, strict=True)
            logger.info(f"[DINO] Loaded dn_query_generator | missing={len(missing)}, unexpected={len(unexpected)}")

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.language_model.max_tokens)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def choose_embedding(self, filename, gdino_mem, dino_mem): 
        import hashlib 
        # hash filename to a number between 0 and 1 
        h = int(hashlib.md5(filename.encode()).hexdigest(), 16) 
        frac = (h % 10000) / 10000.0  
        if frac < self.dino_ratio: 
            return dino_mem.squeeze(0) # use GDINO 
        else: 
            return gdino_mem 
        
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Optional[Dict] = None,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Hybrid transformer forward: GDINO encoder + DINO decoder."""
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples=batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        # encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        # with torch.no_grad():  
        #     import os  
        #     if self.gdino_cache is not None:  
        #         for i, data_sample in enumerate(batch_data_samples):  
        #             img_path = data_sample.metainfo['img_path']  
        #             filename = os.path.splitext(os.path.basename(img_path))[0]  
        #             if filename not in self.gdino_cache:  
        #                 raise KeyError(f"DINO embedding missing for {filename}")  
        #             gdino_embed = self.gdino_cache[filename]
        #             emb_online = encoder_outputs_dict['memory'][i]
        #             emb_offline = gdino_embed['memory'].squeeze(0).to(img_feats[0].device)
        #             import torch.nn.functional as F
        #             cos = F.cosine_similarity(emb_offline.flatten(), emb_online.flatten(), dim=0)
        #             print(cos)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        # decoder is pure DINO
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: Optional[Dict] = None
    ) -> Dict:

        # ---- GDINO encoder forward ----
        
        # memory, memory_text = self.encoder(
        image_value, memory_text = self.text_encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_dict['text_token_mask'],
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'], 
            # return_intermediate = True
        )

        with torch.no_grad():
            memory = self.encoder(  #dino encoder
                query=feat,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,  # for self_attn
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios)

        return dict(
            memory=memory,
            image_value = image_value,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
        )

    def pre_decoder(
        self,
        memory: Tensor,
        image_value: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
        memory_per_layer: Optional[List[Tensor]] = None,   # NEW
    ) -> Tuple[Dict]:
        
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        
        # def to_numpy(x):
        #     return x.detach().cpu().numpy()

        # if (not self.training) and batch_data_samples is not None:
        #     data_sample = batch_data_samples[0] 
        #     img_meta = data_sample.metainfo 
        #     image_name = ( img_meta.get("ori_filename", None) 
        #                   or img_meta.get("file_name", None) or 
        #                   img_meta.get("filename", None) ) 
        #     if image_name is None and "img_path" in img_meta:
        #         image_name = os.path.basename(img_meta["img_path"])

        #     if image_name is not None:
        #         root_dir = os.getcwd()
        #         dump_dir = os.path.join(root_dir, "enc_all_props")
        #         os.makedirs(dump_dir, exist_ok=True)

        #         h5_path = os.path.join(dump_dir, "enc_all_props_hdino.h5")
        #         lock_path = h5_path + ".lock"

        #         with FileLock(lock_path):
        #             with h5py.File(h5_path, "a") as f:
        #                 root = f.require_group("images")

        #                 if image_name not in root:
        #                     g = root.create_group(image_name)

        #                     g.create_dataset(
        #                         "enc_outputs_class",
        #                         data=to_numpy(enc_outputs_class[0]),
        #                         compression="gzip",
        #                         compression_opts=4
        #                     )

        #                     g.create_dataset(
        #                         "enc_outputs_coord_unact",
        #                         data=to_numpy(enc_outputs_coord_unact[0]),
        #                         compression="gzip",
        #                         compression_opts=4
        #                     )

        #                     g.create_dataset(
        #                         "topk_indices",
        #                         data=to_numpy(topk_indices[0]),
        #                         compression="gzip"
        #                     )

        #                     g.create_dataset(
        #                         "spatial_shapes",
        #                         data=to_numpy(spatial_shapes),
        #                         compression="gzip"
        #                     )

        #                     g.attrs["num_encoder_tokens"] = int(enc_outputs_class.shape[1])
        #                     g.attrs["num_classes"] = int(enc_outputs_class.shape[-1])
        #                     g.attrs["num_queries"] = int(self.num_queries)
        
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            # memory=memory,
            memory=image_value,
            reference_points=reference_points,
            # memory_per_layer = memory_per_layer,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples=batch_data_samples)

        # 2. Save reference points visualization
        # if "references" in head_inputs_dict:
        #     self._save_ref_points(
        #         head_inputs_dict,
        #         batch_inputs=batch_inputs,
        #         batch_data_samples=batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        
        return losses
    
    
    def compute_iou(self, box1, box2):
        """IoU for two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0



    def analyze_predictions(self, batch_data_samples, save_dir, anno_file, score_thr=0.3):
        os.makedirs(save_dir, exist_ok=True)

        # Load COCO GT annotations
        coco_gt = COCO(anno_file)

        # Collect per-image stats for this batch
        per_image_stats = []
        class_stats = defaultdict(lambda: {"TP":0, "FP":0, "FN":0, "Misclass":0, "BBoxError":0})

        for data_sample in batch_data_samples:
            img_path = data_sample.img_path
            img = cv2.imread(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # ------------------------
            # Ground Truth
            # ------------------------
            img_id = None
            for i in coco_gt.getImgIds():
                info = coco_gt.loadImgs([i])[0]
                if info['file_name'] == os.path.basename(img_path):
                    img_id = i
                    break

            gt_anns = []
            gt_img = img.copy()
            if img_id is not None:
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                anns = coco_gt.loadAnns(ann_ids)
                for ann in anns:
                    x, y, w, h = map(int, ann["bbox"])
                    cat_name = coco_gt.loadCats([ann["category_id"]])[0]["name"]
                    gt_anns.append({"bbox":[x,y,x+w,y+h], "label":cat_name})
                    cv2.rectangle(gt_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(gt_img, f"{cat_name}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # ------------------------
            # Predictions
            # ------------------------
            pred_img = img.copy()
            pred = data_sample.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.label_names if hasattr(pred, "label_names") else pred.labels.cpu().numpy()

            pred_anns = []
            for bbox, score, label in zip(bboxes, scores, labels):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                pred_anns.append({"bbox":[x1,y1,x2,y2], "label":label, "score":score})
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(pred_img, f"{label}:{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ------------------------
            # Combine GT + Pred
            # ------------------------
            combined = np.concatenate((gt_img, pred_img), axis=1)
            out_file = os.path.join(save_dir, f"{base_name}_gt_pred.jpg")
            cv2.imwrite(out_file, combined)

            # ------------------------
            # Stats Calculation
            # ------------------------
            img_stats = {"image": base_name, "TotalBoxes":0, "TP":0, "FP":0, "FN":0, "Misclass":0, "BBoxError":0}

            matched_gt = set()
            matched_pred = set()

            def iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB-xA) * max(0, yB-yA)
                boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
                boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
                return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            img_stats["TotalBoxes"] = len(gt_anns)
            for gi, gt in enumerate(gt_anns):
                best_iou, best_pi = 0, None
                for pi, pred in enumerate(pred_anns):
                    if pi in matched_pred: 
                        continue
                    iou_val = iou(gt["bbox"], pred["bbox"])
                    if iou_val > best_iou:
                        best_iou, best_pi = iou_val, pi

                if best_pi is not None and best_iou > 0.5:
                    pred = pred_anns[best_pi]
                    matched_gt.add(gi)
                    matched_pred.add(best_pi)
                    if pred["label"] == gt["label"]:
                        img_stats["TP"] += 1
                        class_stats[gt["label"]]["TP"] += 1
                        if best_iou < 0.75:  # loose threshold for bbox error
                            img_stats["BBoxError"] += 1
                            class_stats[gt["label"]]["BBoxError"] += 1
                    else:
                        img_stats["Misclass"] += 1
                        class_stats[gt["label"]]["Misclass"] += 1
                else:
                    img_stats["FN"] += 1
                    class_stats[gt["label"]]["FN"] += 1

            for pi, pred in enumerate(pred_anns):
                if pi not in matched_pred:
                    img_stats["FP"] += 1
                    class_stats[pred["label"]]["FP"] += 1

            per_image_stats.append(img_stats)

        # ------------------------
        # Save Results
        # ------------------------
        df = pd.DataFrame(per_image_stats)
        csv_file = "detection_stats.csv"
        csv_path = os.path.join(save_dir, csv_file)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)  # append mode
        else:
            df.to_csv(csv_path, index=False)  # write with header if first time
        print(f"Appended batch stats to {csv_path}")

        return class_stats  # so you can aggregate per-class later




    def analyze_predictions2(self, batch_data_samples, save_dir, anno_file, score_thr=0.1):
        os.makedirs(save_dir, exist_ok=True)
        coco_gt = COCO(anno_file)

        per_image_stats = []
        class_stats = defaultdict(lambda: {"TP":0, "FP":0, "FN":0, "Misclass":0, "BBoxError":0})

        for data_sample in batch_data_samples:
            img_path = data_sample.img_path
            img = cv2.imread(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # ------------------------
            # Load Ground Truth
            # ------------------------
            img_id = None
            for i in coco_gt.getImgIds():
                info = coco_gt.loadImgs([i])[0]
                if info['file_name'] == os.path.basename(img_path):
                    img_id = i
                    break

            gt_anns = []
            if img_id is not None:
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                anns = coco_gt.loadAnns(ann_ids)
                for ann in anns:
                    x, y, w, h = map(int, ann["bbox"])
                    cat_name = coco_gt.loadCats([ann["category_id"]])[0]["name"]
                    gt_anns.append({"bbox": [x, y, x + w, y + h], "label": cat_name})

            # ------------------------
            # Load Predictions
            # ------------------------
            pred = data_sample.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.label_names if hasattr(pred, "label_names") else pred.labels.cpu().numpy()

            pred_anns = []
            for bbox, score, label in zip(bboxes, scores, labels):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                pred_anns.append({"bbox": [x1, y1, x2, y2], "label": label, "score": score})

            # ------------------------
            # Matching Logic
            # ------------------------
            matched_gt = set()
            matched_pred = set()

            def iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

            img_stats = {"image": base_name, "TotalBoxes": len(gt_anns), "TP": 0, "FP": 0, "FN": 0, "Misclass": 0, "BBoxError": 0}

            matches = []  # store matched pairs (GT, Pred, is_correct)

            for gi, gt in enumerate(gt_anns):
                best_iou, best_pi = 0, None
                for pi, pred in enumerate(pred_anns):
                    if pi in matched_pred:
                        continue
                    iou_val = iou(gt["bbox"], pred["bbox"])
                    if iou_val > best_iou:
                        best_iou, best_pi = iou_val, pi

                if best_pi is not None and best_iou > 0.5:
                    pred = pred_anns[best_pi]
                    matched_gt.add(gi)
                    matched_pred.add(best_pi)
                    is_correct = (pred["label"] == gt["label"])
                    matches.append((gt, pred, is_correct))
                    if is_correct:
                        img_stats["TP"] += 1
                        class_stats[gt["label"]]["TP"] += 1
                    else:
                        img_stats["Misclass"] += 1
                        class_stats[gt["label"]]["Misclass"] += 1
                else:
                    img_stats["FN"] += 1
                    class_stats[gt["label"]]["FN"] += 1

            for pi, pred in enumerate(pred_anns):
                if pi not in matched_pred:
                    img_stats["FP"] += 1
                    class_stats[pred["label"]]["FP"] += 1

            # ------------------------
            # Visualization (TP, FP, FN, Misclass)
            # ------------------------
            vis_img = img.copy()

            # ✅ True Positives (Green)
            for gt, pred, is_correct in matches:
                if is_correct:
                    x1, y1, x2, y2 = gt["bbox"]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"TP:{gt['label']}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ⬛ Misclassifications (Black) — show both GT and predicted label
            for gt, pred, is_correct in matches:
                if not is_correct:
                    x1, y1, x2, y2 = gt["bbox"]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    text = f"Mis: {gt['label']}→{pred['label']}"
                    cv2.putText(vis_img, text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(vis_img, text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white overlay text for contrast

            # 🟥 False Negatives (Red)
            for gi, gt in enumerate(gt_anns):
                if gi not in matched_gt:
                    x1, y1, x2, y2 = gt["bbox"]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(vis_img, f"FN:{gt['label']}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 🟦 False Positives (Blue)
            for pi, pred in enumerate(pred_anns):
                if pi not in matched_pred:
                    x1, y1, x2, y2 = pred["bbox"]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis_img, f"FP:{pred['label']}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            out_file = os.path.join(save_dir, f"{base_name}_analysis.jpg")
            cv2.imwrite(out_file, vis_img)

            per_image_stats.append(img_stats)

        # ------------------------
        # Save Stats
        # ------------------------
        df = pd.DataFrame(per_image_stats)
        csv_path = os.path.join(save_dir, "detection_stats.csv")
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Appended batch stats to {csv_path}")

        return class_stats




    def update_annotation_with_colour(self, batch_data_samples, anno_file, save_path, k=2, score_thr=0.3, iou_thresh = 0.7):

        """
        Update a copy of COCO annotation file by adding 'colour' attribute to annotations.
        Increments 'colour' by k for each ground truth box that is NOT matched by predictions
        (same matching rule as analyze_predictions).
        """
        # --- Step 1: Load or create copy of annotations ---
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                coco_data = json.load(f)
        else:
            with open(anno_file, "r") as f:
                coco_data = json.load(f)
            # Initialize colour field
            for ann in coco_data["annotations"]:
                ann["colour"] = 0

        coco_gt = COCO(anno_file)
        
        # Map image_id -> anns in modified file (we’ll edit these)
        anns_by_img = {}
        for ann in coco_data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # --- Step 2: Loop through batch and apply same matching as analyze_predictions ---
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB-xA) * max(0, yB-yA)
            boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
            boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        for data_sample in batch_data_samples:
            img_path = data_sample.img_path
            # Find img_id by filename
            img_id = None
            for i in coco_gt.getImgIds():
                info = coco_gt.loadImgs([i])[0]
                # print("info['file_name']", info['file_name'] )
                # print("os.path.basename(img_path)", os.path.basename(img_path))
                if info['file_name'] == os.path.basename(img_path):
                    img_id = i
                    break
            if img_id is None:
                print("yp")
                continue

            # Load GT anns (from original COCO)
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])

            anns = coco_gt.loadAnns(ann_ids)
            gt_anns = []
            if img_id is not None:
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                anns = coco_gt.loadAnns(ann_ids)
                for ann in anns:
                    x, y, w, h = map(int, ann["bbox"])
                    cat_name = coco_gt.loadCats([ann["category_id"]])[0]["name"]
                    gt_anns.append({"bbox":[x,y,x+w,y+h], "label":cat_name, "id":ann["id"]})

            # Predictions
            pred = data_sample.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.label_names if hasattr(pred, "label_names") else pred.labels.cpu().numpy()
            # print("bbx: ", len(bboxes), "score; ", scores, "labels: ", labels)
            pred_anns = []
            for bbox, score, label in zip(bboxes, scores, labels):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                pred_anns.append({"bbox":[x1,y1,x2,y2], "label":label, "score":score})
            # Matching (same as analyze_predictions)
            matched_gt = set()
            matched_pred = set()
            for gi, gt in enumerate(gt_anns):
                best_iou, best_pi = 0, None
                for pi, pred in enumerate(pred_anns):
                    if pi in matched_pred: 
                        continue
                    iou_val = iou(gt["bbox"], pred["bbox"])
                    if iou_val > best_iou:
                        best_iou, best_pi = iou_val, pi
                if best_pi is not None and (best_iou > iou_thresh) and pred_anns[best_pi]["label"] == gt["label"]:
                    print("label: ", gt["label"])
                    matched_gt.add(gi)
                    matched_pred.add(best_pi)

            # Now increment colour for unmatched GTs (FNs)
            # print("gt_anns" , gt_anns)
            for gi, gt in enumerate(gt_anns):
                if gi not in matched_gt:
                    # find the annotation in coco_data and bump its colour
                    for ann in anns_by_img[img_id]:
                        if ann["id"] == gt["id"]:
                            ann["colour"] += k
                            break

        # --- Step 3: Save updates ---
        with open(save_path, "w") as f:
            json.dump(coco_data, f, indent=2)



    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        # if self.training:
        #     return batch_data_samples
        

        """Save predictions, draw GT & Pred side-by-side.

        Args:
            batch_data_samples (list): Output from `self.predict(...)`.
            save_dir (str): Directory to save results.
            anno_file (str): Path to COCO-style annotation file.
            score_thr (float): Score threshold for predictions.
        """
        # save_dir = '/home/tushar/scratch/mmdetection/test_outputs/hdino'
        # anno_file = '/home/tushar/scratch/dataset/RSUD_dataset/annotations/instances_test2017.json'
        # score_thr = 0.3
        # iou_thr=0.5
        # self.update_annotation_with_colour(batch_data_samples, anno_file, save_path = save_dir + '/colour.json', k = 2,score_thr=0.3, iou_thresh=0.7)
        # self.analyze_predictions(batch_data_samples, save_dir,anno_file, score_thr)
        return batch_data_samples

