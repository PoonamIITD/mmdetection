# Copyright (c) OpenMMLab. All rights reserved.
# Hybrid model: GDINO Encoder + DINO Decoder
import copy
import torch
import os
import json
import cv2
import numpy as np
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
from mmcv.ops import MultiScaleDeformableAttention
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerEncoder
)
from ..layers.transformer.dino_layers import (
    DinoTransformerDecoder
)
from mmengine import MessageHub

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

# class MemoryFusion(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()
#         self.proj_gdino = nn.Linear(d_model, d_model)
#         self.gate = nn.Linear(d_model * 2, d_model)
#         self.activation = nn.Sigmoid()

#     def forward(self, detr_mem, gdino_mem):
#         """
#         detr_mem: [bs, 20k, 256]
#         gdino_mem: [bs, 20k, 256]
#         """
#         gdino_proj = self.proj_gdino(gdino_mem)  # project to DETR space

#         # gating based on concatenated features
#         gate_input = torch.cat([detr_mem, gdino_proj], dim=-1)
#         gate = self.activation(self.gate(gate_input))

#         # gated fusion
#         fused = detr_mem + gate * gdino_proj
#         return fused

# class MemoryFusion(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()
#         self.proj_gdino = nn.Linear(d_model, d_model)
#         self.ln_detr = nn.LayerNorm(d_model)
#         self.ln_gdino = nn.LayerNorm(d_model)
#         self.gate = nn.Linear(d_model * 2, d_model)
#         self.alpha = nn.Parameter(torch.tensor(0.5))   # learn how much GDINO to use
#         self.act = nn.Sigmoid()

#     def forward(self, detr_mem, gdino_mem):
#         gdino_p = self.proj_gdino(gdino_mem)

#         d = self.ln_detr(detr_mem)
#         g = self.ln_gdino(gdino_p)
        
#         z = torch.cat([d, g], dim=-1)
#         gate = self.act(self.gate(z))  # [bs, 20k, 256]

#         fused = detr_mem + self.alpha * gate * gdino_p
#         return fused

# class MemoryFusion(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()

#         # projections
#         self.proj_gdino = nn.Linear(d_model, d_model)

#         # norms
#         self.ln_detr = nn.LayerNorm(d_model)
#         self.ln_gdino = nn.LayerNorm(d_model)

#         # channel-wise gate
#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, d_model)
#         )

#         # ✅ token-wise alpha (scalar per token)
#         self.alpha = nn.Sequential(
#             nn.Linear(d_model * 2, 1),
#             nn.Sigmoid()
#         )

#         # light refinement for GDINO memory
#         self.gdino_ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, detr_mem, gdino_mem, detr_conf=None):
#         """
#         detr_mem  : [bs, N, C]
#         gdino_mem : [bs, N, C]
#         detr_conf : [bs, N, 1]  (from DETR pre-decoder)
#         """

#         # project + refine GDINO memory
#         gdino_p = self.proj_gdino(gdino_mem)
#         gdino_p = gdino_p + self.gdino_ffn(gdino_p)

#         # normalize
#         d = self.ln_detr(detr_mem)
#         g = self.ln_gdino(gdino_p)

#         # joint representation
#         z = torch.cat([d, g], dim=-1)  # [bs, N, 2C]

#         # gating
#         gate = torch.sigmoid(self.gate(z))     # [bs, N, C]

#         # token-wise alpha
#         alpha = self.alpha(z)                  # [bs, N, 1]

#         # confidence-aware suppression (DETR confidence)
#         if detr_conf is not None:
#             conf_weight = (1.0 - detr_conf.sigmoid()).detach()
#             alpha = alpha * conf_weight        # broadcast safe

#         # fusion
#         fused = detr_mem + alpha * gate * gdino_p
#         return fused

class MemoryDeformableFusion(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_levels=5,
        n_points=4,
    ):
        super().__init__()

        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            batch_first=True,
        )

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        # gated residual fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # token-wise alpha
        self.alpha = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        detr_mem: torch.Tensor,          # [bs, Nq, C]
        gdino_mem: torch.Tensor,         # [bs, Nmem, C]
        # reference_points: torch.Tensor,  # [bs, Nq, 4] (cx, cy, w, h) UNACT or ACT
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        # detr_conf: torch.Tensor,         # [bs, Nq, 1] (NOT detached)
        key_padding_mask=None,
    ):
        """
        Decoder-aligned deformable fusion:
        queries = detr_mem
        values  = gdino_mem
        anchors = reference_points (from encoder proposals)
        """
        
        # message_hub = MessageHub.get_current_instance()
        # iter_ = message_hub.get_info('iter')

        # fusion_scale = 0.0 if (self.training and iter_ is not None and iter_ < 1000) else 1.0

        # normalize
        q = self.ln_q(detr_mem)
        kv = self.ln_kv(gdino_mem)

        # decoder-style reference points (use centers only)
        # ref_points = reference_points[..., :2].unsqueeze(2)
        # [bs, Nq, 1, 2]
        reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, detr_mem.device ) 

        # deformable cross-attention
        fused_feat = self.cross_attn(
            query=q,
            value=kv,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )

        fused_feat = self.ln_out(fused_feat)

        # gated residual fusion
        z = torch.cat([detr_mem, fused_feat], dim=-1)
        gate = self.gate(z)
        alpha = self.alpha(z)

        # # confidence-aware modulation (learnable)
        # conf_weight = (1.0 - detr_conf.sigmoid()).clamp(min=0.1)
        # alpha = alpha * conf_weight

        return detr_mem + alpha * gate * fused_feat
    
    @staticmethod
    def get_encoder_reference_points(
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        device: Union[torch.device, str],
    ) -> Tensor:
        """
        Reference points for encoder-style deformable attention.

        Returns:
            reference_points: [bs, sum(hw), num_levels, 2]
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=device),
                torch.linspace(0.5, W - 0.5, W, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W
            )
            ref = torch.stack((ref_x, ref_y), dim=-1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    



# class MemoryFusion(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()
#         self.proj_gdino = nn.Linear(d_model, d_model)
        
#         self.norm = nn.LayerNorm(d_model)

#         # gating
#         self.token_gate = nn.Linear(d_model * 2, d_model)
#         self.channel_gate = nn.Linear(d_model * 2, d_model)

#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.act = nn.Sigmoid()

#     def forward(self, detr_mem, gdino_mem):
#         gdino_p = self.proj_gdino(gdino_mem)

#         d = self.norm(detr_mem)
#         g = self.norm(gdino_p)

#         z = torch.cat([d, g], dim=-1)

#         tgate = self.act(self.token_gate(z))
#         cgate = self.act(self.channel_gate(z))

#         fused = detr_mem + self.alpha * tgate * cgate * gdino_p
#         return fused

# class MemoryDeformableCrossAttn(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         n_heads=8,
#         n_levels=5,
#         n_points=4,
#     ):
#         super().__init__()

#         self.attn = MultiScaleDeformableAttention(
#             embed_dims=d_model,
#             num_levels=n_levels,
#             num_heads=n_heads,
#             num_points=n_points,
#             batch_first=True,
#         )


#     @staticmethod
#     def get_encoder_reference_points(
#             spatial_shapes: Tensor, valid_ratios: Tensor,
#             device: Union[torch.device, str]) -> Tensor:
#         """Get the reference points used in encoder.

#         Args:
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#             device (obj:`device` or str): The device acquired by the
#                 `reference_points`.

#         Returns:
#             Tensor: Reference points used in decoder, has shape (bs, length,
#             num_levels, 2).
#         """

#         reference_points_list = []
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(
#                     0.5, H - 0.5, H, dtype=torch.float32, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 1] * H)
#             ref_x = ref_x.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 0] * W)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         # [bs, sum(hw), num_level, 2]
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points

#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, spatial_shapes: Tensor,
#                 level_start_index: Tensor, valid_ratios: Tensor,
#                 **kwargs)-> Tensor:

#         reference_points = self.get_encoder_reference_points(
#             spatial_shapes, valid_ratios, device=query.device)
#         query = self.attn(
#                 query=query,
#                 query_pos=query_pos,
#                 key_padding_mask=key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points,
#                 **kwargs)
#         return query

@MODELS.register_module()
class HybridDINO(DINO):
    """Hybrid model: GroundingDINO Encoder + DINO Decoder.

    Encoder: multimodal fusion from GroundingDINO
    Decoder: vanilla DINO decoder
    """

    def __init__(self,
                 language_model,
                 gdino_encoder,
                 dino_ratio=0.0,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:
        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.gdino_encoder = gdino_encoder
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Use GroundingDINO encoder and DINO decoder."""
        # Positional encoding
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        # Encoder from GroundingDINO
        self.gdino_encoder = GroundingDinoTransformerEncoder(**self.gdino_encoder)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        # Decoder from DINO
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims

        # Query embedding as in DINO
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # self.memory_fusion = MemoryFusion(self.embed_dims)
        
        # self.gdino_ref_deformer = MemoryDeformableCrossAttn(
        #     d_model=self.embed_dims,  
        #     n_levels=self.num_feature_levels
        # )
        # self.memory_fusion = MemoryFusion(d_model=self.embed_dims)
        # self.memory_fusion = CrossAttnFusionPerLevel(d_model=self.embed_dims)

        self.gdino_ref_deformer = MemoryDeformableFusion(d_model=self.embed_dims, n_levels=self.num_feature_levels)

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
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.neck.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

        for param in self.text_feat_map.parameters():
            param.requires_grad = False
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.gdino_encoder.parameters():    
            param.requires_grad = False

        for param in self.positional_encoding.parameters():
            param.requires_grad = False
        
        self.level_embed.requires_grad = False


    def init_weights(self):
        """Initialize HybridDINO with GDINO encoder + DINO decoder."""
        super(DeformableDETR, self).init_weights()
        from mmengine import MMLogger

        logger = MMLogger.get_current_instance()

        # ---------- Load checkpoints (CPU only) ----------
        with torch.no_grad():
            gdino_ckpt = torch.load(
                # '/home/poonam_rajput/scratch/mmdetection/checkpoints/GDINO_swin-l_uvh_best_coco_bbox_mAP_epoch_19.pth',
                '/home/poonam_rajput/scratch/mmdetection/checkpoints/GDINO_swin-l_pretrained_rsud_best_mAP_epoch_19.pth',
                map_location='cpu'
            )['state_dict']

            dino_ckpt = torch.load(
                # '/home/poonam_rajput/scratch/mmdetection/checkpoints/DINO-swin-l_uvh_best_coco_bbox_mAP_epoch_20.pth',
                '/home/poonam_rajput/scratch/mmdetection/checkpoints/DINO-swin-L_pretrained_rsud_best_epoch_16.pth',
                map_location='cpu'
            )['state_dict']

        # ---------- Helper function ----------
        def load_module(prefix, module, ckpt, strict=True, allow_missing=False):
            with torch.no_grad():
                weights = {
                    k.replace(prefix + ".", ""): v
                    for k, v in ckpt.items() if k.startswith(prefix)
                }
                missing, unexpected = module.load_state_dict(
                    weights, strict=strict)
                logger.info(
                    f"[LOAD] {prefix} | missing={len(missing)} unexpected={len(unexpected)}"
                )

        # ---------- GDINO loads ----------
        load_module("text_feat_map", self.text_feat_map, gdino_ckpt)
        load_module("language_model", self.language_model, gdino_ckpt)
        load_module("backbone", self.backbone, gdino_ckpt)
        load_module("neck", self.neck, gdino_ckpt)
        load_module("positional_encoding", self.positional_encoding, gdino_ckpt)
        load_module("encoder", self.gdino_encoder, gdino_ckpt)

        # level_embed (special case)
        if "level_embed" in gdino_ckpt:
            with torch.no_grad():
                self.level_embed.copy_(gdino_ckpt["level_embed"])

        # ---------- DINO loads ----------
        load_module("encoder", self.encoder, dino_ckpt)
        load_module("decoder", self.decoder, dino_ckpt)
        load_module("query_embedding", self.query_embedding, dino_ckpt)

        # bbox_head and dn_query_generator allow missing keys
        if hasattr(self, "bbox_head"):
            load_module("bbox_head", self.bbox_head, dino_ckpt, strict=True)

        if hasattr(self, "dn_query_generator"):
            load_module("dn_query_generator", self.dn_query_generator,
                        dino_ckpt, strict=True)

        # ---------- Free checkpoint memory ----------
        del gdino_ckpt
        del dino_ckpt
        import gc
        gc.collect()
        torch.cuda.empty_cache()


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
        gdino_memory, memory_text = self.gdino_encoder(
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

        memory = self.encoder(  #dino encoder
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        # enhanced_memory = self.memory_fusion(memory, gdino_memory)
        # enhanced_memory = self.memory_fusion(memory, gdino_memory,spatial_shapes)

        # enhanced_memory = self.gdino_ref_deformer( 
        #     query=gdino_memory,
        #     query_pos=memory,
        #     key_padding_mask=feat_mask,  # for self_attn
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     valid_ratios=valid_ratios)
        
        return dict(
            memory=memory,
            # image_value=enhanced_memory,
            image_value = gdino_memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

    def pre_decoder(
        self,
        memory: Tensor,
        image_value: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
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
        
        # visual objectness confidence (no GDINO head needed)
        # visual_conf = enc_outputs_class.max(dim=-1, keepdim=True).values
        # visual_conf = visual_conf.detach()  # VERY important

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
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
        # ref_points = enc_outputs_coord_unact.sigmoid().detach()

        enhanced_memory = self.gdino_ref_deformer(
            detr_mem=memory,
            gdino_mem=image_value,
            # reference_points=ref_points,
            key_padding_mask=memory_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )
        decoder_inputs_dict = dict(
            query=query,
            # memory=image_value,
            memory=enhanced_memory,
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

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        
        return losses


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
        
        return batch_data_samples