# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)

CLASS_DESCRIPTOR_PROMPTS = {
    "person": [
        "A full body view of a person",
        "A person with head partially occluded",
        "A person showing only upper body silhouette between objects",
        "A person with only legs or torso visible due to obstruction",
    ],
    "rickshaw": [
        "A rickshaw fully visible on the road",
        "A rickshaw with rear passenger seat partially blocked",
        "A rickshaw with curved back canopy visible and front occluded",
        "A rickshaw with decorated rear structure partially hidden in traffic",
    ],
    "rickshaw van": [
        "A rickshaw van fully visible carrying cargo",
        "A rickshaw van with cargo area partially occluded",
        "A rickshaw van with open rear platform visible but driver section hidden",
        "A rickshaw van with metal frame partially blocked",
    ],
    "auto rickshaw": [
        "A clear front or side view of an auto rickshaw",
        "An auto rickshaw with canopy partially visible",
        "An auto rickshaw with front windshield visible and sides occluded",
        "An auto rickshaw with partial rear or side visibility",
    ],
    "truck": [
        "A large truck clearly visible from the side",
        "A truck with cargo container partially occluded",
        "A truck with visible wheels or container edges but hidden cabin",
        "A truck with large rectangular body partially visible",
    ],
    "pickup truck": [
        "A clear side view of a pickup truck",
        "A pickup truck with open rear bed partially occluded",
        "A pickup truck with front cabin visible and cargo hidden",
        "A pickup truck partially visible behind other vehicles",
    ],
    "private car": [
        "A clear front or side view of a private car",
        "A private car with roof or windshield visible but body occluded",
        "A private car with front or rear blocked by traffic",
        "A private car with only lights or mirrors visible",
    ],
    "motorcycle": [
        "A motorcycle clearly visible with rider",
        "A motorcycle with rider partially visible and occluded",
        "A motorcycle with handlebar and front wheel visible",
        "A motorcycle with rider silhouette partially hidden",
    ],
    "bicycle": [
        "A clear view of a bicycle from the side",
        "A bicycle with only wheels or frame partially visible",
        "A bicycle with thin frame seen through gaps",
        "A bicycle with only front or rear wheel visible",
    ],
    "bus": [
        "A large bus clearly visible from the side",
        "A bus with windows or side panels partially occluded",
        "A bus with only upper portion visible",
        "A bus with front or side blocked",
    ],
    "micro bus": [
        "A micro bus fully visible on the road",
        "A micro bus with windows partially visible",
        "A micro bus with roof and side windows visible but lower part hidden",
        "A micro bus partially blocked in dense traffic",
    ],
    "covered van": [
        "A covered van clearly visible from the side",
        "A covered van with enclosed rear partially visible",
        "A covered van with rectangular cargo body partially visible",
        "A covered van hidden behind other objects",
    ],
    "human hauler": [
        "A human Hauler that is a small passenger carrier clearly visible with open sides",
        "A human hauler with open side seating partially visible",
        "A human hauler with metal frame and canopy occluded",
        "A human hauler partially blocked with covered roof and open sides",
    ],
}



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
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)
        self.Ks = [300, 200, 100, 50, 30, 20, 10]

        self.total_matches_cls = {K: 0 for K in self.Ks}
        self.total_matches_desc = {K: 0 for K in self.Ks}
        self.total_samples = 0

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

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
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
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

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict, desc_encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict, desc_encoder_outputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        aug_text_token_mask = text_dict['text_token_mask']
        #Hardcoded
        T = 39
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~aug_text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        
        memory_text_full = memory_text  # output of encoder
        # split
        memory_text = memory_text_full[:, :T, :]
        desc_memory_text = memory_text_full[:, T:, :]

        # masks
        text_token_mask = aug_text_token_mask[:, :T]
        desc_token_mask = aug_text_token_mask[:, T:]

        desc_encoder_outputs_dict = dict(
            desc_memory_text= desc_memory_text,
            desc_token_mask = desc_token_mask
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict, desc_encoder_outputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

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

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

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
                                                    batch_data_samples)

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

            # Code added for descriptor prompts
            all_embeddings = []
            all_masks = []
            class_embed_list = []
            start_idx_global = 0
            descriptor_ranges_per_class = []

            bs = len(text_prompts)

            for cls_name, desc_list in CLASS_DESCRIPTOR_PROMPTS.items():

                # =========================
                #  STEP 1: CLASS EMBEDDING
                # =========================
                text_dict_cls = self.language_model([cls_name])

                if self.text_feat_map is not None:
                    text_dict_cls['embedded'] = self.text_feat_map(text_dict_cls['embedded'])

                cls_emb = text_dict_cls['embedded']          # [1, t, C]
                cls_mask = text_dict_cls['text_token_mask']  # [1, t]

                # mean pooling (VERY IMPORTANT)
                cls_emb = (cls_emb * cls_mask.unsqueeze(-1)).sum(dim=1) / cls_mask.sum(dim=1, keepdim=True)

                class_embed_list.append(cls_emb)  # [1, C]

                # =========================
                # STEP 2: DESCRIPTORS
                # =========================
                descriptor_embeds = []
                descriptor_masks = []
                descriptor_ranges = []

                for desc in desc_list:
                    text_dict_i = self.language_model([desc])

                    if self.text_feat_map is not None:
                        text_dict_i['embedded'] = self.text_feat_map(text_dict_i['embedded'])

                    emb = text_dict_i['embedded']         # [1, t_i, C]
                    mask = text_dict_i['text_token_mask'] # [1, t_i]

                    t_i = mask.shape[1]  # correct token length

                    emb = emb.repeat(bs, 1, 1)
                    mask = mask.repeat(bs, 1)

                    descriptor_embeds.append(emb)
                    descriptor_masks.append(mask)
                    descriptor_ranges.append((start_idx_global, start_idx_global + t_i))
                    start_idx_global += t_i

                # concat descriptors of this class
                class_wise_descriptor_embeds = torch.cat(descriptor_embeds, dim=1)   # [bs, Mk, C]
                class_wise_descriptor_masks = torch.cat(descriptor_masks, dim=1)     # [bs, Mk]
                
                all_embeddings.append(class_wise_descriptor_embeds)
                all_masks.append(class_wise_descriptor_masks)
                
                descriptor_ranges_per_class.append(descriptor_ranges)

            # =========================
            # FINAL OUTPUTS
            # =========================
            descriptor_embedded = torch.cat(all_embeddings, dim=1)  # [bs, total_tokens, C]
            descriptor_mask = torch.cat(all_masks, dim=1)           # [bs, total_tokens]

            class_embeds = torch.cat(class_embed_list, dim=0)       # [K, C]
            descriptor_ranges_per_class
            
            memory_text= text_dict['embedded']
            text_token_mask = text_dict['text_token_mask']
            device = memory_text.device
            # concat
            augmented_text = torch.cat([memory_text, descriptor_embedded], dim=1)
            aug_text_token_mask = torch.cat([text_token_mask, descriptor_mask], dim=1)

            # sizes
            bs, T, _ = memory_text.shape
            D = descriptor_embedded.shape[1]
            N = T + D

            # init
            new_mask = torch.zeros(bs, N, N, dtype=torch.bool, device=device)

            # preserve text structure
            new_mask[:, :T, :T] = text_token_mask

            # positive map
            pos_map = token_positive_maps[0]
            text_indices_per_class = {
                cls_id: torch.tensor(token_ids, device=device)
                for cls_id, token_ids in pos_map.items()
            }

            # descriptor indices
            desc_indices_per_class = []
            for class_ranges in descriptor_ranges_per_class:
                indices = []
                for (s, e) in class_ranges:
                    indices.extend(range(T + s, T + e))
                desc_indices_per_class.append(torch.tensor(indices, device=device))

            # align text ↔ descriptor
            for i, (cls_id, text_indices) in enumerate(sorted(text_indices_per_class.items())):
                desc_indices = desc_indices_per_class[i]

                new_mask[:, desc_indices[:, None], text_indices[None, :]] = 1
                new_mask[:, text_indices[:, None], desc_indices[None, :]] = 1

            # descriptor intra-class attention
            for indices in desc_indices_per_class:
                new_mask[:, indices[:, None], indices[None, :]] = 1

            # ensure self-attention
            diag_idx = torch.arange(N, device=device)
            new_mask[:, diag_idx, diag_idx] = 1

            full_mask = aug_text_token_mask 
            new_mask = new_mask & full_mask[:, :, None] & full_mask[:, None, :]

            #Position_ids : Class Aware positions. This encodes; intra-descriptor order and inter-class separation
            descriptor_position_ids = torch.zeros(bs, D, device=device)
            for class_id, class_ranges in enumerate(descriptor_ranges_per_class):
                for (s, e) in class_ranges:
                    length = e - s
                    descriptor_position_ids[:, s:e] = (
                        torch.arange(length, device=device) + class_id * 100
                    )
            position_ids_aug = torch.cat(
                [text_dict['position_ids'], descriptor_position_ids],
                dim=1
            )

            text_dict = {
                'embedded': augmented_text,
                'text_token_mask': aug_text_token_mask,
                'position_ids': position_ids_aug,
                'masks': new_mask
            }

            head_inputs_dict, desc_encoder_output = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            descriptor_embedded = desc_encoder_output['desc_memory_text']
            descriptor_mask = desc_encoder_output['desc_token_mask']
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples,
                descriptor_embedded=descriptor_embedded,
                descriptor_mask=descriptor_mask,
                # class_embeds=class_embeds,
                descriptor_ranges_per_class=descriptor_ranges_per_class)

        for i, pred_instances in enumerate(results_list):

            if len(pred_instances) == 0:
                continue

            scores = pred_instances.scores
            desc_conf = pred_instances.desc_conf

            pred_labels = pred_instances.labels
            desc_labels = pred_instances.desc_labels

            gt_labels = batch_data_samples[i].gt_instances.labels

            sorted_idx_cls = scores.argsort(descending=True)
            sorted_idx_desc = desc_conf.argsort(descending=True)

            pred_labels_sorted = pred_labels[sorted_idx_cls]
            desc_labels_sorted = desc_labels[sorted_idx_desc]

            for K in self.Ks:

                K_eff = min(K, len(pred_labels_sorted))

                topk_cls = pred_labels_sorted[:K_eff]
                topk_desc = desc_labels_sorted[:K_eff]

                matches_cls = 0
                matches_desc = 0

                used_gt_cls = set()
                used_gt_desc = set()

                # class-based
                for p in topk_cls:
                    for j, gt in enumerate(gt_labels):
                        if p == gt and j not in used_gt_cls:
                            matches_cls += 1
                            used_gt_cls.add(j)
                            break

                # descriptor-based
                for p in topk_desc:
                    for j, gt in enumerate(gt_labels):
                        if p == gt and j not in used_gt_desc:
                            matches_desc += 1
                            used_gt_desc.add(j)
                            break

                self.total_matches_cls[K] += matches_cls
                self.total_matches_desc[K] += matches_desc

            self.total_samples += 1

        if(self.total_samples == 20):
            for K in self.Ks:
                precision = self.total_matches_cls[K] / (K * self.total_samples)
                avg_matches = self.total_matches_cls[K] / self.total_samples
                print(f"Top-{K}: Avg Matches for Contrastive Embed score = {avg_matches:.2f} | Precision = {precision:.4f}")
                precision = self.total_matches_desc[K] / (K * self.total_samples)
                avg_matches = self.total_matches_desc[K] / self.total_samples
                print(f"Top-{K}: Avg Matches for WCA score = {avg_matches:.2f} | Precision = {precision:.4f}")
        
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
