# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerEncoder
)

@MODELS.register_module()
class DINO(DeformableDETR):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, gdino_ratio = 0.0, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        

        super().__init__(*args, **kwargs)
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        # gdino_embed_dir = 'dino_embeddings' 
        # self.gdino_ratio = gdino_ratio
        # self.dino_cache = {} 

        # import os 
        # for fname in os.listdir(gdino_embed_dir): 
        #     if fname.endswith(".pt"): 
        #         key = os.path.splitext(fname)[0] # filename without extension 
        #         self.dino_cache[key] = torch.load( os.path.join(gdino_embed_dir, fname), map_location="cpu") 
                
                
        # import logging 
        # logging.basicConfig(level=logging.INFO) 
        # logger = logging.getLogger("Loading DINO embeddings") 
        # logger.info(f"[DINO] Loaded {len(self.dino_cache)} DINO embeddings into cache")  

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # for param in self.neck.parameters():
        #     param.requires_grad = False
        
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # for param in self.positional_encoding.parameters():
        #     param.requires_grad = False
        
        # self.level_embed.requires_grad = False

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)
        
    # def init_weights(self):

    #     super(DeformableDETR, self).init_weights()
    #     from mmengine import MMLogger

    #     logger = MMLogger.get_current_instance()

    #     # gdino_ckpt = torch.load(
    #     #     '/home/poonam_rajput/scratch/mmdetection/checkpoints/GDINO_swin-l_pretrained_rsud_best_mAP_epoch_19.pth',
    #     #     map_location='cpu'
    #     # )['state_dict']
    #     dino_ckpt = torch.load(
    #         '/home/poonam_rajput/scratch/mmdetection/checkpoints/DINO-swin-L_pretrained_rsud_best_epoch_16.pth',
    #         map_location='cpu'
    #     )['state_dict']
        
    #     # # -------- GDINO: encoder --------
    #     # if hasattr(self, "encoder"):
    #     #     gdino_encoder_weights = {
    #     #         k.replace("encoder.", ""): v
    #     #         for k, v in gdino_ckpt.items() if k.startswith("encoder.")
    #     #     }

    #     #     missing, unexpected = self.gdino_encoder.load_state_dict(gdino_encoder_weights, strict=True)
    #     #     # print(self.backbone.state_dict().keys())
    #     #     # print([k for k in gdino_ckpt.keys() if k.startswith("backbone.")])

    #     #     logger.info(f"[GDINO] Loaded encoder | missing={len(missing)}, unexpected={len(unexpected)}")

    #     # -------- DINO: backbone --------
    #     if hasattr(self, "backbone"):
    #         backbone_weights = {
    #             k.replace("backbone.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("backbone.")
    #         }

    #         missing, unexpected = self.backbone.load_state_dict(backbone_weights, strict=True)
    #         # print(self.backbone.state_dict().keys())
    #         # print([k for k in gdino_ckpt.keys() if k.startswith("backbone.")])

    #         logger.info(f"[DINO] Loaded backbone | missing={len(missing)}, unexpected={len(unexpected)}")


    #      # -------- DINO: neck --------
    #     if hasattr(self, "neck"):
    #         neck_weights = {
    #             k.replace("neck.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("neck.")
    #         }
    #         missing, unexpected = self.neck.load_state_dict(neck_weights, strict=True)
    #         logger.info(f"[DINO] Loaded neck | missing={len(missing)}, unexpected={len(unexpected)}")

    #     # -------- DINO: positional_encoding --------
    #     if hasattr(self, "positional_encoding"):
    #         positional_encoding_weights = {
    #             k.replace("positional_encoding.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("positional_encoding.")
    #         }

    #         missing, unexpected = self.positional_encoding.load_state_dict(positional_encoding_weights, strict=True)
    #         logger.info(f"[DINO] Loaded positional_encoding | missing={len(missing)}, unexpected={len(unexpected)}")

    #     # -------- DINO: level_embed --------
    #     if "level_embed" in dino_ckpt:
    #         with torch.no_grad():
    #             self.level_embed.copy_(dino_ckpt["level_embed"])
    #         logger.info("[DINO] Loaded level_embed from checkpoint")
    #     else:
    #         logger.warning("[DINO] level_embed not found in checkpoint — using random init")

    #     # -------- DINO: encoder --------
    #     if hasattr(self, "encoder"):
    #         encoder_weights = {
    #             k.replace("encoder.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("encoder.")
    #         }
    #         missing, unexpected = self.encoder.load_state_dict(encoder_weights, strict=True)
    #         logger.info(f"[DINO] Loaded encoder | missing={len(missing)}, unexpected={len(unexpected)}")

    #     if hasattr(self, "memory_trans_fc"):
    #         memory_trans_fc_weights = {
    #             k.replace("memory_trans_fc.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("memory_trans_fc.")
    #         }

    #         missing, unexpected = self.memory_trans_fc.load_state_dict(memory_trans_fc_weights, strict=True)
    #         logger.info(f"[DINO] Loaded memory_trans_fc | missing={len(missing)}, unexpected={len(unexpected)}")

    #     if hasattr(self, "memory_trans_norm"):
    #         memory_trans_norm_weights = {
    #             k.replace("memory_trans_norm.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("memory_trans_norm.")
    #         }

    #         missing, unexpected = self.memory_trans_norm.load_state_dict(memory_trans_norm_weights, strict=True)
    #         logger.info(f"[DINO] Loaded memory_trans_norm | missing={len(missing)}, unexpected={len(unexpected)}")

    #     # -------- DINO: decoder --------
    #     if hasattr(self, "decoder"):
    #         decoder_weights = {
    #             k.replace("decoder.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("decoder.")
    #         }
    #         missing, unexpected = self.decoder.load_state_dict(decoder_weights, strict=True)
    #         logger.info(f"[DINO] Loaded decoder | missing={len(missing)}, unexpected={len(unexpected)}")

    #     # -------- DINO: bbox_head --------
    #     if hasattr(self, "bbox_head"):
    #         head_weights = {
    #             k.replace("bbox_head.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("bbox_head.")
    #         }
    #         missing, unexpected = self.bbox_head.load_state_dict(head_weights, strict=False)
    #         logger.info(f"[DINO] Loaded bbox_head | missing={len(missing)}, unexpected={len(unexpected)}")

    #     if hasattr(self, "query_embedding"):
    #         query_embedding_weights = {
    #             k.replace("query_embedding.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("query_embedding.")
    #         }

    #         missing, unexpected = self.query_embedding.load_state_dict(query_embedding_weights, strict=True)
    #         logger.info(f"[DINO] Loaded query_embedding | missing={len(missing)}, unexpected={len(unexpected)}")

    #     if hasattr(self, "dn_query_generator"):
    #         dn_query_generator_weights = {
    #             k.replace("dn_query_generator.", ""): v
    #             for k, v in dino_ckpt.items() if k.startswith("dn_query_generator.")
    #         }

    #         missing, unexpected = self.dn_query_generator.load_state_dict(dn_query_generator_weights, strict=True)
    #         logger.info(f"[DINO] Loaded dn_query_generator | missing={len(missing)}, unexpected={len(unexpected)}")


    def choose_embedding(self, filename, dino_mem, gdino_mem): 
        import hashlib 
        # hash filename to a number between 0 and 1 
        h = int(hashlib.md5(filename.encode()).hexdigest(), 16) 
        frac = (h % 10000) / 10000.0  
        if frac < self.gdino_ratio: 
            return gdino_mem.squeeze(0) # use GDINO 
        else: 
            return dino_mem 
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        # with torch.no_grad():
        #     import os
        #     save_dir = 'dino_embeddings'
        #     for data_sample in batch_data_samples:
        #         # full path of the image
        #         img_path = data_sample.metainfo['img_path']

        #         # just the filename without extension
        #         filename = os.path.splitext(os.path.basename(img_path))[0]
        #         torch.save(encoder_outputs_dict, f"{save_dir}/{filename}.pt")
        #         import logging

        #         logging.basicConfig(level=logging.INFO)
        #         logger = logging.getLogger("DINO embedding")
        #         logger.info(f"Saved DINO embeddings for {filename}.pt")

        with torch.no_grad():  
            import os  
            if self.dino_cache is not None:  
                for i, data_sample in enumerate(batch_data_samples):  
                    img_path = data_sample.metainfo['img_path']  
                    filename = os.path.splitext(os.path.basename(img_path))[0]  
                    if filename not in self.dino_cache:  
                        raise KeyError(f"DINO embedding missing for {filename}")  
                    dino_embed = self.dino_cache[filename]
                    emb_online = encoder_outputs_dict['memory'][i]
                    emb_offline = dino_embed['memory'].squeeze(0).to(img_feats[0].device)
                    import torch.nn.functional as F
                    cos = F.cosine_similarity(emb_offline.flatten(), emb_online.flatten(), dim=0)
                    print(cos)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
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
