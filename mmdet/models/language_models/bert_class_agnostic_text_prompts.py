# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape

    special_tokens_mask = torch.zeros(
        (bs, num_token), device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = torch.nonzero(special_tokens_mask)

    attention_mask = (
        torch.eye(num_token, device=input_ids.device)
        .bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row,
                           previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row,
                         previous_col + 1:col + 1] = torch.arange(
                             0, col - previous_col,
                             device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

def insert_prompt_attention_mask(attention_mask, prompt_length):
    """
    Insert prompt tokens after CLS token in attention_mask.

    attention_mask:
        - [bs, L]      (standard BERT)
        - [bs, L, L]   (GDINO sub-sentence mask)

    returns:
        new_attention_mask with prompts inserted
    """
    bs = attention_mask.size(0)
    P = prompt_length
    device = attention_mask.device
    dtype = attention_mask.dtype

    if attention_mask.dim() == 2:
        # ----------------------------------
        # Case 1: [bs, L]
        # ----------------------------------
        prompt_mask = torch.ones(bs, P, device=device, dtype=dtype)

        # CLS | PROMPTS | TOKENS
        return torch.cat(
            [attention_mask[:, :1],
             prompt_mask,
             attention_mask[:, 1:]],
            dim=1
        )

    elif attention_mask.dim() == 3:
        # ----------------------------------
        # Case 2: [bs, L, L]
        # ----------------------------------
        _, L, _ = attention_mask.shape
        new_L = L + P

        new_mask = torch.zeros(
            bs, new_L, new_L,
            device=device,
            dtype=dtype
        )

        # ---- 1. CLS → CLS
        new_mask[:, 0, 0] = attention_mask[:, 0, 0]

        # ---- 2. Original tokens (excluding CLS)
        new_mask[:, 1+P:, 1+P:] = attention_mask[:, 1:, 1:]

        # ---- 3. Prompt ↔ prompt + CLS (CONTROL-ONLY)
        new_mask[:, 1:1+P, 0:1+P] = 1   # prompts attend to CLS + prompts
        new_mask[:, 0:1+P, 1:1+P] = 1   # CLS attends to prompts

        return new_mask

    else:
        raise ValueError(
            f"Unsupported attention_mask dim: {attention_mask.dim()}"
        )

def insert_prompt_token_type_ids(token_type_ids, position_ids, prompt_length):
    bs = token_type_ids.size(0)
    P = prompt_length
    device = token_type_ids.device

    prompt_token_type_ids = torch.zeros(
        bs, P,
        dtype=token_type_ids.dtype,
        device=device
    )

    # Insert prompts after CLS (index 1)
    token_type_ids = torch.cat(
        [
            token_type_ids[:, :1],
            prompt_token_type_ids,
            token_type_ids[:, 1:]
        ],
        dim=1
    )

    prompt_position_ids = torch.zeros(
        bs, P, dtype=position_ids.dtype, device=device
    )

    position_ids = torch.cat(
        [
            position_ids[:, :1],    # CLS (0)
            prompt_position_ids,    # prompts (0)
            position_ids[:, 1:]     # original phrase-local ids
        ],
        dim=1
    )

    return token_type_ids, position_ids

def insert_prompt_text_token_mask(text_token_mask, prompt_length):
    """
    text_token_mask: [bs, L] (from tokenizer.attention_mask)
    returns: [bs, L + P]
    """
    bs = text_token_mask.size(0)
    P = prompt_length
    device = text_token_mask.device

    # Prompts are NOT real text tokens
    prompt_mask = torch.zeros(bs, P, device=device, dtype=text_token_mask.dtype)

    return torch.cat(
        [
            text_token_mask[:, :1],  # CLS
            prompt_mask,             # PROMPTS
            text_token_mask[:, 1:]   # original tokens
        ],
        dim=1
    )


@MODELS.register_module()
class BertModel(BaseModel):
    """BERT language encoder with soft prompt tuning (GDINO-compatible)."""

    def __init__(self,
                 name='bert-base-uncased',
                 max_tokens=256,
                 pad_to_max=True,
                 use_sub_sentence_represent=False,
                 special_tokens_list=None,
                 add_pooling_layer=False,
                 num_layers_of_embedded=1,
                 use_checkpoint=False,
                 prompt_length=8,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.prompt_length = prompt_length
        self.embed_dim = 768

        if AutoTokenizer is None:
            raise RuntimeError('Please install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self.language_backbone = nn.Sequential(
            OrderedDict([(
                'body',
                BertEncoder(
                    name,
                    add_pooling_layer=add_pooling_layer,
                    num_layers_of_embedded=num_layers_of_embedded,
                    use_checkpoint=use_checkpoint
                )
            )])
        )

        # 🔥 Soft text prompts
        self.text_prompts = nn.Parameter(
            torch.randn(prompt_length, self.embed_dim) * 0.02
        )

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None
            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        device = next(self.language_backbone.parameters()).device

        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True
        ).to(device)

        input_ids = tokenized.input_ids

        if self.use_sub_sentence_represent:
            attention_mask, position_ids = (
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens))
            token_type_ids = tokenized.token_type_ids
        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None

        # ---- PROMPT INSERTION ----
        bert = self.language_backbone.body.model
        embed_layer = bert.embeddings.word_embeddings

        input_embeds = embed_layer(input_ids)  # [bs, seq, 256]
        bs = input_embeds.size(0)

        prompt_embeds = self.text_prompts.unsqueeze(0).expand(bs, -1, -1)

        cls_embed = input_embeds[:, :1, :]
        rest_embed = input_embeds[:, 1:, :]

        input_embeds = torch.cat(
            [cls_embed, prompt_embeds, rest_embed], dim=1)

        attention_mask = insert_prompt_attention_mask(attention_mask, self.prompt_length)
        if token_type_ids is not None:
            token_type_ids, position_ids = insert_prompt_token_type_ids(token_type_ids, position_ids, self.prompt_length)
        
        tokenizer_input = {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }

        language_dict_features = self.language_backbone(tokenizer_input)

        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids

            text_token_mask = tokenized.attention_mask.bool()
            text_token_mask = insert_prompt_text_token_mask(
                text_token_mask, self.prompt_length
            )

        language_dict_features['text_token_mask'] = text_token_mask

        return language_dict_features


class BertEncoder(nn.Module):
    """Frozen BERT encoder."""

    def __init__(self,
                 name,
                 add_pooling_layer=False,
                 num_layers_of_embedded=1,
                 use_checkpoint=False):
        super().__init__()

        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint

        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)

        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

        # # 🔒 Freeze BERT
        # for p in self.model.parameters():
        #     p.requires_grad = False

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x.get('input_ids', None),
            inputs_embeds=x.get('inputs_embeds', None),
            attention_mask=mask,
            position_ids=x['position_ids'],
            token_type_ids=x['token_type_ids'],
            output_hidden_states=True
        )

        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(
            encoded_layers[-self.num_layers_of_embedded:], 1).mean(1)

        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        return {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
