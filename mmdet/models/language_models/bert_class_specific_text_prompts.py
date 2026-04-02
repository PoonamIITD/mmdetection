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
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

def pad_embed(x, max_len):
    out = x.new_zeros((max_len, x.size(1)))
    out[:x.size(0)] = x
    return out


def pad_mask(x, max_len):
    """Pad 1D attention mask to max_len"""
    out = x.new_zeros(max_len)
    out[:x.size(0)] = x
    return out


def pad_2d_mask(x, max_len):
    """Pad 2D attention mask to max_len x max_len"""
    out = x.new_zeros((max_len, max_len))
    out[:x.size(0), :x.size(1)] = x
    return out

def create_text_token_mask_with_prompts(orig_text_token_mask, phrase_starts, prompt_length, max_len, device):
    """
    Create text_token_mask for sequences with class-specific prompts inserted.
    Prompts are NOT real text tokens, so they get False in the mask.
    
    Args:
        orig_text_token_mask: [L] - original mask from tokenizer (1 for real tokens, 0 for padding)
        phrase_starts: list of indices where phrases start (where prompts are inserted before)
        prompt_length: number of prompt tokens per class
        max_len: final padded length
        device: torch device
    
    Returns:
        new_mask: [max_len] - mask with False for prompts, True for real text tokens
    
    Example:
        Original: [CLS] word1 word2 . word3 word4 . [SEP] [PAD]
        Mask:       1     1     1   1   1     1   1   1     0
        
        With prompts: [CLS] . [P P P] word1 word2 . [P P P] word3 word4 . [SEP] [PAD]
        New mask:       1   1   0 0 0    1     1   1   0 0 0    1     1   1   1     0
    """
    orig_len = orig_text_token_mask.size(0)
    
    # Build new mask
    new_mask_list = []
    new_mask_list.append(orig_text_token_mask[0:1])  # CLS
    
    cursor = 1
    for start in phrase_starts:
        # Add mask for tokens before phrase (separator)
        if start > cursor:
            new_mask_list.append(orig_text_token_mask[cursor:start])
        
        # Add zeros for prompts (prompts are NOT real text tokens)
        prompt_mask = torch.zeros(prompt_length, device=device, dtype=orig_text_token_mask.dtype)
        new_mask_list.append(prompt_mask)
        
        cursor = start
    
    # Add remaining tokens (phrase tokens + SEP + padding)
    new_mask_list.append(orig_text_token_mask[cursor:])
    
    # Concatenate
    new_mask = torch.cat(new_mask_list, dim=0)
    
    # Pad to max_len
    if new_mask.size(0) < max_len:
        padding = torch.zeros(max_len - new_mask.size(0), device=device, dtype=orig_text_token_mask.dtype)
        new_mask = torch.cat([new_mask, padding], dim=0)
    
    return new_mask

@MODELS.register_module()
class BertModel(BaseModel):
    """BERT language encoder with soft prompt tuning (GDINO-compatible)."""

    def __init__(self,
                 name='bert-base-uncased',
                 max_tokens=256,
                 pad_to_max=True,
                 use_sub_sentence_represent=False,
                 special_tokens_list=None,
                 num_classes=13,
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
        self.num_classes = num_classes

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
            torch.randn(self.num_classes, self.prompt_length, self.embed_dim) * 0.02
        )

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None
            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """
        Example structure:
        
        Original: [CLS] word1 word2 . word3 word4 word5 . word6 word7 . [SEP]
        Position:   0     0     1   2   0     1     2   3   0     1   2   0
        
        With prompts (P=3):
        Tokens:   [CLS] . [P0 P1 P2] word1 word2 . [P0 P1 P2] word3 word4 word5 . [P0 P1 P2] word6 word7 . [SEP]
        Position:   0   0   0  0  0    0     1   2   0  0  0    0     1     2   3   0  0  0    0     1   2   0
        
        All prompts get position 0, phrase tokens continue from 0, 1, 2, ... (positions reset at each separator)
        """
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

        input_embeds = embed_layer(input_ids)  # [bs, seq, 768]
        bs = input_embeds.size(0)

        new_embeds = []
        new_attention_masks = []
        new_position_ids = []
        new_token_type_ids = []
        new_text_token_masks = []

        for b in range(bs):
            ids_b = input_ids[b]                  # [L]
            embeds_b = input_embeds[b]            # [L, C]

            # --------------------------------------------------
            # 1. Find phrase boundaries
            # --------------------------------------------------
            dot_id = self.tokenizer.convert_tokens_to_ids('.')
            sep_id = self.tokenizer.sep_token_id

            phrase_starts = []
            prev = 0
            for i, tid in enumerate(ids_b.tolist()):
                if tid in (dot_id, sep_id):
                    if i > prev + 1:
                        phrase_starts.append(prev + 1)
                    prev = i

            # --------------------------------------------------
            # 2. Build new embeddings
            # --------------------------------------------------
            out = []
            out.append(embeds_b[:1])     # CLS: [1, C]

            cursor = 1
            phrase_idx = 0

            for start in phrase_starts:
                if start > cursor:
                    out.append(embeds_b[cursor:start])   # tokens before phrase
                
                prompt = self.text_prompts[phrase_idx]   # [P, C]
                out.append(prompt)
                phrase_idx += 1
                cursor = start

            out.append(embeds_b[cursor:])  # remaining tokens + SEP
            embeds_new = torch.cat(out, dim=0)
            new_embeds.append(embeds_new)

            # --------------------------------------------------
            # 3. Update attention_mask
            # --------------------------------------------------
            if self.use_sub_sentence_represent:
                # attention_mask is [L, L] - 2D causal/bidirectional mask
                orig_mask = attention_mask[b]  # [L, L]
                orig_len = ids_b.size(0)
                new_len = embeds_new.size(0)
                new_mask = torch.zeros((new_len, new_len), 
                                      device=device, dtype=torch.bool)
                
                # ---- Step 1: Build position mapping (old → new) ----
                old_to_new = {}
                new_idx = 0
                old_to_new[0] = 0  # CLS
                new_idx = 1
                
                # Track where each prompt set is inserted
                prompt_ranges = []  # [(start_idx, end_idx, phrase_start_old, phrase_end_old)]
                
                cursor = 1
                phrase_idx = 0
                for start in phrase_starts:
                    # tokens before phrase
                    for old_pos in range(cursor, start):
                        old_to_new[old_pos] = new_idx
                        new_idx += 1
                    
                    # Record prompt range and associated phrase
                    prompt_start = new_idx
                    prompt_end = new_idx + self.prompt_length
                    
                    # Find phrase end (next dot/sep or end)
                    phrase_end_old = start
                    for i in range(start, orig_len):
                        if ids_b[i] in (dot_id, sep_id):
                            phrase_end_old = i
                            break
                        phrase_end_old = i + 1
                    
                    prompt_ranges.append((prompt_start, prompt_end, start, phrase_end_old))
                    
                    new_idx += self.prompt_length
                    cursor = start
                    phrase_idx += 1
                
                # remaining tokens
                for old_pos in range(cursor, orig_len):
                    old_to_new[old_pos] = new_idx
                    new_idx += 1
                
                # ---- Step 2: CLS attends to CLS ----
                new_mask[0, 0] = orig_mask[0, 0]
                
                # ---- Step 3: Copy attention for original tokens ----
                for old_i in range(orig_len):
                    for old_j in range(orig_len):
                        if old_i in old_to_new and old_j in old_to_new:
                            new_i = old_to_new[old_i]
                            new_j = old_to_new[old_j]
                            new_mask[new_i, new_j] = orig_mask[old_i, old_j]
                
                # ---- Step 4: Prompt attention patterns ----
                for prompt_start, prompt_end, phrase_start_old, phrase_end_old in prompt_ranges:
                    # Map phrase tokens to new positions
                    phrase_tokens_new = [old_to_new[p] for p in range(phrase_start_old, phrase_end_old) 
                                        if p in old_to_new]
                    
                    for p_idx in range(prompt_start, prompt_end):
                        # 1. Prompts attend to CLS
                        new_mask[p_idx, 0] = True
                        
                        # 2. Prompts attend to themselves (within same class)
                        new_mask[p_idx, prompt_start:prompt_end] = True
                        
                        # 3. Prompts attend to their associated phrase tokens
                        for phrase_tok in phrase_tokens_new:
                            new_mask[p_idx, phrase_tok] = True
                    
                    # 4. CLS attends to all prompts
                    new_mask[0, prompt_start:prompt_end] = True
                    
                    # 5. Phrase tokens attend to their prompts
                    for phrase_tok in phrase_tokens_new:
                        new_mask[phrase_tok, prompt_start:prompt_end] = True
                
                new_attention_masks.append(new_mask)
                
            else:
                # attention_mask is [L] (1D)
                orig_mask = attention_mask[b]  # [L]
                orig_len = ids_b.size(0)
                new_len = embeds_new.size(0)
                new_mask = torch.zeros(new_len, device=device, dtype=orig_mask.dtype)
                
                new_idx = 0
                new_mask[new_idx] = orig_mask[0]  # CLS
                new_idx = 1
                
                cursor = 1
                for start in phrase_starts:
                    # tokens before phrase
                    for old_pos in range(cursor, start):
                        new_mask[new_idx] = orig_mask[old_pos]
                        new_idx += 1
                    
                    # prompt tokens (set to 1 - attend)
                    new_mask[new_idx:new_idx + self.prompt_length] = 1
                    new_idx += self.prompt_length
                    cursor = start
                
                # remaining tokens (including SEP and any padding)
                for old_pos in range(cursor, orig_len):
                    new_mask[new_idx] = orig_mask[old_pos]
                    new_idx += 1
                
                new_attention_masks.append(new_mask)

            # --------------------------------------------------
            # 4. Update position_ids
            # --------------------------------------------------
            if position_ids is not None:
                orig_pos = position_ids[b]  # [L]
                orig_len = ids_b.size(0)
                new_len = embeds_new.size(0)
                new_pos = torch.zeros(new_len, device=device, dtype=orig_pos.dtype)
                
                new_idx = 0
                new_pos[new_idx] = orig_pos[0]  # CLS position
                new_idx = 1
                
                cursor = 1
                for start in phrase_starts:
                    # tokens before phrase
                    for old_pos_idx in range(cursor, start):
                        new_pos[new_idx] = orig_pos[old_pos_idx]
                        new_idx += 1
                    
                    for p in range(self.prompt_length):
                        new_pos[new_idx] = 0
                        new_idx += 1
                    
                    cursor = start
                
                # remaining tokens (including SEP and padding)
                for old_pos_idx in range(cursor, orig_len):
                    new_pos[new_idx] = orig_pos[old_pos_idx]
                    new_idx += 1
                
                new_position_ids.append(new_pos)

            # --------------------------------------------------
            # 5. Update token_type_ids
            # --------------------------------------------------
            if token_type_ids is not None:
                orig_type = token_type_ids[b]  # [L]
                orig_len = ids_b.size(0)
                new_len = embeds_new.size(0)
                new_type = torch.zeros(new_len, device=device, dtype=orig_type.dtype)
                
                new_idx = 0
                new_type[new_idx] = orig_type[0]  # CLS token type
                new_idx = 1
                
                cursor = 1
                for start in phrase_starts:
                    # tokens before phrase
                    for old_pos_idx in range(cursor, start):
                        new_type[new_idx] = orig_type[old_pos_idx]
                        new_idx += 1
                    
                    # prompt token types (same as surrounding context)
                    context_type = orig_type[start] if start < orig_len else 0
                    new_type[new_idx:new_idx + self.prompt_length] = context_type
                    new_idx += self.prompt_length
                    
                    cursor = start
                
                # remaining tokens (including SEP and padding)
                for old_pos_idx in range(cursor, orig_len):
                    new_type[new_idx] = orig_type[old_pos_idx]
                    new_idx += 1
                
                new_token_type_ids.append(new_type)

                # --------------------------------------------------
                # 6. Build text_token_mask (prompts = False, real tokens = True)
                # --------------------------------------------------
                # Get original text_token_mask from tokenizer
                if self.use_sub_sentence_represent:
                    orig_text_mask = tokenized.attention_mask[b]  # [L]
                    
                    # Build new mask with prompts marked as False
                    new_text_mask = create_text_token_mask_with_prompts(
                        orig_text_mask, 
                        phrase_starts, 
                        self.prompt_length,
                        embeds_new.size(0),  # Current length before padding
                        device
                    )
                    new_text_token_masks.append(new_text_mask)
        # --------------------------------------------------
        # 7. Pad and stack all tensors
        # --------------------------------------------------
        max_len = max(x.size(0) for x in new_embeds)
        
        inputs_embeds = torch.stack([
            pad_embed(x, max_len) for x in new_embeds
        ])  # [B, L_max, C]

        if self.use_sub_sentence_represent:
            # Pad 2D attention masks
            attention_mask = torch.stack([
                pad_2d_mask(x, max_len) for x in new_attention_masks
            ])  # [B, L_max, L_max]
        else:
            # Pad 1D attention masks
            attention_mask = torch.stack([
                pad_mask(x, max_len) for x in new_attention_masks
            ])  # [B, L_max]

        if position_ids is not None:
            position_ids = torch.stack([
                pad_mask(x, max_len) for x in new_position_ids
            ]).to(torch.long)  # [B, L_max]
        
        if token_type_ids is not None:
            token_type_ids = torch.stack([
                pad_mask(x, max_len) for x in new_token_type_ids
            ]).to(torch.long)  # [B, L_max]

        if self.use_sub_sentence_represent:
            text_token_mask = torch.stack([
                pad_mask(x, max_len) for x in new_text_token_masks
            ]).bool()  # [B, L_max]

        tokenizer_input = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }

        language_dict_features = self.language_backbone(tokenizer_input)

        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
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
