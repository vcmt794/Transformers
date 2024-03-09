from typing import Optional, Tuple, List

import torch
from torch import nn
import PositionalEncoding
import Attention
import FFN
from TransConfig import TransformersConfig
from Norm import RMSNorm, LayerNorm
from Attention import MultiheadAttn, AdvancedAttn

ATTENTION_TYPE = {
    "base": Attention.MultiheadAttn,
    "advanced": Attention.AdvancedAttn,
    "flash": Attention.AdvancedAttn
}


class DecoderLayer(nn.Module):  # Base Decoder in Attention is All you need
    def __init__(self, config: TransformersConfig = TransformersConfig()):
        super(DecoderLayer, self).__init__()
        self.hidden_dim = config.hidden_dim

        self.ffn = FFN.FeedForward(config)

        self.mask_attn = MultiheadAttn(config)
        self.cross_attn = ATTENTION_TYPE["base"](config)

        self.mask_attn_norm = LayerNorm(config.hidden_dim, config.eps)
        self.cross_attn_norm = LayerNorm(config.hidden_dim, config.eps)
        self.out_norm = LayerNorm(config.hidden_dim, config.eps)

    def forward(self, dec_in, enc_in, scr_mask, trg_mask):
        x = dec_in
        x = self.mask_attn_norm(x + self.mask_attn(q=dec_in, k=dec_in, v=dec_in, mask=trg_mask))

        if enc_in is not None:
            x = self.cross_attn_norm(x + self.cross_attn(q=x, k=enc_in, v=enc_in, mask=scr_mask))
        x = self.out_norm(self.ffn(x) + x)
        return x


class DecoderOnlyLayer(nn.Module):
    def __init__(self, config: TransformersConfig):
        super(DecoderOnlyLayer, self).__init__()
        self.hidden_dim = config.hidden_dim

        self.attn = ATTENTION_TYPE["advanced"](config)  # ur choice, flashattn or not.

        self.MoE = FFN.MoELayer(config)
        self.in_norm = RMSNorm(config.hidden_dim, config.eps)
        self.out_norm = RMSNorm(config.hidden_dim, config.eps)

    def forward(self,
                hidden_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[List[torch.Tensor], List[torch.Tensor], int]] = None):
        residual = hidden_states
        hidden_states = self.in_norm(hidden_states)

        hidden_states, self_attn_weight, cur_past_kv = self.attn(hidden_states,
                                                                 mask,
                                                                 position_ids,
                                                                 past_kv)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.out_norm(hidden_states)
        hidden_states, router_logits = self.MoE(hidden_states)
        hidden_states = hidden_states+residual

        return hidden_states, self_attn_weight, cur_past_kv, router_logits



