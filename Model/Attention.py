from typing import Optional, List

import torch
import torch.nn as nn
import math

from torch import Tensor

from PositionalEncoding import RotaryPositionalEncoding, apply_rotary


class MultiheadAttn(nn.Module):
    def __init__(self,
                 model_dim=512,
                 head_num=8,
                 dropout=0.1,
                 padding_idx=1):
        super(MultiheadAttn, self).__init__()
        self.dim_per_head = model_dim // head_num
        self.model_dim = model_dim
        self.head_num = head_num

        self.dropout = nn.Dropout(p=dropout)
        self.linear_key = nn.Linear(in_features=model_dim, out_features=self.dim_per_head * head_num)
        self.linear_query = nn.Linear(model_dim, self.dim_per_head * head_num)
        self.linear_value = nn.Linear(model_dim, self.dim_per_head * head_num)
        self.final_linear = nn.Linear(model_dim, self.dim_per_head * head_num, bias=False)
        self.padding_idx = padding_idx

    def shape(self, x):
        """
        :param x: (tensor) all data ( Batch_size * seq_len * model_dim )

        :return: split into N heads: (B*N*sl*model_d//N)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.head_num, self.dim_per_head).transpose(1, 2).contiguous()
        return x

    """
        So why [transpose(1,2)] ?
        Okay, Ehmmm(I dunno either)
        In the first part of the code, you convert the tensor into (BsqN*d//N).

        When performing the 'Multi-head attention', 
        we aim to split the Query/Key into N vectors, 
        then perform MatMul and concatenate them. 
        -->So, we can consider this part as MatMul on N sequences,
         where the size of each token's vector is d//N, a.k.a (B*N*sq*mini_d).

        (or if we use this shape to calculate the Attention, there is no change from a 'One-head Attention'. 
        The shape (N*d//N) essentially still comprises 'd' elements. 
        After Transpose(1,2), the shape becomes (sq, mini_d),
        which contains a different number of elements. :D)
        -------
    """

    def unshape(self, x):
        """
        :param x: (tensor) data after splitting

        :return: combine//concat all head together
        """
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.dim_per_head)

    def process_qkv(self, q, k, v):
        q, k, v = self.linear_query(q), self.linear_key(k), self.linear_value(v)
        k = self.shape(k)
        v = self.shape(v)
        q = self.shape(q)
        return q, k, v

    def forward(self, q, k, v, mask=None):  # Override
        """
        :param q: query tensor
        :param k: key tensor
        :param v: value tensor
        :param mask: Use for decoder, marking the positions that haven't had any data yet

        :return:self-Attention tensor of token X.
        """
        q, k, v = self.process_qkv(q, k, v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_per_head)
        # (bs*h*sq*d_q) mul (bs*h*d_k * sq) ~ (sq*d_q) mul (d_k*sq) d_k = d_q
        """
        After this matmul, the matrix (in each head) which we've just received 
        represents the "link" between Tokens. 
        To more specific: the x_th column show how much "attention" x_th token 
        pay to all the other token (and also itself).
        For Generalization, we want all these attention(s) are non-negative and sum to 1.
        -> Softmax(dim=-1)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3)  # (B,sq) -> (B,1,sq,1)
            scores = scores.masked_fill(mask == 0, -1e9)  # score (b*h*sq*sq)

        attn = nn.Softmax(-1)(scores)
        drop_attn = self.dropout(attn)
        self_attn = torch.matmul(drop_attn, v)  # (sq*sq) mul (sq*d_v) -> (sq @ d_v)
        """
        Finally, the self-attention: n_th Token sa[x_n] is a weighted sum of all the values v_n, 
        these weights are those attention scalars we've talked above.
        """
        self_attn = self.unshape(self_attn)
        return self_attn,

    """
        "Why not simply use X (with X is a token's emb) as the input of process_qkv and forward(...)?" U asked
        "Because we will have Cross-attention a.k.a Attention between Encoder and Decoder
        In which, q is from deco and k,v are from enc." Me ans.
    """

    def update_dropout(self, dropout):
        self.dropout.p = dropout


"""
Apply RoPE, KV_Caching, Sliding Window (In FlashAttn ver), Grouped Query Attention (Optional)
"""


def repeat_kv(x, nrep):
    """

    :param x: hidden states, (bs, kv_hn, sq, d)
    :param nrep: number of group (q_group_num), if nrep = 1 -> Multi-Head
    :return: duplicated k,v (bs, h, sq, d)
    """
    bs, kv_hn, sq, d = x.shape()
    if nrep == 1:
        return x
    x = x[:, :, None, :, :].expand(bs, kv_hn, nrep, sq, d)
    return x.reshape(bs, kv_hn * nrep, sq, d)


def get_pre_seq_length(layer_index: int | None, cache: Optional[List[List[Tensor], List[Tensor], int]] = None):
    if len(cache[0]) <= layer_index:  # cache[0] is key_cache
        return 0
    return cache[0][layer_index].shape[-2]


def update_cache(new_k: Optional[Tensor], new_v: Optional[Tensor], layer_idx: int = 0,
                 cache: List[List[Tensor], List[Tensor], int] = None):
    if layer_idx == 0:
        cache[2] += new_k.shape[-2]
        # Update seen_token, Update at the first layer (we input token at this layer)
        # In AG model, the output will be brought back to the first layer -> Seen_token will be updated otherwise

    if layer_idx >= len(cache[0]):  # haven't had any k_v cached in this layer yet -> len(k/v_cache +1)
        cache[0].append(new_k)
        cache[1].append(new_v)
    else:  # Already had -> concat to k/v of this layer.
        cache[0][layer_idx] = torch.cat([cache[0][layer_idx], new_k], dim=-2)
        cache[1][layer_idx] = torch.cat([cache[1][layer_idx], new_v], dim=-2)

    return cache[0], cache[1]



class AdvancedAttn(nn.Module):
    def __init__(self,
                dim=512,
                head_num=8,
                padding_idx=0,
                max_len=2048,
                rope_theta=10000,
                kv_head_num=4,
                layer_idx=6,
                dropout=0.1):
        super(AdvancedAttn, self).__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.head_num = head_num
        self.dim_per_head = dim // head_num
        self.rope_theta = rope_theta
        self.max_len = max_len
        self.dropout = dropout
        self.rotary = RotaryPositionalEncoding(dim=self.dim_per_head,
                                               base=self.rope_theta,
                                               max_len=self.max_len)

        # Check if you use Grouped Queries Attention
        self.kv_head_num = kv_head_num
        self.q_group_num = self.head_num // self.kv_head_num  # if equal 1 -> MultiHA

        self.linear_q = nn.Linear(in_features=dim, out_features=self.dim_per_head * self.head_num, bias=False)
        self.linear_k = nn.Linear(dim, self.dim_per_head * self.kv_head_num, False)  # (smaller or equal q)
        self.linear_v = nn.Linear(dim, self.dim_per_head * self.kv_head_num, False)
        self.last_linear = nn.Linear(self.dim_per_head * self.head_num, self.dim, False)

    def shape_(self, x):
        bs = x.shape(0)
        return x.view(bs, -1, self.head_num, self.dim_per_head).transpose(1, 2).contiguous()

    def shape_gqa(self, x):
        bs = x.shape(0)
        return x.view(bs, -1, self.kv_head_num, self.dim_per_head).transpose(1, 2).contiguous()

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor[int]] = None,
                position_ids: Optional[Tensor[int]] = None,
                past_kv: Optional[List[List[Tensor], List[Tensor], int]] = None):  # K, V, seen_token
        bs, sq_len = x.size()

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = self.shape_(q)
        k = self.shape_gqa(k)
        v = self.shape_gqa(v)

        kv_sq_len = k.shape(-2)

        if past_kv is not None:
            kv_sq_len += get_pre_seq_length(layer_index=self.layer_idx, cache=past_kv)
        cos, sin = self.rotary(k, kv_sq_len)
        q, k = apply_rotary(q, k, sin, cos, position_ids)

        if past_kv is not None:
            k, v = update_cache(k, v, self.layer_idx, cache=past_kv)

        # GQA
        k = repeat_kv(k, self.q_group_num)
        v = repeat_kv(v, self.q_group_num)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_per_head)
        if mask is not None:
            attn = attn + mask

        attn = nn.functional.softmax(attn, -1, dtype=torch.float32).to(q.dtype)
        attn = nn.functional.dropout(attn, p=self.dropout)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(bs, sq_len, self.dim)

        attn_out = self.last_linear(attn_out)

        return attn_out, attn, past_kv



