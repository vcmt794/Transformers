import math
import torch
from torch import nn


class SequenceTooLongError(Exception):
    pass


class PositionalEncoding(nn.Module):
    def __int__(self,
                dim=512,
                dropout=0.1,
                max_len=5000):
        super(PositionalEncoding, self).__init__()
        if dim % 2 != 0:
            msg = "Can't use odd dim"
            raise ValueError(msg)
        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        :param emb: (b*sq*d)
        :param step:

        :return:
        """

        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if self.pe.size(0) < step + emb.size(1):  # compare max_len and sq_len
            error_msg = (f"Sequence is {emb.size(0) + step} but PositionalEncoding is limited to {self.pe.size(0)}. "
                         f"See max_len argument.")
            raise SequenceTooLongError(error_msg)
        emb = emb + self.pe[:, step:emb.size(0) + step]
        emb = self.dropout(emb)

        return emb


"""
Above is ABSOLUTE PE
we don't need it anymore.
And i don't have enough time to create the RELATIVE one
So we'll skip to the most modern one:
behold, the ROTARY POSITIONAL ENCODINGGG (RoPE)
(actually, there is xPos, but it's more complicated than my capability
so... XD
"""


class RotaryPositionalEncoding(nn.Module):
    def __int__(self,
                dim,
                base=10000.0,
                max_len=2048,
                device=None):
        super(RotaryPositionalEncoding, self).__int__()
        self.d = dim
        self.max_len = max_len
        self.base = base
        # theta = 1./(self.base * 10000 ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        theta = (torch.exp(torch.arange(0, self.d, 2).float() * -(math.log(self.base) / self.d))).to(device)
        self.register_buffer('theta', theta, persistent=False)

    def _build_cached(self, cur_seq_len, device):
        self.max_seq_len_cached = cur_seq_len
        pos = torch.arange(end=self.max_seq_len_cached, device=device).type_as(self.theta)
        postheta = torch.outer(pos, self.theta)
        emb = torch.cat([postheta, postheta], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(float), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(float), persistent=False)

    def forward(self, x, cur_seq_len):
        if cur_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=cur_seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:cur_seq_len].to(dtype=x.dtype),
            self.sin_cached[:cur_seq_len].to(dtype=x.dtype),
        )

    """
    We don't use x.shape(2) (seq_len) for cur_seq_len cuz...
    So obv, the current sequent length can be bigger than the length which is cached. (decoder)
    """


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, sin, cos, position_idx):
    """
    :param q: (b,h,sq,d)
    :param k: _________
    :param sin: (sq,d)
    :param cos: _____
    :param position_idx: (1,sq)

    :return: rotate k,v
    """

    sin = sin[position_idx].unsqueeze(1)  # (1,1,sq,d) -> Broadcastable with q,k
    cos = cos[position_idx].unsqueeze(1)
    q_ro = (q * cos) + (rotate_half(q) * sin)
    k_ro = (k * cos) + (rotate_half(k) * sin)
    return q_ro, k_ro
