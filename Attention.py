import torch
import torch.nn as nn
import math


class MultiheadAttn(nn.Module):
    def __init__(self,
                 model_dim=512,
                 head_num=8,
                 attn_type='self',
                 dropout=0.1,
                 padding_idx=1):
        super(MultiheadAttn, self).__init__()
        self.dim_per_head = model_dim // head_num
        self.model_dim = model_dim
        self.head_num = head_num
        self.type = attn_type

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
        return x.view(batch_size, -1, self.head_num, self.dim_per_head).transpose(1, 2)

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

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_per_head)   # (sq*d_q) @ (d_k * sq)
        """
        After this matmul, the matrix (in each head) which we've just received 
        represents the "link" between Tokens. 
        To more specific: the x_th column show how much "attention" x_th token 
        pay to all the other token (and also itself).
        For Generalization, we want all these attention(s) are non-negative and sum to 1.
        -> Softmax(dim=-1)
        """
        if mask is not None:
            mask = mask.unsqueeze(1) #(B,T) -> ()
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = nn.Softmax(-1)(scores)
        drop_attn = self.dropout(attn)
        self_attn = torch.matmul(drop_attn, v)  # (sq*sq) @ (sq*d_v) -> (sq @ d_v)
        """
        Finally, the self-attention: n_th Token sa[x_n] is a weighted sum of all the values v_n, 
        these weights are those attention scalars we've talked above.
        """
        self_attn = self.unshape(self_attn)
        return self_attn
    """
        "Why not simply use X (with X is a token's emb) as the input of process_qkv and forward(...)?" U asked
        "Because we will have Cross-attention a.k.a Attention between Encoder and Decoder
        In which, q is from deco and k,v are from enc." Me ans.
    """

    def update_dropout(self, dropout):
        self.dropout.p = dropout
