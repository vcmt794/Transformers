import torch
import torch.nn as nn
import math


class FeedForward(nn.Module):  # This is FFN that is used in "Attention is all u need" paper
    def __int__(self, hidden_dim, ffn_dim, dropout, bias=True):
        """
        :param hidden_dim: the dimension of the hidden states
        :param ffn_dim: dimension of the ffn
        :param dropout: dropout probability
        :return: FFN(x) = ReLU(xW1+b1)W2+b2 (in T5, we don't use bias)
        """
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias)
        self.dropout = nn.Dropout(p=dropout)  # this is optional

    def forward(self, x):
        return self.w2(self.dropout(nn.functional.relu(self.w1(x))))


class SwishFFN(nn.Module):
    def __int__(self, hidden_dim, ffn_dim):
        """
        :param hidden_dim:
        :param ffn_dim:
        :return: <SiLU(xW1), xW3>W2, no bias is used.
        """
        super(SwishFFN, self).__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __int__(self, hidden_dim, ffn_dim, expert_num, top_K):
        super(MoELayer, self).__int__()
        self.expert_num = expert_num
        self.experts = nn.ModuleList([SwishFFN(hidden_dim, ffn_dim) for _ in range(self.expert_num)])
        self.gate = nn.Linear(hidden_dim, self.expert_num, bias=False)
        self.top_K = top_K

    def forward(self, x):
        bs, sq, dim = x.shape()
        x = x.view(-1, dim)  # (bs*sq, hidden), but i'll call it bs,hidden for short
        router_logits = self.gate(x)  # (bs,hidden_d)->(bs, exp_num)
        router_logits = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)  # turn into Probability
        router_weights, chosen_expert = torch.topk(router_logits, k=self.top_K, dim=-1)  # value, indices
        router_weights /= router_weights.sum(dim=-1, keepdims=True)  # (
        router_weights = router_weights.to(x.dtype)

        #  turn into one-hot vectorbs
        experts_mask = torch.nn.functional.one_hot(chosen_expert, num_classes=self.expert_num).permute(2, 1, 0)

        # "permute (2,1,0)?"
        # the "non-permuted" one is (bs*sq, k , num_ex), it shows that, at n_th token, it will use k sorted_Experts:
        # the 1st vector is the expert has the highest probability, number 1 tell which expert in num_ex.
        # And so on.
        # Eg: 2 experts, 3rd and 0th expert.
        #          [[0, 0, 0, 1, 0],
        #          [1, 0, 0, 0, 0]],

        # But we won't loop through all tokens and get the experts that tokens choose.
        # Instead, we wil reverse it: loop through all experts, and choose tokens that use the current expert.
        # -> permute(2,1,0) -> (num_ex, k, bs*sq)

        final_hidden_states = torch.zeros(
            (bs * sq, dim), dtype=x.dtype, device=x.device
        )

        for expert_id in range(self.expert_num):
            cur_exp = self.experts[expert_id]
            ranking, token_use = torch.where(experts_mask[expert_id])

            if token_use.shape[0] is 0:
                continue  # pass if there were no token use this expert
            # use list for faster indexing
            ranking_list = ranking.to(list)
            token_use_list = token_use.to(list)
            cur_states = x[None, token_use_list].reshape(-1, dim)
            cur_hidden_states = cur_exp(cur_states) * router_weights[token_use_list, ranking_list, None]
            # p_i*Expert_i(x)
            # (token_use, dim) x (token_use, 1) ( The None in order to change row vector -> col vector)
            # position_wise_mul
            final_hidden_states = final_hidden_states.index_add_(dim=0, index=token_use, source=cur_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(bs, sq, dim)
        return final_hidden_states, router_weights
