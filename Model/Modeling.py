from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import Tensor

from Norm import RMSNorm
from TransConfig import TransformersConfig
from Cache import Cache
from Decoder import DecoderOnlyLayer

from sentencepiece import SentencePieceProcessor
from torch import optim


# ------HYPERPARAMETER-------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 64  # how many independent sequences will we process in parallel?
max_seq_len = 256
max_iteration = 5000
# ------HYPERPARAMETER-------------#

with open('HarryPotter1.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokenizer = SentencePieceProcessor()
tokenizer.Load('tokenizer.model')
data = torch.tensor(tokenizer.Encode(text, add_bos=False, add_eos=False), dtype=torch.long)
n = int(0.9 * len(data))  # the first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - max_seq_len, (batch_size,))
    src = torch.stack([data_[i:i + max_seq_len] for i in ix])
    trg = torch.stack([data_[i + 1:i + max_seq_len + 1] for i in ix])
    src, trg = src.to(device), trg.to(device)
    return src, trg


x, y = get_batch('train')

#
# def _load_balancing_auxiliary_loss(all_router_logits,
#                                    num_experts: torch.Tensor = None,
#                                    top_k =2,
#                                    attention_mask: Optional[torch.Tensor] = None):
#     if all_router_logits is None:
#         return 0
#     if attention_mask is None:
#     else:


def generate_mask(mask: Tensor,
                  input_shape: Tuple[int, int],
                  _dtype: torch.dtype,
                  past_key_value_length: int = 0,
                  _device: torch.device = torch.device("mps")):
    bs, sq_len = input_shape

    if mask is not None and len(mask.shape) == 2:
        if sq_len > 1:  # When in this situation, we know that this is truly training time (KV_cache is one input
            # query baby)
            """
            Create a casual attention mask (the upper triangular one), than "add" this input mask.
            """
            casual_mask = torch.full(size=(sq_len, sq_len), fill_value=torch.finfo(_dtype).min, device=_device)
            mask_cond = torch.arange(casual_mask.size(-1), device=_device)
            casual_mask.masked_fill_(mask_cond < (mask_cond + 1).view(casual_mask.size(-1), 1), 0).to(device=_device)
            if past_key_value_length > 0:
                casual_mask = torch.cat(
                    [torch.zeros(sq_len, past_key_value_length + sq_len, dtype=_dtype, device=device), casual_mask],
                    dim=-1)
            casual_mask = casual_mask[None, None, :, :].expand(bs, 1, sq_len, sq_len + past_key_value_length)

            """Expand the input mask"""
            mask = mask[None, None, :, :].expand(bs, 1, sq_len, sq_len + past_key_value_length)
            """Apply the input mask to Casual mask"""
            mask = casual_mask.masked_fill(mask.to(torch.bool), torch.finfo(_dtype).min).to(device=device)
    elif mask is not None and len(mask.shape) == 4:
        inverted_mask = 1.0 - mask
        mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(_dtype).min)
        mask.to(_device)
    else:  # Mask is None -> Casual mask only
        mask = torch.full(size=(sq_len, sq_len), fill_value=torch.finfo(_dtype).min, device=_device)
        mask_cond = torch.arange(mask.size(-1), device=_device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0).to(device=_device)
        if past_key_value_length > 0:
            mask = torch.cat([torch.zeros(sq_len, past_key_value_length, dtype=_dtype, device=_device), mask], dim=-1)
        mask = mask[None, None, :, :].expand(bs, 1, sq_len, sq_len + past_key_value_length)
    return mask


"""
Don't use torch.tri() or .triu(), this is faster! (on GPU)
"""


class Mixtral(nn.Module):
    def __init__(self, config: TransformersConfig,
                 vocab_size: int):
        super(Mixtral, self).__init__()

        self.token_emb = nn.Embedding(vocab_size, config.hidden_dim)
        self.config = config
        self.vocab_size = vocab_size

        self.Layers = nn.ModuleList(
            [DecoderOnlyLayer(config, layer_idx, True) for layer_idx in range(config.layer_num)]
        )
        self.last_linear = nn.Linear(in_features=config.hidden_dim, out_features=vocab_size)
        self.last_norm = RMSNorm(dim=config.hidden_dim, eps=config.eps)

    @torch.inference_mode()
    def forward(self,
                input_ids: Tensor,
                mask: Optional[Tensor] = None,
                position_ids: Tensor = None,
                past_key_value: List[Tuple[Tensor, Tensor]] | Cache = None,
                targets: Optional[Tensor] = None):
        hidden = self.token_emb(input_ids).to(device)
        bs, sq, _ = hidden.shape
        if not isinstance(past_key_value, Cache):
            past_key_value = Cache.from_legacy(past_key_value)
        past_kv_length = past_key_value.get_pre_seq_length()

        mask = generate_mask(mask, (hidden.shape[0], hidden.shape[1]), hidden.dtype, past_kv_length, device)

        if position_ids is None:
            position_ids = torch.arange(past_kv_length, past_kv_length + sq, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, sq)
        else:
            position_ids = position_ids.view(-1, sq).long()

        """
        For debug
        """
        all_hidden_states = ()
        all_self_attns = ()
        all_router_logits = ()
        next_decoder_cache = None
        for layer in self.Layers:
            layer_output = layer(hidden,
                                 mask,
                                 position_ids,
                                 past_key_value)

            hidden = layer_output[0]
            all_self_attns += (layer_output[1],)
            next_decoder_cache = layer_output[2]
            all_hidden_states += (hidden,)
            all_router_logits += (layer_output[-1],)
        hidden = self.last_norm(hidden)
        hidden = self.last_linear(hidden)
        next_token_prob = torch.nn.Softmax(dim=-1)(hidden)
        loss = 0
        if targets is not None:
            logits = next_token_prob.contiguous()
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            targets = targets.view(-1)
            loss = loss_fct(logits, targets)
        return {
            "next_token_probability": next_token_prob,
            "last_hidden_state": hidden,
            "next-kv-cache": next_decoder_cache,
            "all_hidden_states": all_hidden_states,
            "all_self_attentions": all_self_attns,
            "all_router_logits": all_router_logits,
            "loss": loss
        }

    def generate(self, input_ids: Tensor, max_length_provide: int, past_kv=None):
        idx = input_ids
        past_kv_cache = past_kv
        for _ in range(max_length_provide):
            out_put_dict = self(idx, None, None, past_kv_cache)
            next_prob_tensor = out_put_dict["next_token_probability"]
            past_kv_cache = out_put_dict['next-kv-cache']
            idx = torch.cat([idx, torch.multinomial(next_prob_tensor[:, -1, :], 1)], dim=1)
        return idx


model = Mixtral(TransformersConfig(), 32000).to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.forward(x, targets=y)
print(out['loss'])
