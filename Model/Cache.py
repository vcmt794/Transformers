import torch
from torch import Tensor
from typing import Tuple, List, Optional


class Cache:
    """
    abstract class for Cache
    """

    def __init__(self) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError("Check your input at layer_idx")

    def update(self, layer_idx: int,
               new_k: Optional[Tensor] = None,
               new_v: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Check if there was any K_V tensor at this Cache
        if layer_idx == 0:
            self.seen_tokens += new_k.shape[-2]
            # Update seen_token, Update at the first layer (we input token at this layer)
            # In an AG model, the output will be brought back to the first layer -> Seen_token will be updated otherwise

        # Update Cache/Push the Key_Value Tensor into Cache
        if layer_idx >= len(self):  # haven't had any k_v cached in this layer yet -> len(k/v_cache +1)
            self.key_cache.append(new_k)
            self.value_cache.append(new_v)
        else:  # Already had -> concat to k/v of this layer.
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_k], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_v], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_pre_seq_length(self, layer_index: Optional[int] = None):
        if len(self) <= layer_index:
            return 0
        return self.key_cache[layer_index].shape[-2]

    @classmethod
    def from_legacy(cls,
                    past_kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None):
        cache = cls()
        if past_kv_cache is None:
            return cache
        for layer_idx in range(len(past_kv_cache)):
            new_k, new_v = past_kv_cache[layer_idx]
            cache.update(layer_idx, new_k, new_v)
        return cache
