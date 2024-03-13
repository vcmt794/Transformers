My Trasformers implementation. Using pytorch.

  Positional Encoding: Abs and RoPE.

  Attention: Multihead: Have RoPE and KV_cache (continue building, will have FlashAttn).

  Normalization: RMSNorm and LayerNorm.

  FFN: Base FFN use in "Attention is all u need", FFN with SiLU, MoE.

  DecoderLayer: the one in "Attention is all u need", the MoE and LLama
