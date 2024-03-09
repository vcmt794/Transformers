class TransformersConfig:
    def __init__(self,
                 hidden_dim=512,
                 head_num=8,
                 dropout=0.1,
                 padding_idx=1,
                 max_len=2048,
                 vocab_size=32000,
                 ffn_dim=1792,
                 attn="base",
                 ffn_dropout=0,
                 ffn_bias=False,
                 expert_num=8,
                 top_k=2,
                 eps=1e-5,
                 rope_theta=1e6,
                 kv_headnum=2):
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.attn = attn
        self.ffn_dim = ffn_dim
        self.ffn_bias = ffn_bias
        self.ffn_dropout = ffn_dropout
        self.expert_num = expert_num
        self.top_k = top_k
        self.eps = eps
        self.rope_theta = rope_theta
        self.kv_head_num = kv_headnum


