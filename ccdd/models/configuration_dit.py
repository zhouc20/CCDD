from transformers import PretrainedConfig


class DITConfig(PretrainedConfig):
    model_type = "dit"

    def __init__(
        self,
        vocab_size: int = 50258,
        max_seq_len: int = 1024,
        hidden_size: int = 768,
        timestep_cond_dim: int = 128,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.0,
        p_uniform: float = 0.0,
        t_eps: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.timestep_cond_dim = timestep_cond_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.p_uniform = p_uniform
        self.t_eps = t_eps
