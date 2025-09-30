from transformers import PretrainedConfig
from typing import Optional, Union
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class OstrichModelConfig(PretrainedConfig):
    model_type = "Ostrich-Llm"

    def __init__(
        self,
        dim: int = 2048,  # 模型维度
        n_layers: int = 28,  # Transformer的层数
        n_heads: int = 16,  # 注意力机制的头数
        n_kv_heads: int = 8,  # 键值头的数量
        vocab_size: int = 8192,  # 词汇表大小
        hidden_dim: int = 6144,  # 隐藏层维度
        multiple_of: int = 64,
        norm_eps: float = 1e-5,  # 归一化层的eps
        max_seq_len: int = 32768,  # 最大序列长度
        dropout: float = 0.0,  # dropout概率
        flash_attn: bool = True,  # 是否使用Flash Attention
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
