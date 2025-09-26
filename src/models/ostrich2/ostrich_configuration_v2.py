from typing import Any
from torch import dtype
from torch._C import dtype
from transformers import PretrainedConfig

"""
📐 模型参数:
├── 词汇表大小: 151,936
├── 隐藏层维度: 4,096
├── 中间层维度: 22,016
├── 层数: 32
├── 注意力头数: 32
├── 头维度: 128
├── 最大序列长度: 32,768
├── RoPE theta: 10,000
└── RMS Norm epsilon: 1e-6

"""


class OstrichConfigurationV2(PretrainedConfig):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 151936
        self.dim = 4096
        self.hidden_size = 22016
        self.num_layers = 32
        self.num_heads = 128
        self.num_kv_heads = 32
        self.max_seq_len = 32768
        self.rope_theta = 10000.0
        self.rms_norm_eps = 1e-6
        self.attention_dropout=0.0
