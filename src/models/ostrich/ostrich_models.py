from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from src.models.ostrich_configuration import OstrichModelConfig
from transformers import PreTrainedModel
from transformers.models.qwen3 import Qwen3ForTokenClassification
from typing import Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


## repeat_kv
### 因为使用 gropu_attn， q使用 n_heads, kv使用 kv_heads， 所以在计算最终的attn时，需要扩增k，v的维度到与v相同的维度


def repeat_kv(x: torch.Tensor, n_rep: int):
    """

    Args:
        x (torch.Tensor): k_states or v_states
        n_rep (int): repeat的数量
    """
    bsz, seq_len, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    expand_x = x[
        :, :, :, None, :
    ]  # expand_x shape (bsz, seqlen, kv_heads, 1, head_dim)
    expand_x = expand_x.expand(bsz, seq_len, kv_heads, n_rep, head_dim).reshape(
        bsz, seq_len, kv_heads * n_rep, head_dim
    )
    return expand_x


# 预计算 分组频率 也就是 e^i* theta = cos(theta) + i * sin(theta) i表示虚数单位




def precompute_freq_cis(max_seq_length, dim, theta: float = 10000.0):
    # 维度频率计算
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[: dim // 2] / dim).float()

    # 生成 序列index,
    t = torch.arange(0, max_seq_length).type_as(freqs)

    # 外积计算
    freqs = torch.outer(t, freqs)  # seq_len, dim // 2

    # 计算 大小为1 方向为theta的旋转矩阵 即转为复数域
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope_embedding(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # bsz, seq_len, dim => bsz , seq_len, dim//2 2
    # 按照维度进行22分组，x y
    xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 2)

    # 转为复数 构建为x+iy
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2).type_as(xk)
    return xq_out, xk_out


class Attention(nn.Module):
    def __init__(self, args: OstrichModelConfig) -> None:
        super().__init__()

        # 判别是否包含 n_kv_heads, 若不包含 则n_kv_heads = n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # 确保 n_heads 是否是 n_kv_heads的整数倍
        assert args.n_heads % self.n_kv_heads == 0

        model_parallel_size = 1

        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size

        self.n_reps = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        freq_cis = precompute_freq_cis(args.max_seq_len, dim=self.head_dim)
        self.register_buffer("freq_cis", freq_cis)
        self.dropout_prob = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.flash:
            # 若不支持，则手动mask
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len), fill_value=float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        bsz, seq_len = x.shape[:2]

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = repeat_kv(xv, self.n_reps)
        xk = repeat_kv(xk, self.n_reps)
        # transpose
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq, xk = apply_rope_embedding(
            xq,
            xk,
            self.freq_cis[  # pyright: ignore[reportArgumentType, reportIndexIssue]
                :seq_len, :
            ],  # pyright: ignore[reportArgumentType, reportIndexIssue]
        )  # pyright: ignore[reportArgumentType, reportIndexIssue]
        # 计算 scores

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout_prob, is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

            assert hasattr(self, "mask")
            scores = (
                scores
                + self.mask[  # pyright: ignore[reportIndexIssue]
                    :, :, :seq_len, :seq_len
                ]  # pyright: ignore[reportIndexIssue]
            )  # pyright: ignore[reportIndexIssue]

            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            output = torch.matmul(attn, xv)

        # 恢复维度
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(
        self, dim: int, hidden_size: int, multiple_of: int, dropout_prob: float
    ) -> None:
        super().__init__()
        if hidden_size is None:
            # 如果hidden_size 不设置的话，我们往往会先将其设置为 dim 的4 倍，然后将至 2/3 倍，也就是 8 * dim // 3
            # 另外 hidden_size 应该为multiple_of 的整数倍
            hidden_size = int(4 * dim)
            hidden_size = int(2 * hidden_size / 3)
            hidden_size = ((hidden_size + multiple_of - 1) // multiple_of) * multiple_of

        self.w1 = nn.Linear(dim, hidden_size, bias=False)
        self.w2 = nn.Linear(dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.selu(self.w1(x)) + self.w2(x)))


## 构建Decoder


class DecoderLayer(nn.Module):
    def __init__(self, args: OstrichModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)

        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.feed_forward = MLP(
            args.dim, args.hidden_dim, args.multiple_of, args.dropout
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, args: OstrichModelConfig) -> None:
        super().__init__()
        self.layers = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


## 至此我们开始组装我们的模型



class OstrichModel(PreTrainedModel):
    config_class = OstrichModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, config: OstrichModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.decoder = Decoder(config)

        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for param_name, param in self.named_parameters():
            if param_name.endswith("wo.weight") or param_name.endswith("w3.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )
        self.last_loss = None
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs
    ):
        if "input_ids" in kwargs:
            tokens = kwargs["input_ids"]
        if "attention_mask" in kwargs:
            targets = kwargs["attention_mask"]

        token_embeds = self.embed(tokens)
        token_embeds = self.embed_dropout(token_embeds)

        decoder_out = self.decoder(token_embeds)
        if targets is not None:
            logits = self.output(decoder_out)
            self.last_loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
        else:
            logitis = self.output(decoder_out[:, [-1], :])
            self.last_loss = None

        return CausalLMOutputWithPast(loss=self.last_loss, logits=logitis) # type: ignore

    @torch.inference_mode()
    def generator(
        self,
        idx: torch.Tensor,
        stop_id=None,
        max_new_tokens=256,
        temperature=1.0,
        top_k=None,
    ):
        """
        给定输入序列 idx (bsz, seq_len), 需要不断的预测下一个token, 直到输出max_new_tokens

        """
        index = idx.size(-1)
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(-1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len :]
            )
            logits = self(idx_cond).logits
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits: torch.Tensor = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
                    logits[logits < v[-1]] = float("-inf")
                    # 计算softmax
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            idx = torch.cat([idx, idx_next], dim=1)
        return idx[index:]


if __name__ == "__main__":
    x = torch.randint(0, 6144, (1, 50))
    
    config = OstrichModelConfig()
    model = OstrichModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    out = model(x)
    print(out.logits.shape) # [batch_size, 1, vocab_size]
    print('Number of parameters:', num_params)