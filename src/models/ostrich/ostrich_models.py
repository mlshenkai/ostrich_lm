from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from src.models.ostrich.ostrich_configuration import OstrichModelConfig
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




def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

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
        self.dropout_prob = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # 若不支持flash attention，则手动创建mask
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len), fill_value=float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        bsz, seq_len = x.shape[:2]

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(
            xq,
            xk,
            freqs_cos,freqs_sin
        )  # pyright: ignore[reportArgumentType, reportIndexIssue]
        # 计算 scores
        xv = repeat_kv(xv, self.n_reps)
        xk = repeat_kv(xk, self.n_reps)
        # transpose
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

       

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

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.feed_forward(self.ffn_norm(x))
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, args: OstrichModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos, freqs_sin) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)
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


        freqs_cos, freqs_sin = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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
        bsz, seq_len = tokens.shape[:2]

        token_embeds = self.embed(tokens)
        token_embeds = self.embed_dropout(token_embeds)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seq_len] # type: ignore
        freqs_sin = self.freqs_sin[:seq_len] # type: ignore

        decoder_out = self.decoder(token_embeds, freqs_cos, freqs_sin)
        if targets is not None:
            logits = self.output(decoder_out)
            self.last_loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
        else:
            logits = self.output(decoder_out[:, [-1], :])
            self.last_loss = None

        return CausalLMOutputWithPast(loss=self.last_loss, logits=logits) # type: ignore

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