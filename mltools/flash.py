"""Collection of the different functional forms of attention."""

import torch as T
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func


def flash_self_attention(
    x: T.Tensor,
    culens: T.Tensor,
    maxlen: int,
    drop: float,
    causal: bool,
    weight: T.Tensor,
    bias: T.Tensor,
    num_heads: int,
) -> T.Tensor:
    dim = x.size(-1)
    qkv = F.linear(x, weight, bias)
    qkv = qkv.view(-1, 3, num_heads, dim // num_heads)
    attn = flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, drop, causal=causal)
    return attn.contiguous().view(-1, dim)


def flash_cross_attention(
    x: T.Tensor,
    culens: T.Tensor,
    maxlen: int,
    kv: T.Tensor,
    kv_culens: T.Tensor,
    kv_maxlen: int,
    drop: float,
    causal: bool,
    weight: T.Tensor,
    bias: T.Tensor | None,
    num_heads: int,
) -> T.Tensor:
    dim = x.size(-1)
    head_dim = dim // num_heads
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)
    q_proj = F.linear(x, w_q, b_q).view(-1, num_heads, head_dim)
    kv_proj = F.linear(kv, w_kv, b_kv).view(-1, 2, num_heads, head_dim)
    attn = flash_attn_varlen_kvpacked_func(
        q_proj, kv_proj, culens, kv_culens, maxlen, kv_maxlen, drop, causal=causal
    )
    return attn.contiguous().view(-1, dim)
