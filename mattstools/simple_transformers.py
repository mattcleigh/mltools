"""Some classes to describe transformer architectures."""

import math
from functools import partial
from typing import Mapping

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .modules import DenseNetwork
from .transformers import merge_masks


def param_init(module: nn.Module, depth: int, method: str = "default") -> None:
    """Initialise the pre residual layers of a module using a specific method."""

    if not hasattr(module, "pre_residual_layers"):
        return

    # Get the list of parameter tensors to modify
    p_list = [m.weight for m in module.pre_residual_layers]
    for m in module.pre_residual_layers:
        if hasattr(m, "bias"):
            p_list += [m.bias]

    # Apply the appropriate weight initialisation
    if method == "default":
        return

    if method == "zero":
        for p in p_list:
            p.data.fill_(0)

    if method == "beit":
        for p in p_list:
            p.data /= math.sqrt(4 * (depth + 1))


def attach_context(x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
    """Concat a tensor with context which has the same or lower dimensions."""
    if ctxt is None:
        return x
    dim_diff = x.dim() - ctxt.dim()
    if dim_diff > 0:
        ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
        ctxt = ctxt.expand(*x.shape[:-1], -1)
    return T.cat((x, ctxt), dim=-1)


def rotate_half(x: T.Tensor) -> T.Tensor:
    """Split a tensor in two to and swaps the order."""
    x1, x2 = x.chunk(2, dim=-1)
    return T.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: T.Tensor, cos: T.Tensor, sin: T.Tensor) -> T.Tensor:
    """Apply rotary positional embedding for relative encoding."""
    return (x * cos) + (rotate_half(x) * sin)


class PreNormResidual(nn.Module):
    """Apply layernorm with residual connection."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: T.Tensor, *args, **kwargs) -> T.Tensor:
        return self.fn(self.norm(x), *args, **kwargs) + x


class RotaryEmbedding(nn.Module):
    """Applies rotary positional embedding for relative encoding."""

    def __init__(self, dim: int, max_pos: int = 10_000):
        super().__init__()
        self.dim = dim

        # Register the scales as a buffer
        scales = 1.0 / (max_pos ** (T.arange(0, dim, 2).float() / dim))
        self.register_buffer("scales", scales)

        # Cache certain information to speed up training and inference
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: T.Tensor) -> None:
        seq_len = x.shape[-2]

        # Reset the tables only if the sequence length / device / dtype has changed
        if (
            seq_len == self._seq_len_cached
            and self._cos_cached.device == x.device
            and self._cos_cached.dtype == x.dtype
        ):
            return

        # Generate the the frequencies used for the sin cosine embedding
        seq = T.arange(seq_len, device=x.device, dtype=T.float32)
        freqs = T.outer(seq, self.scales)
        emb = T.cat((freqs, freqs), dim=-1)

        # Set the cached attributes to the new variables
        self._seq_len_cached = seq_len
        self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
        self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

    def forward(self, q: T.Tensor, k: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class Attention(nn.Module):
    """Basic multiheaded attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        dropout: float = 0,
        do_self_attn: bool = False,
        do_rotary_enc: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        # Attributes
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.do_self_attn = do_self_attn
        self.dropout = dropout
        self.do_rotary_enc = do_rotary_enc

        # Weight matrices - only 1 input for self attn
        if do_self_attn:
            self.attn_in = nn.Linear(dim, 3 * dim)
        else:
            self.attn_in = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)])
        self.attn_out = nn.Linear(dim, dim)

        # Positional encoding
        if self.do_rotary_enc:
            self.rotary = RotaryEmbedding(dim)

    def forward(
        self,
        x: T.Tensor,
        kv: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        # If only q is provided then we automatically apply self attention
        if kv is None:
            kv = x

        # Generate the q, k, v projections
        if self.do_self_attn:
            q, k, v = self.attn_in(x).chunk(3, -1)
        else:
            q = self.attn_in[0](x)
            k, v = self.attn_in[1](kv).chunk(2, -1)

        # Break final dim, transpose to get dimensions: B,NH,Seq,Hdim
        shape = (q.shape[0], -1, self.num_heads, self.attn_dim)
        q, k, v = map(lambda t: t.view(shape).transpose(1, 2), (q, k, v))

        # Apply rotary positional encoding on the q and k tensors
        if self.do_rotary_enc:
            q, k = self.rotary(q, k)

        # Perform the attention
        a_mask = merge_masks(kv_mask, attn_mask, attn_bias, q)
        dropout = self.dropout if self.training else 0.0
        a_out = F.scaled_dot_product_attention(q, k, v, a_mask, dropout)

        # Concatenate the all of the heads together to get shape: B,Seq,F
        shape = (q.shape[0], -1, self.dim)
        a_out = a_out.transpose(1, 2).contiguous().view(shape)

        return self.attn_out(a_out)


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network."""

    def __init__(
        self, dim: int, hddn_dim: int, ctxt_dim: int = 0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim + ctxt_dim, 2 * hddn_dim)
        self.lin2 = nn.Linear(hddn_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        x = attach_context(x, ctxt)
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class TransformerLayer(nn.Module):
    """Simple and flexible layer for a transformer."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_mult: int = 2,
        num_heads: int = 8,
        dropout: float = 0,
        do_self_attn: bool = False,
        do_rotary_enc: bool = False,
    ) -> None:
        super().__init__()

        # Attributes
        self.dim = dim

        # Submodules
        self.attn = PreNormResidual(
            dim, Attention(dim, num_heads, dropout, do_self_attn, do_rotary_enc)
        )
        self.ff = PreNormResidual(dim, SwiGLUNet(dim, ff_mult * dim, ctxt_dim, dropout))

        # Add flags / pointers to the pre-residual layers to allow for initialisation
        self.pre_residual_layers = [self.attn.fn.attn_out, self.ff.fn.lin2]

    def forward(
        self,
        x: T.Tensor,
        kv: T.Tensor | None = None,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        x = self.attn(x, kv, kv_mask, attn_mask, attn_bias)
        x = self.ff(x, ctxt)
        return x


class CrossAttentionLayer(TransformerLayer):
    """A transformer cross attention layer, has additional norm for kv."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_norm = nn.LayerNorm(self.dim)

    def forward(self, x: T.Tensor, kv: T.Tensor, **kwargs) -> T.Tensor:
        kv = self.kv_norm(kv)
        return super().forward(x, kv, **kwargs)


class TransformerEncoder(nn.Module):
    """Simple and constrained transformer encoder."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 6,
        max_seq_len: int = 0,
        do_absolute_enc: bool = False,
        init_method: str = "default",
        layer_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.do_absolute_enc = do_absolute_enc

        # Absolute positional encoding
        if self.do_absolute_enc:
            if max_seq_len == 0:
                raise ValueError("If using absolute encoding then define max length!")
            self.abs_enc = nn.Parameter(T.zeros((1, max_seq_len, dim)))

        # Modules
        self.te_layers = nn.ModuleList(
            [
                TransformerLayer(dim, ctxt_dim, do_self_attn=True, **layer_config)
                for _ in range(num_layers)
            ]
        )

        # Change the weight initialisation in the te blocks based on depth
        for d, layer in enumerate(self.te_layers):
            param_init(layer, d, init_method)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        # Add the positional encoding
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]

        # Pass through the layers
        for layer in self.te_layers:
            x = layer(x, **kwargs)

        # Output projections
        return x


class TransformerVectorEncoder(nn.Module):
    """Pooling operation that uses attention."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_sa_layers: int = 3,
        num_ca_layers: int = 2,
        max_seq_len: int = 0,
        do_absolute_enc: bool = False,
        init_method: str = "default",
        layer_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.do_absolute_enc = do_absolute_enc

        # Absolute positional encoding
        if self.do_absolute_enc:
            if max_seq_len == 0:
                raise ValueError("If using absolute encoding then define max length!")
            self.abs_enc = nn.Parameter(T.zeros((1, max_seq_len, dim)))

        # The learnable global token
        self.global_token = nn.Parameter(T.randn((1, 1, dim)))

        # Modules
        self.sa_layers = nn.ModuleList(
            [
                TransformerLayer(dim, ctxt_dim, **layer_config)
                for _ in range(num_sa_layers)
            ]
        )
        self.ca_layers = nn.ModuleList(
            [
                CrossAttentionLayer(dim, ctxt_dim, **layer_config)
                for _ in range(num_ca_layers)
            ]
        )

        # Change the weight initialisation in the te blocks based on depth
        for d, layer in enumerate([self.sa_layers + self.ca_layers]):
            param_init(layer, d, init_method)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        # Add the positional encoding
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]

        # Self attention
        for layer in self.sa_layers:
            x = layer(x, **kwargs)

        # Pass through the layers with batch expanded global token
        g = self.global_token.expand(x.shape[0], -1, self.dim)
        for layer in self.ca_layers:
            g = layer(g, x, **kwargs)

        # Pop out the sequence dimension
        return g.squeeze(-2)


class CrossAttentionEncoder(nn.Module):
    """Permutation equivariant encoder with linear N computational expense."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 6,
        num_tokens: int = 16,
        max_seq_len: int = 0,
        do_absolute_enc: bool = False,
        init_method: str = "default",
        layer_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.do_absolute_enc = do_absolute_enc

        # Absolute positional encoding
        if self.do_absolute_enc:
            if max_seq_len == 0:
                raise ValueError("If using absolute encoding then define max length!")
            self.abs_enc = nn.Parameter(T.zeros((1, max_seq_len, dim)))

        # The learnable global tokens
        self.global_tokens = nn.Parameter(T.randn((1, num_tokens, dim)))

        # Modules
        self.pool_layers = nn.ModuleList(
            [
                CrossAttentionLayer(dim, ctxt_dim, **layer_config)
                for _ in range(num_layers)
            ]
        )
        self.dist_layers = nn.ModuleList(
            [
                CrossAttentionLayer(dim, ctxt_dim, **layer_config)
                for _ in range(num_layers)
            ]
        )

        # Change the weight initialisation in the te blocks based on depth
        for d, layer in enumerate(self.pool_layers):
            param_init(layer, d, init_method)
        for d, layer in enumerate(self.dist_layers):
            param_init(layer, d, init_method)

    def forward(self, x: T.Tensor, kv_mask: T.BoolTensor, **kwargs) -> T.Tensor:
        # Add the positional encoding
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]

        # Pass through the layers with batch expanded global tokens
        g = self.global_tokens.expand(x.shape[0], -1, self.dim)
        for pool_layers, dist_layers in zip(self.pool_layers, self.dist_layers):
            g = pool_layers(g, x, kv_mask=kv_mask, **kwargs)
            x = dist_layers(x, g, **kwargs)
        return x


class FullEncoder(nn.Module):
    """A transformer with input and output embedding networks."""

    def __init__(
        self,
        *,
        inpt_dim: int,
        outp_dim: int,
        transformer: partial,
        ctxt_dim: int = 0,
        node_embd_config: Mapping | None = None,
        outp_embd_config: Mapping | None = None,
        ctxt_embd_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        node_embd_config = node_embd_config or None
        outp_embd_config = outp_embd_config or None
        ctxt_embd_config = ctxt_embd_config or None

        # Attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # Context embedding network (optional)
        self.ctxt_out = 0
        if self.ctxt_dim:
            self.ctxt_emdb = DenseNetwork(inpt_dim=self.ctxt_dim, **ctxt_embd_config)
            self.ctxt_out = self.ctxt_emdb.outp_dim

        # Main transformer
        self.transformer = transformer(ctxt_dim=self.ctxt_out)
        self.dim = self.transformer.dim

        # The input and output embedding network
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

    def forward(
        self,
        x: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        x = self.node_embd(x, ctxt)
        x = self.transformer(x, kv_mask=mask, ctxt=ctxt)
        x = self.outp_embd(x, ctxt)
        return x
