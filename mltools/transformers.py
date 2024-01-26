"""Some classes to describe transformer architectures."""

import math
import warnings
from functools import partial
from typing import Mapping

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


def merge_masks(
    kv_mask: T.BoolTensor | None,
    attn_mask: T.BoolTensor | None,
    attn_bias: T.Tensor | None,
    query: T.Size,
) -> None | T.BoolTensor:
    """Create a full attention mask which using the padding information and bias."""

    # Create the placeholder for the full mask, None is full attention
    merged_mask = None

    # If the kv_mask mask exists, we ensure that padded tokens never send information
    if kv_mask is not None:
        merged_mask = kv_mask.unsqueeze(-2).expand(-1, query.shape[-2], -1)

    # If attention mask exists, combine it with the existing
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    # If the attention bias exists, convert to a float and add to the mask
    if attn_bias is not None:
        if merged_mask is not None:
            merged_mask = T.where(merged_mask, 0, -T.inf).type(query.dtype)
            merged_mask = merged_mask + attn_bias.permute(0, 3, 1, 2)
        else:
            merged_mask = attn_bias.permute(0, 3, 1, 2)

    return merged_mask


def my_scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: T.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    attn_act: callable = partial(F.softmax, dim=-1),
    pad_val: float = -float("inf"),
) -> T.Tensor:
    """Compute dot product attention using the given query, key, and value tensors.

    Pure PyTorch implementation of the scaled dot product attention operation.
    Note that ONNX supports Pytorch's native function since opset 14.

    Parameters
    ----------
    query : T.Tensor
        The query tensor.
    key : T.Tensor
        The key tensor.
    value : T.Tensor
        The value tensor.
    attn_mask : T.Tensor | None, optional
        The attention mask tensor, by default None.
    dropout_p : float, optional
        The dropout probability, by default 0.0.
    is_causal : bool, optional
        Whether to use causal attention, by default False.
    scale: float | None, optional
        The scale factor to divide the attention weights by, by default None.
    attn_act : callable, optional
        The attention activation function, by default partial(softmax, dim=-1).
    pad_val : float, optional
        The padding value for the attention mask, by default -float("inf").

    Returns
    -------
    T.Tensor
        The result of the scaled dot product attention operation.
    """

    # Get the shapes and set the scale
    L = query.shape[-2]
    S = key.shape[-2]
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Build the attention bias as a float
    attn_bias = T.zeros(L, S, dtype=query.dtype, device=query.device)

    # If using a causal mask, then set the upper triangle to the pad value
    if is_causal:
        assert attn_mask is None, "Causal attention does not support attention masks!"
        attn_mask = T.ones(L, S, dtype=T.bool).tril(diagonal=0)
        attn_bias.masked_fill_(~attn_mask, pad_val)

    # If proved own attention mask, then add it to the bias
    elif attn_mask is not None:
        if attn_mask.dtype == T.bool:
            attn_bias.masked_fill_(~attn_mask, pad_val)
        else:
            attn_bias += attn_mask

    # Apply the attention operation using the mask as a bias
    attn_weight = query @ key.transpose(-2, -1) * scale
    attn_weight = attn_act(attn_weight + attn_bias)
    attn_weight = T.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def attach_context(x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
    """Concat a tensor with context which has the same or lower dimensions."""
    if ctxt is None:
        return x
    dim_diff = x.dim() - ctxt.dim()
    if dim_diff > 0:
        ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
        ctxt = ctxt.expand(*x.shape[:-1], -1)
    return T.cat((x, ctxt), dim=-1)


def add_context(x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
    """Add a tensor with context which has the same or lower dimensions."""
    if ctxt is None:
        return x
    dim_diff = x.dim() - ctxt.dim()
    if dim_diff > 0:
        ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
        ctxt = ctxt.expand(*x.shape[:-1], -1)
    return x + ctxt


def rotate_half(x: T.Tensor) -> T.Tensor:
    """Split a tensor in two to and swaps the order."""
    x1, x2 = x.chunk(2, dim=-1)
    return T.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: T.Tensor, cos: T.Tensor, sin: T.Tensor) -> T.Tensor:
    """Apply rotary positional embedding for relative encoding."""
    return (x * cos) + (rotate_half(x) * sin)


class LayerScale(nn.Module):
    """Applies the LayerScale operation from the Cait vision transformer."""

    def __init__(
        self,
        dim: int,
        init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * T.ones(dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x * self.gamma


class PreNormScaledResidual(nn.Module):
    """Wraps a module with pre-norm and layerscale with a residual connection."""

    def __init__(
        self, fn: nn.Module, dim: int, layerscale_init: float | None = 1e-5
    ) -> None:
        """
        Parameters
        ----------
        fn : nn.Module
            The module to wrap.
        dim : int
            The dimension of the input and output.
        layerscale_init : float | None, optional
            The initial value for the layerscale, by default 1e-5.
            If None, then no layerscale is applied.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.ls = (
            LayerScale(dim, layerscale_init)
            if layerscale_init is not None
            else nn.Identity()
        )

    def forward(self, x: T.Tensor, *args, **kwargs) -> T.Tensor:
        return self.ls(self.fn(self.norm(x), *args, **kwargs)) + x


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
    """Basic multiheaded attention block.

    Now supports DiffiT style context embedding: https://arxiv.org/abs/2312.02139
    """

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        num_heads: int = 1,
        dropout: float = 0,
        do_self_attn: bool = False,
        do_rotary_enc: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        dim : int
            The dimension of the input and output.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
        num_heads : int, optional
            The number of attention heads, by default 1.
        dropout : float, optional
            The dropout probability, by default 0.
        do_self_attn : bool, optional
            Whether to optimise for self attention by using a single weight matrix,
            by default False.
        do_rotary_enc : bool, optional
            Whether to use rotary positional encoding, by default False.
        use_context : bool
        """
        super().__init__()
        assert dim % num_heads == 0

        # Attributes
        self.dim = dim
        self.num_heads = num_heads
        self.ctxt_dim = ctxt_dim
        self.attn_dim = dim // num_heads
        self.do_self_attn = do_self_attn
        self.dropout = dropout
        self.do_rotary_enc = do_rotary_enc

        # Weight matrices - only 1 input for self attn
        if self.do_self_attn:
            self.attn_in = nn.Linear(dim, 3 * dim)
        else:
            self.attn_in = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)])
        self.attn_out = nn.Linear(dim, dim)

        # Positional encoding
        if self.do_rotary_enc:
            self.rotary = RotaryEmbedding(dim)

        # Context embedding, scale the parameters by 1e-3
        if self.ctxt_dim:
            self.ctxt_mixer = nn.Linear(ctxt_dim + dim, dim)

    def forward(
        self,
        x: T.Tensor,
        kv: T.Tensor | None = None,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the attention block."""

        # Mix the input together with the context - 2312.02139
        # This overparameterises the Q, K, V projections but seems to help
        if self.ctxt_dim:
            x = self.ctxt_mixer(attach_context(x, ctxt))

        # Generate the q, k, v projections -> B,S,D
        if self.do_self_attn:
            q, k, v = self.attn_in(x).chunk(3, -1)
        else:
            if kv is None:
                kv = x
                warnings.warn("Suboptimal use of self attention detected!")
                warnings.warn("Initialise block with do_self_attn=True!")
            q = self.attn_in[0](x)
            k, v = self.attn_in[1](kv).chunk(2, -1)

        # Break final dim and transpose -> B,NH,S,Hd
        shape = (q.shape[0], -1, self.num_heads, self.attn_dim)
        q, k, v = map(lambda t: t.view(shape).transpose(1, 2), (q, k, v))

        # Apply rotary positional encoding on the q and k tensors
        if self.do_rotary_enc:
            q, k = self.rotary(q, k)

        # Perform the attention -> B,NH,S,Hd
        a_mask = merge_masks(kv_mask, attn_mask, attn_bias, q)
        dropout = self.dropout if self.training else 0.0
        a_out = F.scaled_dot_product_attention(q, k, v, a_mask, dropout)

        # Concatenate the all of the heads -> B,S,D
        a_out = a_out.transpose(1, 2).contiguous().view(q.shape[0], -1, self.dim)

        return self.attn_out(a_out)


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(
        self, dim: int, hddn_dim: int, ctxt_dim: int = 0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.use_ctxt = ctxt_dim > 0
        self.lin1 = nn.Linear(dim + ctxt_dim, 2 * hddn_dim)
        self.lin2 = nn.Linear(hddn_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if self.use_ctxt:
            x = attach_context(x, ctxt)
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class EncoderBlock(nn.Module):
    """Building block for the transformer encoder containing MHSA and FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_mult: int = 2,
        num_heads: int = 8,
        dropout: float = 0,
        do_self_attn: bool = False,
        do_rotary_enc: bool = False,
        layerscale_init: float | None = 1e-4,
        do_ctxt_in_attn: bool = False,
        do_ctxt_in_ff: bool = True,
    ) -> None:
        super().__init__()

        # Attributes
        self.dim = dim
        self.num_heads = num_heads

        # Some dimensions
        attn_ctxt = do_ctxt_in_attn * ctxt_dim
        ff_ctxt = do_ctxt_in_ff * ctxt_dim
        ff_hddn = ff_mult * dim

        # Submodules
        self.attn = PreNormScaledResidual(
            Attention(dim, attn_ctxt, num_heads, dropout, do_self_attn, do_rotary_enc),
            dim,
            layerscale_init,
        )
        self.ff = PreNormScaledResidual(
            SwiGLUNet(dim, ff_hddn, ff_ctxt, dropout),
            dim,
            layerscale_init,
        )

    def forward(
        self,
        x: T.Tensor,
        kv: T.Tensor | None = None,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        x = self.attn(x, kv, ctxt, kv_mask, attn_mask, attn_bias)
        x = self.ff(x, ctxt)
        return x


class DecoderBlock(nn.Module):
    """Decoder block which includes a cross attention operation."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_mult: int = 2,
        num_heads: int = 8,
        dropout: float = 0,
        do_rotary_enc: bool = False,
        layerscale_init: float | None = 1e-5,
        do_ctxt_in_attn: bool = False,
        do_ctxt_in_ff: bool = True,
    ) -> None:
        super().__init__()

        # Attributes
        self.dim = dim
        self.num_heads = num_heads

        # Some dimensions
        attn_ctxt = do_ctxt_in_attn * ctxt_dim
        ff_ctxt = do_ctxt_in_ff * ctxt_dim
        ff_hddn = ff_mult * dim

        # Submodules
        self.self_attn = PreNormScaledResidual(
            Attention(dim, attn_ctxt, num_heads, dropout, True, do_rotary_enc),
            dim,
            layerscale_init,
        )
        self.cross_attn = PreNormScaledResidual(
            Attention(dim, attn_ctxt, num_heads, dropout, False, False),
            dim,
            layerscale_init,
        )
        self.ff = PreNormScaledResidual(
            SwiGLUNet(dim, ff_hddn, ff_ctxt, dropout),
            dim,
            layerscale_init,
        )

    def forward(
        self,
        x: T.Tensor,
        *,
        kv: T.Tensor,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
        x_mask: T.BoolTensor | None = None,
    ) -> T.Tensor:
        """Pass through the decoder block.

        Same as the encoder but with an extra x_mask for the self attention and kv is
        required
        """
        x = self.self_attn(x, None, ctxt, x_mask, attn_mask, attn_bias)
        x = self.cross_attn(x, kv, ctxt, kv_mask)
        x = self.ff(x, ctxt)
        return x


class Transformer(nn.Module):
    """Simple transformer stack of encoder or decoder blocks."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 6,
        max_seq_len: int = 0,
        num_registers: int = 0,
        do_input_linear: bool = False,
        do_output_linear: bool = False,
        do_absolute_enc: bool = False,
        do_final_norm: bool = False,
        layer_config: Mapping | None = None,
        inpt_dim: None | int = None,
        outp_dim: None | int = None,
        use_decoder: bool = False,
    ) -> None:
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_registers = num_registers
        self.do_final_norm = do_final_norm
        self.do_input_linear = do_input_linear
        self.do_output_linear = do_output_linear
        self.do_absolute_enc = do_absolute_enc
        self.layer_config = layer_config
        self.inpt_dim = inpt_dim if do_input_linear else dim
        self.outp_dim = outp_dim if do_output_linear else dim
        self.use_decoder = use_decoder

        # Initial linear projection (Always done before absolute encoding)
        if self.do_input_linear:
            self.linear_embed = nn.Linear(inpt_dim, dim)

        # Absolute positional encoding
        if self.do_absolute_enc:
            if max_seq_len == 0:
                raise ValueError("If using absolute encoding then define max length!")
            self.abs_enc = nn.Parameter(T.randn((1, max_seq_len, dim)) * 1e-3)

        # Layers
        if self.use_decoder:
            self.layers = nn.ModuleList(
                [DecoderBlock(dim, ctxt_dim, **layer_config) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [EncoderBlock(dim, ctxt_dim, **layer_config) for _ in range(num_layers)]
            )

        # Final normalisation layer
        if self.do_final_norm:
            self.final_norm = nn.LayerNorm(dim)

        # Final linear projection
        if self.do_output_linear:
            self.linear_out = nn.Linear(dim, outp_dim)

        # The registers from "TRANSFORMERS NEED REGISTERS" - 2309.16588
        if self.num_registers:
            self.registers = nn.Parameter(T.randn((1, self.num_registers, dim)))

    def project(self, x: T.Tensor) -> T.Tensor:
        """Project the input to the transformer dimension."""
        if self.do_input_linear:
            x = self.linear_embed(x)
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]
        return x

    def _add_registers(self, x: T.Tensor, **kwargs) -> tuple:
        """Add the registers to the front of the input and the appropriate mask.

        TODO How to return the modified kwargs while staying backwards compatible? Needs
        to be in the forward method.
        """

        # Expand the registers so they can be broadcasted for the whole batch
        registers = self.registers.expand(x.shape[0], self.num_registers, self.dim)

        # Add the registers to the FRONT of the input
        x = T.cat([registers, x], dim=-2)

        # If using decoder blocks then mask for x comes from the x_mask (duh!)
        if self.use_decoder and "x_mask" in kwargs:
            p = T.ones((x.shape[0], self.num_registers), dtype=T.bool, device=x.device)
            kwargs["x_mask"] = T.cat([p, kwargs["x_mask"]], dim=-1)

        # If using self attention then we must add to the kv_mask
        if not self.use_decoder and "kv_mask" in kwargs and "kv" not in kwargs:
            p = T.ones((x.shape[0], self.num_registers), dtype=T.bool, device=x.device)
            kwargs["kv_mask"] = T.cat([p, kwargs["kv_mask"]], dim=-1)

        return x, kwargs

    def remove_registers(self, x: T.Tensor) -> T.Tensor:
        """Remove the registers from the front of the input."""
        return x[:, self.num_registers :]

    def encode(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.num_registers:
            x, kwargs = self._add_registers(x, **kwargs)
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def output(self, x: T.Tensor) -> T.Tensor:
        if self.do_final_norm:
            x = self.final_norm(x)
        if self.do_output_linear:
            x = self.linear_out(x)
        return x

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Project and encode, seperated for flexibility and FlowBert."""
        return self.output(self.encode(self.project(x), **kwargs))


class CrossAttentionEncoder(Transformer):
    """Permutation equivariant encoder with linear N computational expense."""

    def __init__(self, num_tokens: int = 16, **kwargs) -> None:
        super().__init__(use_decoder=False, **kwargs)
        self.num_tokens = num_tokens

        # The learnable global tokens and extra modules
        self.global_tokens = nn.Parameter(T.randn((1, num_tokens, self.dim)))
        self.pool_layers = nn.ModuleList(
            [
                EncoderBlock(self.dim, self.ctxt_dim, **self.layer_config)
                for _ in range(self.num_layers)
            ]
        )

    def encode(
        self, x: T.Tensor, kv_mask: T.BoolTensor | None = None, **kwargs
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""

        # Expand the global token so it can be broadcasted for the whole batch
        g = self.global_tokens.expand(x.shape[0], -1, self.dim)

        # Pool and distribute
        for pool_layers, dist_layers in zip(self.pool_layers, self.layers):
            g = pool_layers(g, x, kv_mask=kv_mask, **kwargs)
            x = dist_layers(x, g, **kwargs)

        return x


class TransformerEncoder(Transformer):
    """Transformer encoder stack, here for backwards compatibility."""

    def __init__(self, **kwargs) -> None:
        print("TransformerEncoder is deprecated, use Transformer instead.")
        super().__init__(use_decoder=False, **kwargs)


class ClassAttentionPooling(nn.Module):
    """Pooling operation that uses attention."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 2,
        layer_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.layer_config = layer_config

        # Modules
        self.global_token = nn.Parameter(T.randn((1, 1, self.dim)))
        self.layers = nn.ModuleList(
            [
                EncoderBlock(dim, ctxt_dim, **self.layer_config)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        # Expand the global token so it can be broadcasted for the whole batch
        g = self.global_token.expand(x.shape[0], -1, self.dim)

        # Apply the iterative pooling
        for layer in self.layers:
            g = layer(g, x, **kwargs)

        # Pop out the sequence dimension and return
        return g.squeeze(-2)


class TransformerVectorEncoder(nn.Module):
    """Combination of Encoder+ClassAttention to produce a vector given a set."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        encoder_config: Mapping | None = None,
        classattention_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        encoder_config = encoder_config or {}
        classattention_config = classattention_config or {}

        # Attributes
        self.dim = dim
        self.ctxt_dim = ctxt_dim

        # Modules
        self.encoder = Transformer(dim=dim, ctxt_dim=ctxt_dim, **encoder_config)
        self.pool = ClassAttentionPooling(
            dim=self.encoder.outp_dim, ctxt_dim=ctxt_dim, **classattention_config
        )

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        return self.pool(self.encoder(x, **kwargs), **kwargs)


class WrappedTransformer(nn.Module):
    """Wrap a transformer with input and output embedding networks."""

    def __init__(
        self,
        *,
        inpt_dim: int,
        outp_dim: int,
        transformer: partial,
        ctxt_dim: int = 0,
        edge_dim: int = 0,
        node_embd_config: Mapping | None = None,
        outp_embd_config: Mapping | None = None,
        ctxt_embd_config: Mapping | None = None,
        edge_embd_config: Mapping | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        node_embd_config = node_embd_config or None
        outp_embd_config = outp_embd_config or None
        ctxt_embd_config = ctxt_embd_config or None
        edge_embd_config = edge_embd_config or None

        # Attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim

        # Context embedding network (optional)
        self.ctxt_out = 0
        if self.ctxt_dim:
            self.ctxt_emdb = MLP(inpt_dim=self.ctxt_dim, **ctxt_embd_config)
            self.ctxt_out = self.ctxt_emdb.outp_dim

        # Main transformer
        self.transformer = transformer(ctxt_dim=self.ctxt_out)
        self.dim = self.transformer.dim
        self.num_heads = self.transformer.layers[0].num_heads

        # The input and output embedding network
        self.node_embd = MLP(
            inpt_dim=self.inpt_dim,
            outp_dim=self.dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = MLP(
            inpt_dim=self.dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

        # Edge embedding network (optional)
        if self.edge_dim:
            self.edge_emdb = MLP(
                inpt_dim=self.edge_dim,
                ctxt_dim=self.ctxt_out,
                outp_dim=self.num_heads**edge_embd_config,
            )

    def forward(
        self,
        x: T.Tensor,
        ctxt: T.Tensor | None = None,
        edge: T.Tensor | None = None,
        mask: T.BoolTensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        if self.edge_dim:
            edge = self.edge_dim(edge, ctxt)
        x = self.node_embd(x, ctxt)
        x = self.transformer(x, kv_mask=mask, ctxt=ctxt, attn_bias=edge, **kwargs)
        x = self.outp_embd(x, ctxt)
        return x
