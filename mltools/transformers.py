"""Some classes to describe transformer architectures."""

import logging
import math
from collections.abc import Callable
from functools import partial

import torch as T
import torch.nn.functional as F
from torch import nn

from .flash import flash_cross_attention, flash_self_attention
from .mlp import MLP
from .torch_utils import append_dims

log = logging.getLogger(__name__)


def pos_embed(embed_dim: int, max_seq_len: int, num_registers: int = 0):
    """Create the positional embedding for the transformer."""
    assert embed_dim % 2 == 0

    # Create the increasing frequencies for the sin and cos functions
    omega = T.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    # Get the positions from the max sequence length
    pos = T.arange(max_seq_len, dtype=float).reshape(-1)

    # Create the matrix using the outer product of the positions and frequencies
    out = omega.unsqueeze(0) * pos.unsqueeze(-1)  # (S, D/2)

    # Embed using sin and cos functions then combine
    emb_sin = T.sin(out)  # (S, D/2)
    emb_cos = T.cos(out)  # (S, D/2)
    pos_emb = T.cat([emb_sin, emb_cos], axis=1)  # (S, D)

    # Add positional encoding for the registers
    if num_registers:
        reg_emb = T.randn((num_registers, embed_dim)) / 1000
        pos_emb = T.cat([reg_emb, pos_emb], axis=0)

    return pos_emb.unsqueeze(0).float()  # For batch dimension


def merge_masks(
    kv_mask: T.BoolTensor | None,
    attn_mask: T.BoolTensor | None,
    attn_bias: T.Tensor | None,
    query: T.Tensor,
    causal: bool = False,
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

    # If using a causal mask, then set the upper triangle to the pad value
    if causal and merged_mask is not None:
        merged_mask = merged_mask.tril()

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


def precompute_freqs_cis(x=T.Tensor, theta: float = 10000.0):
    _B, S, D = x.shape
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = 1.0 / (theta ** (T.arange(0, D, 2).float() / D))
    freqs = T.outer(t, freqs)
    return T.polar(T.ones_like(freqs), freqs)


def rope(x: T.Tensor, freqs_cis: T.Tensor) -> T.Tensor:
    B, S, D = x.shape
    q = T.view_as_complex(x.float().reshape(B, S, D // 2, 2))
    q = T.view_as_real(q * freqs_cis)
    return q.view_as(x).type_as(x)


def pack(
    x: T.Tensor, mask: T.Tensor | None = None, ctxt: T.Tensor | None = None
) -> T.Tensor:
    """Undo all padding and compress the sequence."""
    if mask is None:
        log.warning("Packing without a mask is not recommended!")
        mask = T.ones(x.shape[:-1], dtype=T.bool, device=x.device)

    # Get the culens and maxlen variables needed by the flash attention func
    seqlens = mask.sum(dim=-1)
    culens = F.pad(T.cumsum(seqlens, dim=-1), (1, 0), value=0).to(T.int32)
    maxlen = seqlens.max().item()

    # Context info gets tricky because it may need to be repeated
    if ctxt is not None:
        if (dim_diff := x.dim() - ctxt.dim()) > 0:  # Expand then pack (no mem copy)
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*x.shape[:-1], -1)
        ctxt = ctxt[mask]

    return x[mask], ctxt, culens, maxlen


def unpack(x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Take a compressed sequence and unpack it to a padded tensor."""
    out = T.zeros((*mask.shape, x.shape[-1]), dtype=x.dtype, device=x.device)
    out[mask] = x
    return out


def single_projection(
    q: T.Tensor,
    kv: T.Tensor | None,
    weight: T.Tensor,
    bias: T.Tensor | None,
) -> tuple:
    """Efficient input projection for MHA when using a single linear layer.

    Essentially the same as torch.nn.functional._in_projection_packed
    But here we use chunk which is 40x faster than unflatten
    Not sure why they don't use chunk in the original implementation
    """
    # self-attention, very quick
    if kv is None:
        return F.linear(q, weight, bias).chunk(3, dim=-1)

    # cross-attention, slightly slower, must seperate weights, split is a view
    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    # do seperate projections
    q_proj = F.linear(q, w_q, b_q)
    k_proj, v_proj = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj


def add_registers(
    x: T.Tensor,
    reg: T.Tensor,
    mask: T.BoolTensor,
    attn_mask: T.BoolTensor | None = None,
    attn_bias: T.Tensor | None = None,
    add_to_send: bool = False,
) -> tuple:
    """Add registers to the front of the input and accomidate the mask."""
    # expand the registers so they can be broadcasted for the whole batch
    reg = reg.expand(x.size(0), -1, x.shape[-1])
    nreg = reg.shape[1]

    # add the registers to the FRONT of the input
    x = T.cat([reg, x], dim=-2)  # Sequence dimension

    # Add the mask for the registers with trues at the front
    if mask is not None:
        mask = F.pad(mask, (nreg, 0), value=True)

    # Add the attention mask for the registers
    # The attention mask is b x recv x send
    # We are adding to the recv dimension
    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, (nreg * add_to_send, 0, nreg, 0), value=True)

    # Add an attention bias of zero for the registers
    if attn_bias is not None:
        attn_bias = F.pad(attn_bias, (0, 0, nreg * add_to_send, 0, nreg, 0), value=0)

    return x, mask, attn_mask, attn_bias


def my_scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: T.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    attn_act: Callable | None = None,
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
        The attention activation function, by default softmax.
    pad_val : float, optional
        The padding value for the attention mask, by default -float("inf").

    Returns
    -------
    T.Tensor
        The result of the scaled dot product attention operation.
    """
    # Default to the softmax activation function
    if attn_act is None:
        attn_act = partial(F.softmax, dim=-1)

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


class Identity(nn.Module):
    """Simple identity module that can take any number of arguments."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    def forward(self, x: T.Tensor, *_args, **_kwargs) -> T.Tensor:
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(T.ones(dim))
        self.const = dim ** (-0.5)

    def forward(self, x: T.Tensor) -> T.Tensor:
        norm = T.linalg.norm(x.float(), dim=-1, keepdim=True)
        return x * self.scale / (norm * self.const + 1e-8).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Applies rotary positional embedding for relative encoding."""

    def __init__(self, dim: int, theta: int = 10_000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("freqs_cis", T.empty(0))

    def _update_freqs_cis(self, x: T.Tensor) -> None:
        """Reset the tables buffer if the sequence / device has changed."""
        new_len = x.shape[1] != self.freqs_cis.shape[0]
        new_device = x.device != self.freqs_cis.device
        if new_len or new_device:
            self.freqs_cis = precompute_freqs_cis(x, self.theta)

    def forward(self, q: T.Tensor, k: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        self._update_freqs_cis(q)
        q = rope(q, self.freqs_cis)
        self._update_freqs_cis(k)
        k = rope(k, self.freqs_cis)
        return q, k


class Residual(nn.Module):
    """Wraps a module with a normalisation layer, residual connection and gating.

    The scale, shift and gate can be determined by a context tensor.
    Otherwise it is simply the LayerNorm weight+bias and LayerScale
    """

    def __init__(
        self,
        fn: nn.Module,
        dim: int = 0,
        ctxt_dim: int = 0,
        ls_init: float | None = 1.0,
    ) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        dim : int
            The dimension of the input and output.
            If zero we will try get it from the fn module.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
            Used in the modulator to determine the scale, shift and gate.
        ls_init : float | None, optional
            The initial value for the gate, by default 1.0.
        """
        super().__init__()
        self.dim = dim or fn.dim
        self.fn = fn
        self.ctxt_dim = ctxt_dim
        self.ls_init = ls_init
        if ctxt_dim:
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.ctxt_layer = nn.Sequential(nn.SiLU(), nn.Linear(ctxt_dim, dim * 3))
        else:
            self.norm = nn.LayerNorm(dim)  # Scale and shift are in here (faster)
            self.gate = nn.Parameter(T.empty(dim))  # LayerScale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        if not self.ctxt_dim:
            nn.init.constant_(self.gate, self.ls_init)

    def forward(
        self,
        x: T.Tensor,
        *args,
        ctxt: T.Tensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        if self.ctxt_dim:
            ctxt_out = self.ctxt_layer(ctxt)
            ctxt_out = append_dims(ctxt_out, x.dim(), dim=1)
            scale, shift, gate = ctxt_out.chunk(3, dim=-1)
            tmp = self.norm(x) * (scale + 1) + shift
        else:
            gate = self.gate
            tmp = self.norm(x)  # Scale and shift applied internally (faster)
        return x + self.fn(tmp, *args, **kwargs) * gate


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(self, dim: int, hddn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim  # Usefull for wrapping the module
        self.lin1 = nn.Linear(dim, 2 * hddn_dim)
        self.lin2 = nn.Linear(hddn_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class Attention(nn.Module):
    """Basic multiheaded attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        dropout: float = 0,
        do_rotary: bool = False,
        qk_norm: bool = False,
    ) -> None:
        """Initialise the attention block.

        Parameters
        ----------
        dim : int
            The dimension of the input and output.
        num_heads : int, optional
            The number of attention heads, by default 1.
        dropout : float, optional
            The dropout probability, by default 0.
        do_rotary : bool, optional
            Whether to use rotary positional encoding, by default False.
        qk_norm : bool, optional
            Whether to use RMSNorm on the query and key, by default False.
        """
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by the number of heads!"

        # Attributes
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.dropout = dropout
        self.do_rotary = do_rotary
        self.qk_norm = qk_norm

        # Better parallelism for self-attention when using parameters directly
        self.attn_in_w = nn.Parameter(T.empty(3 * dim, dim))  # weights
        self.attn_in_b = nn.Parameter(T.empty(3 * dim))  # biases
        self.attn_out = nn.Linear(dim, dim)

        # Optional extra layers
        if self.do_rotary:
            self.rotary = RotaryEmbedding(dim)
        if self.qk_norm:
            self.q_norm = RMSNorm(dim)
            self.k_norm = RMSNorm(dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self.attn_in_w)
        nn.init.constant_(self.attn_in_b, 0.0)
        self.attn_out.reset_parameters()

    def forward(
        self,
        x: T.Tensor,
        mask: T.BoolTensor | None = None,
        kv: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
        culens: T.Tensor | None = None,
        maxlen: int | None = None,
        causal: bool = False,
        kv_culens: T.Tensor | None = None,
        kv_maxlen: int | None = None,
    ) -> T.Tensor:
        """Pass through the attention block."""
        drop = self.dropout if self.training else 0.0

        # If providing the input with culens and maxlen, we assume packed attention
        if culens is not None and maxlen is not None:
            assert attn_mask is None, "Packed attn does not support attention masks!"
            assert attn_bias is None, "Packed attn does not support attention bias!"
            assert not self.do_rotary, "Packed attn does not support rotary emb!"
            assert not self.qk_norm, "Packed attn does not support qk norm!"

            if kv is None:
                a_out = flash_self_attention(
                    x,
                    culens,
                    maxlen,
                    drop,
                    causal,
                    self.attn_in_w,
                    self.attn_in_b,
                    self.num_heads,
                )

            else:
                a_out = flash_cross_attention(
                    x,
                    culens,
                    maxlen,
                    kv,
                    kv_culens,
                    kv_maxlen,
                    drop,
                    causal,
                    self.attn_in_w,
                    self.attn_in_b,
                    self.num_heads,
                )

        # Standard attention with masks and biases
        else:
            B, _S, _D = x.shape
            q, k, v = single_projection(x, kv, self.attn_in_w, self.attn_in_b)

            # Apply RMSNorm to the query and key tensors
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            # transform tensors -> B,NH,S,HD
            shape = (B, -1, self.num_heads, self.attn_dim)
            q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

            # Apply rotary positional encoding on the q and k tensors
            if self.do_rotary:
                q, k = self.rotary(q, k)

            # run attention -> B,NH,S,HD
            kv_mask = mask if kv is None else kv_mask  # who is sending?
            a_mask = merge_masks(kv_mask, attn_mask, attn_bias, q, causal)
            c = causal and a_mask is None  # a_mask will at least incl causal
            a_out = F.scaled_dot_product_attention(q, k, v, a_mask, drop, is_causal=c)

            # recombine heads -> B,S,D
            a_out = a_out.transpose(1, 2).contiguous().view(B, -1, self.dim)

        return self.attn_out(a_out)


class EncoderBlock(nn.Module):
    """Building block for the Transformer Encoder containing MHSA and FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_mult: int = 2,
        ff_dropout: float = 0,
        attn_dropout: float = 0,
        num_heads: int = 8,
        do_rotary: bool = False,
        ls_init: float | None = 1,
    ) -> None:
        """Initialise the encoder block.

        Parameters
        ----------
        dim : int
            The dimension of of the block
        ctxt_dim : int, optional
            The dimension of the context, by default 0
            Used in both the attention and feedforward submodules
        ff_mult : int, optional
            The multiplier for the feedforward network, by default 2
        num_heads : int, optional
            The number of attention heads, by default 8
        ff_dropout : float, optional
            The dropout probability used in the ff network, by default 0
        attn_dropout : float, optional
            The dropout probability used in the attention network, by default 0
        do_rotary : bool, optional
            Whether to use rotary positional encoding, by default False
        ls_init : float | None, optional
            The initial value for the layerscale, by default 1
            If None, then no layerscale is applied
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Submodules
        attn = Attention(dim, num_heads, attn_dropout, do_rotary)
        ff = SwiGLUNet(dim, ff_mult * dim, ff_dropout)

        # Residual blocks
        self.res_attn = Residual(attn, dim, ctxt_dim, ls_init)
        self.ff = Residual(ff, dim, ctxt_dim, ls_init)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None, **kwargs) -> T.Tensor:
        x = self.res_attn(x, ctxt=ctxt, **kwargs)
        return self.ff(x, ctxt=ctxt)


class DecoderBlock(nn.Module):
    """Building block for the Transformer Decoder containing SA+CA+FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_mult: int = 2,
        ff_dropout: float = 0,
        attn_dropout: float = 0,
        num_heads: int = 8,
        do_rotary: bool = False,
        ls_init: float | None = 1,
    ) -> None:
        """Initialise the decoder block."""
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Submodules
        s_attn = Attention(dim, num_heads, attn_dropout, do_rotary)
        c_attn = Attention(dim, num_heads, attn_dropout)
        ff = SwiGLUNet(dim, ff_mult * dim, ff_dropout)

        # Residual blocks
        self.s_attn = Residual(s_attn, dim, ctxt_dim, ls_init)
        self.c_attn = Residual(c_attn, dim, ctxt_dim, ls_init)
        self.ff = Residual(ff, dim, ctxt_dim, ls_init)

    def forward(
        self,
        x: T.Tensor,
        *,  # Indicates that kv is a required argument
        kv: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the decoder block."""
        x = self.s_attn(x, mask, None, None, attn_mask, attn_bias, ctxt=ctxt)
        x = self.c_attn(x, mask, kv, kv_mask, None, None, ctxt=ctxt)
        return self.ff(x, ctxt=ctxt)


class Transformer(nn.Module):
    """Simple transformer stack of encoder or decoder blocks.

    Includes option to add registers from Vision Transformers Need Registers 2309.16588.

    Singe register can be thought of as the class token.
    """

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
        layer_config: dict | None = None,
        inpt_dim: None | int = None,
        outp_dim: None | int = None,
        use_decoder: bool = False,
        do_packed: bool = False,
        unpack_output: bool = True,
    ) -> None:
        """Parameters
        ----------
        dim : int, optional
            The dimension of the model, by default 128.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
        num_layers : int, optional
            The number of layers in the transformer, by default 6.
        max_seq_len : int, optional
            The maximum sequence length, by default 0.
            Needed for absolute encoding.
        num_registers : int, optional
            The number of registers to add to the input, by default 0.
        do_input_linear : bool, optional
            Whether to add an input linear layer, by default False.
            Will decouple the input dimension from the transformer dimension.
        do_output_linear : bool, optional
            Whether to add an output linear layer, by default False.
            Will decouple the output dimension from the transformer dimension.
        do_absolute_enc : bool, optional
            Whether to add absolute encoding, by default False.
            Must provide max_seq_len if True.
        do_final_norm : bool, optional
            Whether to add a final layer norm, by default False.
        inpt_dim : None | int, optional
            The input dimension, by default None.
            If None, then will be set to dim.
        outp_dim : None | int, optional
            The output dimension, by default None.
            If None, then will be set to dim.
        use_decoder : bool, optional
            Whether to use the decoder blocks, by default False.
        do_packed : bool, optional
            Whether to use the packed varlen attention, by default False.
        unpack_output : bool, optional
            Whether to unpack the output, by default True.
        layer_config : dict | None, optional
            The configuration for the encoder/decoder blocks, by default None.
        """
        super().__init__()

        # Check the inputs
        assert not (use_decoder and do_packed), "Packed only supports encoder blocks!"
        assert not (do_absolute_enc and max_seq_len == 0), "Define max_seq_len!"

        # Safe Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_registers = num_registers
        self.num_layers = num_layers
        self.do_final_norm = do_final_norm
        self.do_input_linear = do_input_linear
        self.do_output_linear = do_output_linear
        self.do_absolute_enc = do_absolute_enc
        self.layer_config = layer_config
        self.inpt_dim = inpt_dim if do_input_linear else dim
        self.outp_dim = outp_dim if do_output_linear else dim
        self.use_decoder = use_decoder
        self.do_packed = do_packed
        self.unpack_output = unpack_output

        # Base repeated transformer layers
        lyr = DecoderBlock if use_decoder else EncoderBlock
        self.layers = nn.ModuleList([
            lyr(dim, ctxt_dim, **layer_config) for _ in range(num_layers)
        ])

        # Optional layers and features
        if self.do_input_linear:
            self.linear_embed = nn.Linear(inpt_dim, dim)
        if self.do_final_norm:
            self.final_norm = nn.LayerNorm(dim)
        if self.do_output_linear:
            self.linear_out = nn.Linear(dim, outp_dim)
        if self.num_registers:
            self.registers = nn.Parameter(T.randn((1, self.num_registers, dim)) * 1e-3)
        if self.do_absolute_enc:
            self.abs_enc = nn.Parameter(pos_embed(dim, max_seq_len, num_registers))

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Project and encode.

        Why are these seperate?
        - Added flexibility for doing something to the inputs (replacing with null)
          once they are projected into the transformer dimension.
        """
        return self.encode(self.project(x), **kwargs)

    def project(self, x: T.Tensor) -> T.Tensor:
        """Project the input to the transformer dimension and add absolute encoding."""
        if self.do_input_linear:
            x = self.linear_embed(x)
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]
        return x

    def encode(
        self,
        x: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
        kv: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        """Pass through all layers of the transformer."""
        if self.num_registers:
            x, mask, attn_mask, attn_bias = add_registers(
                x,
                self.registers,
                mask,
                attn_mask,
                attn_bias,
                add_to_send=(kv is None) or self.use_decoder,
            )
        if self.do_packed and "culens" not in kwargs:
            x, ctxt, culens, maxlen = pack(x, mask, ctxt)
            kwargs["culens"] = culens  # Add to the kwargs for the forward pass
            kwargs["maxlen"] = maxlen
            if kv is not None and "kv_culens" not in kwargs:
                kv, _, kv_culens, kv_maxlen = pack(kv, kv_mask, None)
                kwargs["kv_culens"] = kv_culens
                kwargs["kv_maxlen"] = kv_maxlen
        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
                ctxt=ctxt,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
                kv=kv,
                kv_mask=kv_mask,
                **kwargs,
            )
        if self.do_final_norm:
            x = self.final_norm(x)
        if self.do_output_linear:
            x = self.linear_out(x)
        if self.do_packed:
            if self.unpack_output:
                return unpack(x, mask)
            return x, culens, maxlen
        return x

    def remove_registers(self, x: T.Tensor) -> T.Tensor:
        """Remove the registers from the front of the input."""
        return x[:, : self.num_registers], x[:, self.num_registers :]

    def get_combined_mask(self, mask: T.BoolTensor | None) -> T.BoolTensor:
        """Get a mask which can be used for the combined register+sequence tensor."""
        if self.num_registers == 0:
            return mask
        if mask is None:
            return None
        return F.pad(mask, (self.num_registers, 0), value=True)


class CrossAttentionEncoder(Transformer):
    """Permutation equivariant encoder with linear N computational expense."""

    def __init__(self, num_tokens: int = 16, **kwargs) -> None:
        super().__init__(use_decoder=False, do_packed=False, **kwargs)

        # Attributes
        self.num_tokens = num_tokens

        # The learnable global tokens
        self.global_tokens = nn.Parameter(T.randn((1, num_tokens, self.dim)))

        # The standard encoder layers are used as the distribution layers
        self.pool_layers = nn.ModuleList([
            EncoderBlock(self.dim, self.ctxt_dim, **self.layer_config)
            for _ in range(self.num_layers)
        ])

    def encode(
        self, x: T.Tensor, mask: T.BoolTensor | None = None, **kwargs
    ) -> T.Tensor:
        """Change only the encode operation."""
        if self.num_registers:
            x, mask, _, _ = add_registers(x, self.registers, mask)

        # Expand the global token so it can be broadcasted for the whole batch
        g = self.global_tokens.expand(x.size(0), -1, self.dim)

        # Pool and distribute
        for pool_layers, dist_layers in zip(
            self.pool_layers, self.layers, strict=False
        ):
            g = pool_layers(g, kv=x, kv_mask=mask, **kwargs)
            x = dist_layers(x, kv=g, **kwargs)

        # Perform the same final steps
        if self.do_final_norm:
            x = self.final_norm(x)
        if self.do_output_linear:
            x = self.linear_out(x)
        return x


class ClassAttentionPooling(nn.Module):
    """Pooling operation that uses attention."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 1,
        layer_config: dict | None = None,
        do_input_linear: bool = False,
        do_output_linear: bool = False,
        do_final_norm: bool = False,
        outp_dim: int | None = None,
        inpt_dim: int | None = None,
        do_packed: bool = False,
    ) -> None:
        """Initialise the class attention pooling.

        Parameters
        ----------
        dim : int
            The dimension of the model
        ctxt_dim : int
            The dimension of the context
        num_layers : int
            The number of layers in the pooling network
        layer_config : dict | None
            The configuration for the transformer encoder layers
        do_input_linear : bool, optional
            Whether to add an input linear layer, by default False.
            Will decouple the input dimension from the transformer dimension.
        do_output_linear : bool, optional
            Whether to add an output linear layer, by default False.
            Will decouple the output dimension from the transformer dimension.
        do_final_norm : bool, optional
            Whether to add a final layer norm, by default False.
        inpt_dim : None | int, optional
            The input dimension, by default None.
            If None, then will be set to dim.
        outp_dim : None | int, optional
            The output dimension, by default None.
            If None, then will be set to dim.
        do_packed: bool, optional
            Whether to use the packed varlen attention, by default False.
        """
        super().__init__()

        # Defaults
        layer_config = layer_config or {}

        # Attributes
        self.dim = dim
        self.num_layers = num_layers
        self.ctxt_dim = ctxt_dim
        self.layer_config = layer_config
        self.do_input_linear = do_input_linear
        self.do_output_linear = do_output_linear
        self.do_final_norm = do_final_norm
        self.outp_dim = outp_dim if do_output_linear else dim
        self.inpt_dim = inpt_dim if do_input_linear else dim
        self.do_packed = do_packed

        # The learnable global token
        self.global_token = nn.Parameter(T.randn((1, 1, self.dim)))

        # The cross attention pooling layers
        self.layers = nn.ModuleList([
            EncoderBlock(self.dim, ctxt_dim, **self.layer_config)
            for _ in range(num_layers)
        ])

        # Optional layers
        if self.do_input_linear:
            self.linear_embed = nn.Linear(self.inpt_dim, self.dim)
        if self.do_final_norm:
            self.final_norm = nn.LayerNorm(self.dim)
        if self.do_output_linear:
            self.linear_out = nn.Linear(self.dim, outp_dim)

    def forward(
        self, x: T.Tensor, mask: T.BoolTensor | None = None, **kwargs
    ) -> T.Tensor:
        """Perform class attention pooling on a sequence."""
        # Project the input
        if self.do_input_linear:
            x = self.linear_embed(x)

        if self.do_packed:
            culens = kwargs["culens"]
            maxlen = kwargs["maxlen"]
            B = culens.size(0) - 1
            g = self.global_token.squeeze(1).expand(B, self.dim)
            kwargs["culens"] = T.arange(B + 1, device=culens.device, dtype=culens.dtype)
            kwargs["maxlen"] = 1
            kwargs["kv_culens"] = culens
            kwargs["kv_maxlen"] = maxlen
            for layer in self.layers:
                g = layer(g, kv=x, kv_mask=mask, **kwargs)

        else:
            g = self.global_token.expand(x.shape[0], 1, self.dim)
            for layer in self.layers:
                g = layer(g, kv=x, kv_mask=mask, **kwargs)
            g.squeeze_(-2)  # Pop out the sequence dimension

        # Optional final layers
        if self.do_final_norm:
            g = self.final_norm(g)
        if self.do_output_linear:
            g = self.linear_out(g)

        return g


class TransformerVectorEncoder(nn.Module):
    """Combination of Encoder+ClassAttention to produce a vector given a set.

    By convention the intermediate resizing layer is given to the class attention
    """

    def __init__(
        self,
        *,
        inpt_dim: int = 128,
        ctxt_dim: int = 0,
        outp_dim: int = 128,
        encoder_config: dict | None = None,
        classattention_config: dict | None = None,
        do_packed: bool = False,
    ) -> None:
        super().__init__()

        # Defaults
        encoder_config = encoder_config or {}
        classattention_config = classattention_config or {}

        # Attributes
        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.outp_dim = outp_dim
        self.do_packed = do_packed

        # Modules
        self.encoder = Transformer(
            inpt_dim=inpt_dim,
            ctxt_dim=ctxt_dim,
            do_input_linear=True,
            do_output_linear=False,
            **encoder_config,
        )
        self.pool = ClassAttentionPooling(
            inpt_dim=self.encoder.outp_dim,
            ctxt_dim=ctxt_dim,
            outp_dim=outp_dim,
            do_input_linear=True,
            do_output_linear=True,
            **classattention_config,
        )

        # We want this entire setup to be packed for optimised training
        self.set_packed(do_packed)

    def set_packed(self, do_packed: bool) -> None:
        """Set the packed attribute of the encoder and pooling layers."""
        self.do_packed = do_packed
        self.encoder.do_packed = do_packed
        self.pool.do_packed = do_packed
        self.encoder.unpack_output = not do_packed

    def forward(self, x: T.Tensor, mask: T.Tensor, **kwargs) -> T.Tensor:
        if self.do_packed:
            enc, culens, maxlen = self.encoder(x, mask=mask, **kwargs)
            return self.pool(enc, mask=mask, culens=culens, maxlen=maxlen)
        enc = self.encoder(x, mask=mask, **kwargs)
        mask = self.encoder.get_combined_mask(mask)  # Might have gained registers
        return self.pool(enc, mask=mask, **kwargs)


class WrappedTransformer(nn.Module):
    """Wrap a transformer with input and output embedding MLPs."""

    def __init__(
        self,
        *,
        inpt_dim: int,
        outp_dim: int,
        transformer: partial,
        ctxt_dim: int = 0,
        edge_dim: int = 0,
        node_embd_config: dict | None = None,
        outp_embd_config: dict | None = None,
        ctxt_embd_config: dict | None = None,
        edge_embd_config: dict | None = None,
    ) -> None:
        super().__init__()

        # Defaults
        node_embd_config = node_embd_config or {}
        outp_embd_config = outp_embd_config or {}
        ctxt_embd_config = ctxt_embd_config or {}
        edge_embd_config = edge_embd_config or {}

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

        # The input and output embedding network
        self.node_embd = MLP(
            inpt_dim=self.inpt_dim,
            outp_dim=self.transformer.inpt_dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = MLP(
            inpt_dim=self.dim,
            outp_dim=self.transformer.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

        # Edge embedding network (optional)
        if self.edge_dim:
            self.edge_emdb = MLP(
                inpt_dim=self.edge_dim,
                ctxt_dim=self.ctxt_out,
                outp_dim=self.transformer.layers[0].num_heads,
                **edge_embd_config,
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
        x = self.transformer(x, mask=mask, ctxt=ctxt, attn_bias=edge, **kwargs)
        return self.outp_embd(x, ctxt)
