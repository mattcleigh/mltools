"""Some classes to describe transformer architectures."""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F

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


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network."""

    def __init__(self, dim: int, hddn_dim: int, ctxt_dim: int = 0) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim + ctxt_dim, 2 * hddn_dim)
        self.lin2 = nn.Linear(hddn_dim, dim)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        x = attach_context(x, ctxt)
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(F.silu(x1) * x2)


class EncoderLayer(nn.Module):
    """Simple and constrained transformer encoder layer."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        num_heads: int = 8,
        ff_mult: int = 2,
        do_rotary_enc: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        # Class attributes
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.do_rotary_enc = do_rotary_enc
        self.attn_dim = dim // num_heads

        # Model layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_in = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)
        self.ff = SwiGLUNet(dim, ff_mult * dim, ctxt_dim=ctxt_dim)
        if self.do_rotary_enc:
            self.rotary = RotaryEmbedding(dim)

        # Add flags / pointers to the pre-residual layers to allow for initialisation
        self.pre_residual_layers = [self.attn_out, self.ff.lin2]

    def forward(
        self,
        x: T.Tensor,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        # Normalise and get projections
        q, k, v = self.attn_in(self.norm1(x)).chunk(3, -1)

        # Break final dim, transpose to get dimensions: B,H,Seq,Hdim
        shape = (x.shape[0], -1, self.num_heads, self.attn_dim)
        q, k, v = map(lambda t: t.view(shape).transpose(1, 2), (q, k, v))

        # Perform the attention
        a_mask = merge_masks(kv_mask, attn_mask, attn_bias, x)
        a_out = F.scaled_dot_product_attention(q, k, v, attn_mask=a_mask)

        # Apply rotary positional encoding on the q and k tensors
        if self.do_rotary_enc:
            q, k = self.rotary(q, k)

        # Concatenate the all of the heads together to get shape: B,Seq,F
        shape = (x.shape[0], -1, self.dim)
        a_out = a_out.transpose(1, 2).contiguous().view(shape)

        # Apply attention residual update
        x = self.attn_out(a_out) + x

        # Normalise and pass through ff net with the context
        return self.ff(self.norm2(x), ctxt) + x


class TransformerEncoder(nn.Module):
    """Simple and constrained transformer encoder."""

    def __init__(
        self,
        *,
        inpt_dim: int,
        outp_dim: int = 0,
        dim: int = 128,
        num_layers: int = 6,
        ctxt_dim: int = 0,
        init_method: str = "default",
        num_heads: int = 8,
        ff_mult: int = 2,
        max_seq_len: int = 0,
        do_absolute_enc: bool = False,
        do_rotary_enc: bool = False,
    ) -> None:
        super().__init__()

        # Attributes
        outp_dim = outp_dim or dim
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.dim = dim
        self.num_layers = num_layers
        self.init_method = init_method
        self.do_absolute_enc = do_absolute_enc
        self.do_rotary_enc = do_rotary_enc

        # If using absolute positional encoding then setup
        if self.do_absolute_enc:
            if max_seq_len == 0:
                raise ValueError("If using absolute encoding then define max length!")
            self.abs_enc = nn.Parameter(T.zeros((1, max_seq_len, dim)))

        # The main transformer layers
        self.te_layers = nn.ModuleList(
            [
                EncoderLayer(dim, ctxt_dim, num_heads, ff_mult, do_rotary_enc)
                for _ in range(num_heads)
            ]
        )

        # Input and output projection layers (output only exists if size change)
        self.in_proj = nn.Linear(inpt_dim, dim)
        self.out_proj = nn.Linear(dim, outp_dim) if outp_dim != dim else nn.Identity()

        # Change the weight initialisation in the te blocks based on depth
        for d, layer in enumerate(self.te_layers):
            param_init(layer, d, init_method)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        # Input projections
        x = self.in_proj(x)

        # Add the positional encoding
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]

        # Pass through the layers
        for layer in self.te_layers:
            x = layer(x, **kwargs)

        return self.out_proj(x)
