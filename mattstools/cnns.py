"""Code for everything convolutional."""


import logging
from typing import Mapping, Optional

import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.functional import group_norm, interpolate, scaled_dot_product_attention

from .modules import DenseNetwork
from .torch_utils import append_dims, get_act

log = logging.getLogger(__name__)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.data.zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def drop_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D droupout module."""
    if dims == 1:
        return nn.Dropout(*args, **kwargs)
    elif dims == 2:
        return nn.Dropout2d(*args, **kwargs)
    elif dims == 3:
        return nn.Dropout3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ConditionedModule(nn.Module):
    """Base class for models that need context when processing data."""

    pass


class ConditionedSequential(nn.Sequential):
    """Sequential model that can provide context to each layers if required."""

    def forward(self, inpt, ctxt):
        """Pass through all submodules."""
        for module in self:
            if isinstance(module, ConditionedModule):
                inpt = module(inpt, ctxt)
            else:
                inpt = module(inpt)
        return inpt


class AdaGN(ConditionedModule):
    """A module that implements an adaptive group normalization layer."""

    def __init__(self, ctxt_dim: int, c_out: int, nrm_groups: int, eps=1e-5) -> None:
        """
        Parameters
        ----------
        ctxt_dim : int
            The dimension of the context tensor.
        c_out : int
            The number of output channels.
        nrm_groups : int
            The number of groups for group normalization.
        eps : float, optional
            A small value added to the denominator for numerical stability.
            Default: 1e-5.
        """
        super().__init__()
        self.ctxt_dim = ctxt_dim
        self.c_out = c_out
        self.num_groups = nrm_groups
        self.eps = eps
        self.layer = nn.Linear(ctxt_dim, c_out * 2)

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        """Apply conditioning to inputs."""
        scale, shift = self.layer(ctxt).chunk(2, dim=-1)
        scale = append_dims(scale, inpt.ndim) + 1  # + 1 to not kill on init
        shift = append_dims(shift, inpt.ndim)
        inpt = group_norm(inpt, self.num_groups, eps=self.eps)
        return T.addcmul(shift, inpt, scale)

    def __str__(self) -> str:
        return f"AdaGN({self.ctxt_dim}, {self.c_out})"

    def __repr__(self):
        return str(self)


class ResNetBlock(ConditionedModule):
    """A residual convolutional block.

    Can change channel dimensions but not spacial.
    All convolutions are stride 1 with padding 1.
    May also take in some context tensor which is injected using AdaGN.
    Forward pass applies the following:
    - AdaGN->Act->Conv->AdaGN->Act->Drop->0Conv + skip_connection
    """

    def __init__(
        self,
        inpt_channels: int,
        ctxt_dim: int = 0,
        outp_channels: int | None = None,
        kernel_size: int = 3,
        dims: int = 2,
        act: str = "lrlu",
        drp: float = 0,
        nrm_groups: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        inpt_channels : int
            The number of input channels.
        ctxt_dim : int, optional
            The dimension of the context tensor. Default: 0.
        outp_channels : int, optional
            The number of output channels. Default: None.
        kernel_size : int, optional
            The size of the convolution kernel. Default: 3.
        dims : int, optional
            The number of dimensions (2 is an image). Default: 2.
        act : str, optional
            The activation function to use. Default: "lrlu".
        drp : float, optional
            The dropout rate. Default: 0.
        nrm_groups : int, optional
            The number of groups for group normalization. Default: 1.
        """
        super().__init__()

        # Class attributes
        self.inpt_channels = inpt_channels
        self.outp_channels = outp_channels or inpt_channels
        self.ctxt_dim = ctxt_dim

        # The method for normalisation is where the context is injected
        def norm(c_out) -> AdaGN | nn.GroupNorm:
            if ctxt_dim:
                return AdaGN(ctxt_dim, c_out, nrm_groups)
            return nn.GroupNorm(nrm_groups, c_out)

        # Create the main layer structure of the network
        self.layers = ConditionedSequential(
            norm(inpt_channels),
            get_act(act),
            conv_nd(dims, inpt_channels, outp_channels, kernel_size, padding=1),
            norm(outp_channels),
            get_act(act),
            drop_nd(dims, drp),
            zero_module(
                conv_nd(dims, outp_channels, outp_channels, kernel_size, padding=1)
            ),
        )

        # Create the skip connection, using a 1x1 conv to change channel size
        if self.inpt_channels == self.outp_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, inpt_channels, outp_channels, 1)

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Pass through the layers with the residual connection."""
        return self.layers(inpt, ctxt) + self.skip_connection(inpt)

    def __str__(self) -> str:
        string = f"ResNetBlock({self.inpt_channels}, {self.outp_channels}"
        if self.ctxt_dim:
            string += f", ctxt={self.ctxt_dim}"
        string += ")"
        return string

    def __repr__(self) -> str:
        return str(self)


class MultiHeadedAttentionBlock(ConditionedModule):
    """A multi-headed self attention block that allows spatial positions to attend to
    each other.

    This layer essentailly flattens the image's spacial dimensions, making it a sequence
    where the length equals the original resolution. The dimension of each element of
    the sequence is the number of each channels.

    Then the message passing occurs, which is permutation invariant, using the exact
    same operations as a standard transformer except we use 1x1 convolutions instead of
    linear projections (same maths, but optimised performance)
    """

    def __init__(
        self,
        inpt_channels: int,
        inpt_shape: tuple,
        ctxt_dim: int = 0,
        num_heads: int = 1,
        nrm_groups: int = 1,
        do_pos_encoding: bool = True,
    ) -> None:
        super().__init__()

        # Ensure that the number of channels is divisible by the number of heads
        assert inpt_channels % num_heads == 0

        # Class attributes
        self.inpt_channels = inpt_channels
        self.inpt_shape = inpt_shape
        self.ctxt_dim = ctxt_dim
        self.num_heads = num_heads
        self.do_pos_encoding = do_pos_encoding

        # The learnable positional encodings which are fixed
        self.pos_enc = nn.Parameter(T.zeros(inpt_channels, *inpt_shape))

        # The method for normalisation is where the context is injected
        if ctxt_dim:
            self.norm = AdaGN(ctxt_dim, inpt_channels, nrm_groups)
        else:
            self.norm = nn.GroupNorm(nrm_groups, inpt_channels)

        # The convoluation layers used in the attention operation
        self.qkv = conv_nd(len(inpt_shape), inpt_channels, inpt_channels * 3, 1)
        self.out_conv = zero_module(
            conv_nd(len(inpt_shape), inpt_channels, inpt_channels, 1)
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Apply the model the message passing, context tensor is not used."""
        b, c, *spatial = inpt.shape

        # Pass through the layers to encode and get the qkv values
        qkv = inpt
        if self.do_pos_encoding:
            qkv = qkv + self.pos_enc.unsqueeze(0)
        if self.ctxt_dim:
            qkv = self.norm(qkv, ctxt)
        qkv = self.qkv(qkv)

        # Flatten out the spacial dimensions them swap to get: B, 3N, HxW, h_dim
        qkv = qkv.view(b, self.num_heads * 3, c // self.num_heads, -1).transpose(-1, -2)
        q, k, v = T.chunk(qkv.contiguous(), 3, dim=1)

        # Now we can use the attention operation from the transformers package
        a_out = scaled_dot_product_attention(q, k, v)

        # Concatenate the all of the heads together to get back to: B, c, H, W
        a_out = a_out.transpose(-1, -2).contiguous().view(b, c, *spatial)

        # Apply redidual update and bring back spacial dimensions
        return inpt + self.out_conv(a_out)


class DoublingConvNet(nn.Module):
    """A very simple convolutional neural network which halves the spacial dimension
    with each block while doubling the number of channels.

    Attention operations occur after a certain number of downsampling steps

    Downsampling is performed using 2x2 average pooling

    Ends with a dense network
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        min_size: int = 2,
        max_depth: int = 8,
        n_blocks_per_layer: int = 1,
        attn_below: int = 8,
        start_channels: int = 32,
        max_channels: int = 256,
        resnet_config: dict | None = None,
        attn_config: dict | None = None,
        dense_config: dict | None = None,
    ) -> None:
        super().__init__()

        # Safe dict defaults
        resnet_config = resnet_config or {}
        attn_config = attn_config or {}
        dense_config = dense_config or {}

        # Class attributes
        self.inpt_size = inpt_size
        self.inpt_channels = inpt_channels
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # The downsampling layer (not learnable)
        dims = len(inpt_size)
        stride = 2 if self.dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(dims, kernel_size=stride, stride=stride)

        # The first conv layer sets up the starting channel size
        self.first_block = nn.Sequential(
            conv_nd(inpt_channels, start_channels, 1), nn.SiLU()
        )

        # Keep track of the spacial dimensions for each input and output layer
        inp_size = np.array(inpt_size)
        out_size = inp_size // 2
        inp_c = start_channels
        out_c = start_channels * 2

        # Start creating the levels (should exit but max 100 for safety)
        resnet_blocks = []
        for depth in range(max_depth):
            lvl_layers = []

            # Add the resnet blocks
            for j in range(n_blocks_per_layer):
                lvl_layers.append(
                    ResNetBlock(
                        inpt_channels=inp_c if j == 0 else out_c,
                        ctxt_dim=ctxt_dim,
                        outp_channels=out_c,
                        dims=dims,
                        **resnet_config,
                    )
                )

            # Add an optional attention block if we downsampled enough
            if max(inpt_size) <= attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=out_c,
                        inpt_shape=inp_size,
                        ctxt_dim=ctxt_dim,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            resnet_blocks.append(nn.ModuleList(lvl_layers))

            # Exit if the next iteration would lead too small spacial dimensions
            if min(out_size) // 2 < min_size:
                break

            # Update the dimensions for the next iteration
            inp_size = out_size
            out_size = out_size // 2  # Halve the spacial dims
            inp_c = out_c
            out_c = min(out_c * 2, max_channels)  # Double the channels until max

        # Combine layers into a module list
        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        # Create the dense network
        self.dense = DenseNetwork(
            inpt_dim=np.prod(out_size) * out_c,
            outp_dim=outp_dim,
            ctxt_dim=ctxt_dim,
            **dense_config,
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None):
        """Forward pass of the network."""

        # Pass through the first convolution layer to embed the channel dimension
        inpt = self.first_block(inpt)

        # Pass through the ResNetBlocks and the downsampling
        for level in self.resnet_blocks:
            for layer in level:
                inpt = layer(inpt, ctxt)
            inpt = self.down_sample(inpt)

        # Flatten and pass through final dense network and return
        inpt = T.flatten(inpt, start_dim=1)

        return self.dense(inpt, ctxt)


class UNet(nn.Module):
    """A image to image mapping network which halves the spacial dimension with each
    block while doubling the number of channels, before building back up to the original
    resolution.

    Attention operations occur after a certain number of downsampling steps

    Downsampling is performed using 2x2 average pooling Upsampling is performed using
    nearest neighbour
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_channels: int,
        ctxt_dim: int = 0,
        ctxt_img_channels: int = 0,
        min_size: int = 8,
        max_depth: int = 8,
        n_blocks_per_layer: int = 1,
        attn_below: int = 8,
        start_channels: int = 32,
        max_channels: int = 128,
        zero_out: bool = False,
        resnet_config: Optional[Mapping] = None,
        attn_config: Optional[Mapping] = None,
        ctxt_embed_config: Optional[Mapping] = None,
        use_ctxt_embedder: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        inpt_size : list
            The size of the input image.
        inpt_channels : int
            The number of channels in the input image.
        outp_channels : int
            The number of channels in the output image.
        ctxt_dim : int, optional
            The dimension of the context input. Default is 0.
        ctxt_img_channels : int, optional
            The number of channels from the context image. Default is 0.
        min_size : int, optional
            The minimum size of the spacial dimensions. Default is 8.
        max_depth : int, optional
            The maximum depth of the network. Default is 8.
        n_blocks_per_layer : int, optional
            The number of ResNet blocks per layer. Default is 1.
        attn_below : int, optional
            The maximum size of spacial dimensions for attention operations.
            Default is 8.
        start_channels : int, optional
            The number of channels at the start of the network. Default is 32.
        max_channels : int, optional
            The maximum number of channels in the network. Default is 128.
        zero_out : bool, optional
            Whether to zero out the last block. Default is False.
        resnet_config : Optional[Mapping], optional
            Configuration for ResNet blocks. Default is None.
        attn_config : Optional[Mapping], optional
            Configuration for attention blocks. Default is None.
        ctxt_embed_config : Optional[Mapping], optional
            Configuration for context embedding network. Default is None.
        use_ctxt_embedder: bool
            Use a network to embed the context
        """
        super().__init__()

        # Safe dict defaults
        resnet_config = resnet_config or {}
        attn_config = attn_config or {}
        ctxt_embed_config = ctxt_embed_config or {}

        # Class attributes
        self.inpt_size = inpt_size
        self.inpt_channels = inpt_channels
        self.outp_channels = outp_channels
        self.ctxt_dim = ctxt_dim
        self.ctxt_img_channels = ctxt_img_channels

        # The downsampling layer and upscaling layers (not learnable)
        dims = len(inpt_size)
        stride = 2 if dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(dims, kernel_size=stride, stride=stride)
        self.up_sample = nn.Upsample(scale_factor=2)

        # If there is a context input, maybe you want a network to embed it
        if ctxt_dim:
            if use_ctxt_embedder:
                self.context_embedder = DenseNetwork(
                    inpt_dim=ctxt_dim, **ctxt_embed_config
                )
                emb_ctxt_size = self.context_embedder.outp_dim
            else:
                self.context_embedder = nn.Identity()
                emb_ctxt_size = ctxt_dim
        else:
            emb_ctxt_size = 0

        # The first and last conv layer sets up the starting channel size
        self.first_block = nn.Sequential(
            conv_nd(dims, inpt_channels + ctxt_img_channels, start_channels, 1),
            nn.SiLU(),
        )
        self.last_block = nn.Sequential(
            nn.SiLU(), conv_nd(dims, start_channels, outp_channels, 1)
        )
        if zero_out:
            self.last_block = zero_module(self.last_block)

        # Keep track of the spacial dimensions for each input and output layer
        inp_size = [np.array(inpt_size)]
        out_size = [np.array(inpt_size) // 2]
        inp_c = [start_channels]
        out_c = [min(start_channels * 2, max_channels)]

        # The encoder blocks are ResNet->(attn)->Downsample
        encoder_blocks = []
        for depth in range(max_depth):
            lvl_layers = []

            # Add the resnet blocks
            for j in range(n_blocks_per_layer):
                lvl_layers.append(
                    ResNetBlock(
                        inpt_channels=inp_c[-1] if j == 0 else out_c[-1],
                        outp_channels=out_c[-1],
                        ctxt_dim=emb_ctxt_size,
                        **resnet_config,
                    )
                )

            # Add an optional attention block if we downsampled enough
            if max(inp_size[-1]) <= attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=out_c[-1],
                        inpt_shape=inp_size[-1],
                        ctxt_dim=emb_ctxt_size,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            encoder_blocks.append(nn.ModuleList(lvl_layers))

            # Exit if the next it would lead an output with small spacial dimensions
            if min(out_size[-1]) // 2 < min_size:
                break

            # Update the dimensions for the NEXT iteration
            inp_size.append(out_size[-1])
            out_size.append(out_size[-1] // 2)  # Halve the spacial dimensions
            inp_c.append(out_c[-1])
            out_c.append(min(out_c[-1] * 2, max_channels))  # Double the channels

        # Combine layers into a module list
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # The middle part of the UNet
        self.middle_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    inpt_channels=out_c[-1],
                    outp_channels=out_c[-1],
                    ctxt_dim=emb_ctxt_size,
                    **resnet_config,
                ),
                MultiHeadedAttentionBlock(
                    inpt_channels=out_c[-1],
                    inpt_shape=out_size[-1],
                    ctxt_dim=emb_ctxt_size,
                    **attn_config,
                ),
                ResNetBlock(
                    inpt_channels=out_c[-1],
                    outp_channels=out_c[-1],
                    ctxt_dim=emb_ctxt_size,
                    **resnet_config,
                ),
            ]
        )

        # Loop in reverse to create the decoder blocks
        decoder_blocks = []
        for depth in range(len(out_c) - 1, -1, -1):
            lvl_layers = []

            # Add the resnet blocks
            for j in range(n_blocks_per_layer):
                lvl_layers.append(
                    ResNetBlock(
                        inpt_channels=out_c[depth] * 2 if j == 0 else inp_c[depth],
                        outp_channels=inp_c[depth],
                        ctxt_dim=emb_ctxt_size,
                        **resnet_config,
                    )
                )

            # Add the attention layer at the appropriate levels
            if max(inp_size[depth]) < attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=inp_c[depth],
                        inpt_shape=inp_size[depth],
                        ctxt_dim=emb_ctxt_size,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            decoder_blocks.append(nn.ModuleList(lvl_layers))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(
        self,
        inpt: T.Tensor,
        ctxt: T.Tensor | None = None,
        ctxt_img: T.Tensor | None = None,
    ) -> T.Tensor:
        """Forward pass of the network."""

        # Some context tensors come from labels and must match the same type as inpt
        if ctxt.dtype != inpt.dtype:
            ctxt = ctxt.type(inpt.dtype)

        # Make sure the input size is expected
        if inpt.shape[-1] != self.inpt_size[-1]:
            log.warning("Input image does not match the training sample!")

        # Embed the context tensor
        if self.ctxt_dim:
            ctxt = self.context_embedder(ctxt)

        # Combine the input with the context image
        if self.ctxt_img_channels:
            if ctxt_img.shape != inpt.shape:
                ctxt_img = interpolate(ctxt_img, inpt.shape[-2:])
            inpt = T.cat([inpt, ctxt_img], 1)

        # Pass through the first convolution layer to embed the channel dimension
        inpt = self.first_block(inpt)

        # Pass through the encoder
        dec_outs = []
        for level in self.encoder_blocks:
            for layer in level:
                inpt = layer(inpt, ctxt)
            dec_outs.append(inpt)  # Save the output to the buffer
            inpt = self.down_sample(inpt)  # Apply the downsampling

        # Pass through the middle blocks
        for block in self.middle_blocks:
            inpt = block(inpt, ctxt)

        # Pass through the decoder blocks
        for level in self.decoder_blocks:
            inpt = self.up_sample(inpt)  # Apply the upsampling
            inpt = T.cat([inpt, dec_outs.pop()], dim=1)  # Concat with buffer
            for layer in level:
                inpt = layer(inpt, ctxt)

        # Pass through the final layer
        inpt = self.last_block(inpt)

        return inpt
