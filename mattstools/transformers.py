"""
Some classes to describe transformer architectures
"""

import math
from typing import Any, List

from copy import deepcopy

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from mattstools.modules import DenseNetwork
from mattstools.torch_utils import pass_with_mask


def make_clones(subject: Any, num_clones: int) -> List[Any]:
    """Return a list of identical deep copies of an object"""
    return [deepcopy(subject) for _ in range(num_clones)]


class MultiHeadedAttentionBlock(nn.Module):
    """Combines information from across its inputs using standard attention mechanism

    Takes in three sequences with dim: (batch, sqeuence, features)
    - q: The primary sequence queries (determines output sequence length)
    - k: The attending sequence keys (determines incoming information)
    - v: The attending sequence values

    It should be noted that in 99% of all transformer applications the tensors
    k and v ARE the same!
        - q is the input sequence being updated
        - k and v are the secondary sequences providing information to update q

    When q == k(v) this is a SELF attention operation
    When q != k(v) this is a Cross attention operation

    ===

    Uses three linear layers to embed the sequences:
    - q = q_linear * q
    - k = k_linear * k
    - v = v_linear * v

    Outputs are reshaped to add a head dimension! Then transposed to allow the matmul!
    - dim: batch, heads, sequence, features

    Next it passes these through the scaled dot product attention step
    - Attn(k, q, v) = V softmax(q kT / sqrt(d))
    - Softmax is done row-wise for multiple parallel heads

    Flatten out the head dimension
    - dim: batch, sequence, features*heads

    For simplicity, all tensors, inputs and outputs, have the same number of features
    which is defined by model_dim!!!
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
    ):
        """Init method for AttentionBlock

        args:
            model_dim: The dimension of the model
        kwargs:
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
        """
        super().__init__()

        ## Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        ## Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        ## Initialise the weight matrices
        self.q_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.k_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.v_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.out_linear = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        q: T.Tensor,
        k: T.Tensor = None,
        v: T.Tensor = None,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
        attn_mask: T.BoolTensor = None,
    ) -> T.Tensor:
        """
        args:
            q: The main sequence queries (determines the output length)
        kwargs:
            k: The incoming information keys
            v: The incoming information values
            a_mask: Shows which elements of the main sequence are real
            b_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
        """

        ## If the key and value tensors are not set they copy q
        k = k if k is not None else q
        v = v if v is not None else q
        q_mask = q_mask if q_mask is not None else T.ones_like(q[..., 0], dtype=T.bool)
        kv_mask = kv_mask if kv_mask is not None else T.ones_like(k[..., 0], dtype=T.bool)

        ## Store the batch size, useful for reshaping
        b_size = q.size(0)

        ## First generate the q, k, v embeddings, break final head dimension in 2
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q = pass_with_mask(q, self.q_linear, q_mask).view(shape)
        k = pass_with_mask(k, self.k_linear, kv_mask).view(shape)
        v = pass_with_mask(v, self.v_linear, kv_mask).view(shape)

        ## Transpose to get dimensions: b,h,s,f (required for matmul)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        ## Perform the matrix multiplication
        scores = T.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.model_dim)

        ## Calculate the full attention mask and mask away the scores
        mask = q_mask.unsqueeze(-1) * kv_mask.unsqueeze(-2)
        if attn_mask is not None:
            mask *= attn_mask
        mask = mask.unsqueeze(-3)
        scores = scores.masked_fill(~mask, -T.inf)

        ## Apply the softmax function per head feature
        scores = F.softmax(scores, dim=-1)
        scores = scores.masked_fill(~mask, 0)

        ## Finally multiply these scores by the output
        scores = T.matmul(scores, v)

        ## Concatenate the all of the heads together: bs, seq, feature
        scores = scores.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        ## Pass through final attention layer
        scores = pass_with_mask(scores, self.out_linear, q_mask)

        return scores


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2 style arcitecture.

    It contains:
    - self-attention-block
    - A feedforward network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
        ff_kwargs: dict = None,
    ) -> None:
        """Init method for TransformerEncoderLayer

        args:
            attn_kwargs: Keyword arguments for attention block
            ff_kwargs: Keyword arguments for feed forward network
        """
        super().__init__()

        ## Default dict arguments
        ff_kwargs = ff_kwargs or {}

        ## The basic blocks
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.self_attn = MultiHeadedAttentionBlock(model_dim, num_heads)
        self.feed_forward = DenseNetwork(model_dim, outp_dim=model_dim, **ff_kwargs)

        ## The normalisation layers
        self.norm1 = nn.LayerNorm(self.self_attn.model_dim)
        self.norm2 = nn.LayerNorm(self.feed_forward.inpt_dim)

    def forward(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.self_attn(self.norm1(x), q_mask=mask, kv_mask=mask)
        x = x + pass_with_mask(self.norm2(x), self.feed_forward, mask)
        return x


# class TransformerEncoder(nn.Module):
#     """A stack of N transformer encoder layers"""

#     def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int) -> None:
#         """Init function for the TransformerEncoder

#         args:
#             encoder_layer: A single encoder layer which will be copied multiple times
#             num_layers: The number of encoder layers to use for the stack
#         """

#         super().__init__()
#         self.layers = nn.ModuleList(make_clones(encoder_layer, num_layers))
#         self.num_layers = num_layers
#         self.final_norm = nn.LayerNorm(encoder_layer.size)

#     def forward(
#         self,
#         inpt: T.Tensor,
#         mask: T.Tensor = None,
#         inpt_key_padding_mask: T.Tensor = None,
#     ):
#         """Pass the input through all layers sequentially

#         args:
#             inpt: The sequence provided to the encoder
#             mask: The mask for the inpt sequence
#             inpt_key_padding_mask: The input mask for the tensor

#         """


def main():
    """Main script for debugging"""

    ## Parameters
    b_size = 2
    seq_size = 3
    num_heads = 2
    model_dim = 4
    ff_kwargs = {"num_blocks": 1, "drp": 0.1, "hddn_dim": 128}

    ## Create the inputs
    seq = T.rand((b_size, seq_size, model_dim))
    mask = seq[..., 0] != 0

    ## Manually kill the last element in each sequence so we can test padding
    seq[:, -1] = 0
    mask[:, -1] = False

    ## Create the block
    mha_block = TransformerEncoderLayer(model_dim, num_heads, ff_kwargs)

    ## Pass the sequence through the block
    outputs = mha_block.forward(seq, mask)

    print(outputs)


if __name__ == "__main__":
    main()
