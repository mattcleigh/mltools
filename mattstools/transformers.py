"""
Some classes to describe transformer architectures
"""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .modules import DenseNetwork
from .torch_utils import pass_with_mask

def merge_masks(
    q_mask: T.BoolTensor,
    kv_mask: T.BoolTensor,
    attn_mask: T.BoolTensor,
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
):
    """Create a full attention mask which incoporates the padding information"""

    ## Create the full mask which combines the attention and padding masks!
    full_mask = None

    ## If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        q_mask = (
            q_mask
            if q_mask is not None
            else T.ones(q_shape[:-1], dtype=T.bool, device=device)
        )
        kv_mask = (
            kv_mask
            if kv_mask is not None
            else T.ones(k_shape[:-1], dtype=T.bool, device=device)
        )
        full_mask = q_mask.unsqueeze(-1) * kv_mask.unsqueeze(-2)

    ## If attention mask exists, create
    if attn_mask is not None:
        full_mask = attn_mask if full_mask is None else attn_mask * full_mask

    return full_mask


def attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    dim_key: int,
    attn_mask: T.BoolTensor = None,
    edge_weights: T.Tensor = None,
    mul_weights: bool = True,
    dropout_layer: nn.Module = None,
):
    """Apply the attention using the scaled dot product between the key query and
    key tensors, then matrix multiplied by the value.

    Note that the attention scores are ordered in recv x send, which is the opposite
    to how I usually do it for the graph network, which is send x recv

    args:
        query: Batched query sequence of tensors (b, h, s, f)
        key: Batched key sequence of tensors (b, h, s, f)
        value: Batched value sequence of tensors (b, h, s, f)
        dim_key: The dimension of the key features, used to scale the dot product
        attn_mask: The attention mask, used to blind certain combinations of k,q pairs
        edge_weights: Extra weights to combine with attention weights
        mul_weights: If the weights are multiplied into the scores (False they are add)
        dropout_layer: Optional dropout layer for the scores
    """

    ## Perform the matrix multiplication
    scores = T.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)

    ## Multiply the scores by adding in the manual weights
    if edge_weights is not None:
        if mul_weights:
            scores = scores * edge_weights.unsqueeze(-3)
        else:
            scores = scores + edge_weights.unsqueeze(-3)

    ## Mask away the scores between invalid nodes
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(-3)
        scores = scores.masked_fill(~attn_mask, -T.inf)

    ## Apply the softmax function per head feature
    scores = F.softmax(scores, dim=-1)

    ## Reinsert the mask, for the padded sequences will now have NaNs
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, 0)

    ## Apply dropout to the scores
    if dropout_layer is not None:
        scores = dropout_layer(scores)

    ## Finally multiply these scores by the output
    scores = T.matmul(scores, value)

    return scores


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
        drp: float = 0,
        mul_weights: bool = True,
    ):
        """Init method for AttentionBlock

        args:
            model_dim: The dimension of the model
        kwargs:
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
            drp: The dropout probability used in the MHA operation
            mul_weights: How extra interation weights should be used if passed
                - See attention above
        """
        super().__init__()

        ## Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.mul_weights = mul_weights

        ## Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        ## Initialise the weight matrices
        self.q_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.k_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.v_linear = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout_layer = nn.Dropout(p=drp) if drp > 0 else None
        self.out_linear = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        q: T.Tensor,
        k: T.Tensor = None,
        v: T.Tensor = None,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
        attn_mask: T.BoolTensor = None,
        edge_weights: T.Tensor = None,
    ) -> T.Tensor:
        """
        args:
            q: The main sequence queries (determines the output length)
        kwargs:
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
        """

        ## If the key and value tensors are not set they copy q
        k = k if k is not None else q
        v = v if v is not None else q

        ## Store the batch size, useful for reshaping
        b_size = q.size(0)

        ## First work out the masking situation, with padding, no peaking etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        ## First generate the q, k, v embeddings, break final head dimension in 2
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q = pass_with_mask(q, self.q_linear, q_mask).view(shape)
        k = pass_with_mask(k, self.k_linear, kv_mask).view(shape)
        v = pass_with_mask(v, self.v_linear, kv_mask).view(shape)

        ## Transpose to get dimensions: b,h,s,f (required for matmul)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        ## Calculate the new sequence values, for memory reasons overwrite q
        q = attention(
            q,
            k,
            v,
            self.model_dim,
            attn_mask=attn_mask,
            dropout_layer=self.dropout_layer,
            edge_weights=edge_weights,
            mul_weights=self.mul_weights,
        )  ## Returned shape is b,h,s,f

        ## Concatenate the all of the heads together to get shape: b,s,f
        q = q.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        ## Pass through final linear layer
        q = pass_with_mask(q, self.out_linear, q_mask)

        return q


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
        mha_kwargs: dict = None,
        ff_kwargs: dict = None,
    ) -> None:
        """Init method for TransformerEncoderLayer

        args:
            mha_kwargs: Keyword arguments for multiheaded-attention block
            ff_kwargs: Keyword arguments for feed forward network
        """
        super().__init__()

        ## Default dict arguments
        mha_kwargs = mha_kwargs or {}
        ff_kwargs = ff_kwargs or {}

        ## Save the model dim as an attribute
        self.model_dim = model_dim

        ## The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(model_dim, **mha_kwargs)
        self.feed_forward = DenseNetwork(model_dim, outp_dim=model_dim, **ff_kwargs)

        ## The normalisation layers (lots from NormFormer)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self, x: T.Tensor, mask: T.BoolTensor, edge_weights: T.BoolTensor = None
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.norm2(self.self_attn(
            self.norm1(x), q_mask=mask, kv_mask=mask, edge_weights=edge_weights
        ))
        x = x + pass_with_mask(self.norm3(x), self.feed_forward, mask)
        return x


class CrossAttentionLayer(TransformerEncoderLayer):
    """A transformer cross attention block

    It contains:
    - cross-attention-block
    - A feed forward network

    Can be seen as a type of encoder layer with an overloaded forward method to
    facilitate cross attention
    """

    def __init__(
        self, model_dim: int, mha_kwargs: dict = None, ff_kwargs: dict = None
    ) -> None:
        super().__init__(model_dim, mha_kwargs, ff_kwargs)
        self.norm0 = nn.LayerNorm(model_dim)

    # pylint: disable=arguments-differ,arguments-renamed
    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
    ) -> T.Tensor:
        "Pass through the layers of cross attention"
        kv_seq = self.norm1(kv_seq)
        q_seq = q_seq + self.norm2(self.self_attn(
            self.norm0(q_seq), kv_seq, kv_seq, q_mask=q_mask, kv_mask=kv_mask
        ))
        q_seq = q_seq + pass_with_mask(self.norm3(q_seq), self.feed_forward, q_mask)

        return q_seq


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final normalisation step"""

    def __init__(
        self,
        model_dim: int,
        num_layers: int = 3,
        mha_kwargs: dict = None,
        ff_kwargs: dict = None,
    ) -> None:
        """Init function for the TransformerEncoder

        args:
            model_dim: Feature sieze for input, output, and all intermediate layers
        kwargs:
            num_layers: Number of encoder layers used
            mha_kwargs: Keyword arguments for the mha block
            ff_kwargs: Keyword arguments for the ff network in each layer
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_kwargs, ff_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, sequence: T.Tensor, mask: T.BoolTensor = None) -> T.Tensor:
        """Pass the input through all layers sequentially"""
        for layer in self.layers:
            sequence = layer(sequence, mask)
        return self.final_norm(sequence)


class TransformerVectorEncoder(nn.Module):
    """A type of transformer encoder which procudes a single vector for the whole seq

    It does this by using several self-attention layers on the whole sequence.
    Then it introduces a class token which is initialised with small constant values
    It then uses cross attention using the class token as queries resulting in a
    single element sequence.

    Then passes this final vector through one last dense network
    """

    def __init__(
        self,
        model_dim: int = 64,
        outp_dim: int = 1,
        num_sa_blocks: int = 3,
        num_ca_blocks: int = 2,
        ctxt_dim: int = 0,
        mha_kwargs: dict = None,
        trans_ff_kwargs: dict = None,
        final_ff_kwargs: dict = None,
    ) -> None:
        """Init function for the TransformerVectorEncoder

        args:
            model_dim: Feature size for input, output, and all intermediate sequences
            outp_dim: The dimension of final output vector
        kwargs:
            num_sa_blocks: Number of self attention encoder layers
            num_ca_blocks: Number of cross/class attention encoder layers
            ctxt_dim: Dimension of context tensor introduced before final net
            mha_kwargs: Keyword arguments for all multiheaded attention layers
            trans_ff_kwargs: Keyword arguments for the ff network in each layer
            final_ff_kwargs: Keyword arguments for the ff network in each layer
        """
        super().__init__()

        ## Default dict arguments
        final_ff_kwargs = final_ff_kwargs or {}

        ## Create the class attributes
        self.model_dim = model_dim
        self.outp_dim = outp_dim
        self.num_sa_blocks = num_sa_blocks
        self.num_ca_blocks = num_ca_blocks
        self.ctxt_dim = ctxt_dim

        ## Initialise the models
        self.sa_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_kwargs, trans_ff_kwargs)
                for _ in range(num_sa_blocks)
            ]
        )
        self.ca_blocks = nn.ModuleList(
            [
                CrossAttentionLayer(model_dim, mha_kwargs, trans_ff_kwargs)
                for _ in range(num_sa_blocks)
            ]
        )
        self.final_ff = DenseNetwork(
            model_dim, outp_dim, ctxt_dim=ctxt_dim, **final_ff_kwargs
        )

        ## Initialise the class token embedding as a learnable parameter
        self.class_token = nn.Parameter(T.randn((1, 1, self.model_dim)))

    def forward(
        self,
        seq: T.Tensor,
        mask: T.BoolTensor = None,
        ctxt: T.Tensor = None,
        edge_weights: T.Tensor = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially"""

        ## Pass through the self attention encoder
        for layer in self.sa_blocks:
            seq = layer(seq, mask, edge_weights=edge_weights)

        ## Get the learned class token and expand to the batch size
        class_token = self.class_token.expand(len(seq), 1, self.model_dim)

        ## Pass through the class attention blocks
        for layer in self.ca_blocks:
            class_token = layer(class_token, seq, q_mask=None, kv_mask=mask)

        ## Pass through final dense network and return
        return self.final_ff(class_token.squeeze(), ctxt=ctxt)
