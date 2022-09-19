"""
Some classes to describe transformer architectures
"""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .modules import DenseNetwork
from .torch_utils import pass_with_mask

##TODO Support attention masks in the larger layers


def merge_masks(
    q_mask: T.BoolTensor,
    kv_mask: T.BoolTensor,
    attn_mask: T.BoolTensor,
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
):
    """Create a full attention mask which incoporates the padding information"""

    ## Create the full mask which combines the attention and padding masks
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
        full_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

    ## If attention mask exists, create
    if attn_mask is not None:
        full_mask = attn_mask if full_mask is None else attn_mask & full_mask

    return full_mask


def attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    dim_key: int,
    attn_mask: T.BoolTensor = None,
    edge_weights: T.Tensor = None,
    mul_weights: bool = False,
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
        mul_weights: If the edge weights are multiplied into the scores
            (False means they are added pre softmax)
        dropout_layer: Optional dropout layer for the scores
    """

    ## Perform the matrix multiplication
    scores = T.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)

    ## Multiply the scores by adding in the manual weights
    if edge_weights is not None:
        if mul_weights:
            scores = scores * edge_weights.transpose(-3, -1)
        else:
            scores = scores + edge_weights.transpose(-3, -1)

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
        mul_weights: bool = False,
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
        b_size = q.shape[0]

        ## First work out the masking situation, with padding, no peaking etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        ## First generate the q, k, v embeddings, break final head dimension in 2
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q = pass_with_mask(q, self.q_linear, q_mask).view(shape)
        k = pass_with_mask(k, self.k_linear, kv_mask).view(shape)
        v = pass_with_mask(v, self.v_linear, kv_mask).view(shape)

        ## Transpose to get dimensions: b,h,seq,f (required for matmul)
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

        ## Concatenate the all of the heads together to get shape: b,seq,f
        q = q.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        ## Pass through final linear layer
        q = pass_with_mask(q, self.out_linear, q_mask)

        return q


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style arcitecture.

    It contains:
    - self-attention-block
    - A feedforward network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer
    """

    def __init__(
        self, model_dim: int, mha_kwargs: dict = None, ff_kwargs: dict = None
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
        x = x + self.norm2(
            self.self_attn(
                self.norm1(x), q_mask=mask, kv_mask=mask, edge_weights=edge_weights
            )
        )
        x = x + pass_with_mask(self.norm3(x), self.feed_forward, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """A transformer dencoder layer based on the GPT-2+Normformer style arcitecture.

    It contains:
    - self-attention-block
    - cross-attention block
    - feedforward network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer
    """

    def __init__(
        self, model_dim: int, mha_kwargs: dict = None, ff_kwargs: dict = None
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
        self.cross_attn = MultiHeadedAttentionBlock(model_dim, **mha_kwargs)
        self.feed_forward = DenseNetwork(model_dim, outp_dim=model_dim, **ff_kwargs)

        ## The normalisation layers (lots from NormFormer)
        self.norm_preSA = nn.LayerNorm(model_dim)
        self.norm_pstSA = nn.LayerNorm(model_dim)
        self.norm_preC1 = nn.LayerNorm(model_dim)
        self.norm_preC2 = nn.LayerNorm(model_dim)
        self.norm_pstCA = nn.LayerNorm(model_dim)
        self.norm_preNN = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        kv_seq = self.norm_preC2(kv_seq)  ## Apply this now

        ## Apply the self attention residual update
        q_seq = q_seq + self.norm_pstSA(
            self.self_attn(self.norm_preSA(q_seq), q_mask=q_mask, kv_mask=q_mask)
        )

        ## Apply the cross attention residual update
        q_seq = q_seq + self.norm_pstCA(
            self.cross_attn(
                self.norm_preC1(q_seq),
                k=kv_seq,
                v=kv_seq,
                q_mask=q_mask,
                kv_mask=kv_mask,
            )
        )

        ## Apply the FF residual update
        q_seq = q_seq + pass_with_mask(
            self.norm_preNN(q_seq), self.feed_forward, q_mask
        )

        return q_seq


class TransformerCrossAttentionLayer(TransformerEncoderLayer):
    """A transformer cross attention layer

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
        kv_seq = self.norm0(kv_seq)
        q_seq = q_seq + self.norm2(
            self.self_attn(
                self.norm1(q_seq), kv_seq, kv_seq, q_mask=q_mask, kv_mask=kv_mask
            )
        )
        q_seq = q_seq + pass_with_mask(self.norm3(q_seq), self.feed_forward, q_mask)

        return q_seq


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final normalisation step
    Sequence to Sequence
    """

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


class TransformerDecoder(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step
    Sequence-Sequence to Sequence
    """

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
                TransformerDecoderLayer(model_dim, mha_kwargs, ff_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially"""
        for layer in self.layers:
            q_seq = layer(q_seq, kv_seq, q_mask, kv_mask)
        return self.final_norm(q_seq)


class TransformerVectorEncoder(nn.Module):
    """A type of transformer encoder which procudes a single vector for the whole seq
    Sequence to Vector

    Then the nodes (and optionally edges) are passed through several MHSA layers.
    Then a learnable class token is updated using cross attention.
    This results in a single element sequence.
    Contains a final normalisation layer

    It is non resizing, so model_dim must be used for inputs and outputs
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_sa_layers: int = 3,
        num_ca_layers: int = 2,
        mha_kwargs: dict = None,
        ff_kwargs: dict = None,
    ) -> None:
        """Init function for the TransformerVectorEncoder

        args:
            model_dim: Feature size for input, output, and all intermediate sequences
        kwargs:
            num_sa_layers: Number of self attention encoder layers
            num_ca_layers: Number of cross/class attention encoder layers
            mha_kwargs: Keyword arguments for all multiheaded attention layers
            ff_kwargs: Keyword arguments for the ff network in each layer
        """
        super().__init__()

        ## Create the class attributes
        self.model_dim = model_dim
        self.num_sa_layers = num_sa_layers
        self.num_ca_layers = num_ca_layers

        ## Initialise the models
        self.sa_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_kwargs, ff_kwargs)
                for _ in range(num_sa_layers)
            ]
        )
        self.ca_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(model_dim, mha_kwargs, ff_kwargs)
                for _ in range(num_ca_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

        ## Initialise the class token embedding as a learnable parameter
        self.class_token = nn.Parameter(T.randn((1, 1, self.model_dim)))

    def forward(
        self,
        seq: T.Tensor,
        mask: T.BoolTensor = None,
        edge_weights: T.Tensor = None,
        return_seq: bool = False,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially"""

        ## Pass through the self attention encoder
        for layer in self.sa_layers:
            seq = layer(seq, mask, edge_weights=edge_weights)

        ## Get the learned class token and expand to the batch size
        ## Use shape[0] not len as it is ONNX safe!
        class_token = self.class_token.expand(seq.shape[0], 1, self.model_dim)

        ## Pass through the class attention layers
        for layer in self.ca_layers:
            class_token = layer(class_token, seq, q_mask=None, kv_mask=mask)

        ## Pass through the final normalisation layer
        class_token = self.final_norm(class_token)

        ## Pop out the unneeded sequence dimension of 1
        class_token = class_token.squeeze(1)

        ## Return the class token and optionally the sequence as well
        if return_seq:
            return class_token, seq
        return class_token


class TransformerVectorDecoder(nn.Module):
    """A type of transformer decoder which creates a sequence given a starting
    vector and a desired mask
    Vector to Sequence

    Randomly initialises the q-sequence using the mask shape and a gaussian
    Uses the input vector as 1-long kv-sequence in decoder layers

    It is non resizing, so model_dim must be used for inputs and outputs
    """

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
            num_layers: Number of decoder layers used
            mha_kwargs: Keyword arguments for the mha block
            ff_kwargs: Keyword arguments for the ff network in each layer
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, mha_kwargs, ff_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, vec: T.Tensor, mask: T.BoolTensor = None) -> T.Tensor:
        """Pass the input through all layers sequentially"""

        ## Initialise the q-sequence randomly (adhere to mask)
        q_seq = T.randn(
            (*mask.shape, self.model_dim), device=vec.device, dtype=vec.dtype
        ) * mask.unsqueeze(-1)

        ## Reshape the vector from batch x features to batch x seq=1 x features
        vec = vec.unsqueeze(1)

        ## Pass through the decoder
        for layer in self.layers:
            q_seq = layer(q_seq, vec, q_mask=mask, kv_mask=None)
        return self.final_norm(q_seq)


class FullTransformerVectorEncoder(nn.Module):
    """A TVE with added input and output embedding networks
    Sequence to Vector

    1)  First it embeds the tokens into a higher dimensional space based on model_dim
        using a dense network.
    2)  If there are edge features these are embedded into a single dimensional space
            This is a very optional step which most will want to ignore but it is what
            ParT used! https://arxiv.org/abs/2202.03772
    3)  Then it passes these through a TVE to get a single vector output
    4)  Finally is passes the vector through an embedding network
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        edge_dim: int = 0,
        tve_kwargs: dict = None,
        node_embd_kwargs: dict = None,
        edge_embd_kwargs: dict = None,
        outp_embd_kwargs: dict = None,
    ) -> None:
        """Init function for the TransformerVectorEncoder

        args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of of the final output vector
        kwargs:
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            edge_dim: Dim. of the input edge features
            tve_kwargs: Keyword arguments to pass to the TVE constructor
            node_embd_kwargs: Keyword arguments for node ff embedder
            edge_embd_kwargs: Keyword arguments for edge ff embedder
            outp_embd_kwargs: Keyword arguments for output ff embedder
        """
        super().__init__()

        ## Safe default dict arguments
        node_embd_kwargs = node_embd_kwargs or {}
        edge_embd_kwargs = edge_embd_kwargs or {}
        outp_embd_kwargs = outp_embd_kwargs or {}
        tve_kwargs = tve_kwargs or {}

        ## Create the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim

        ## Initialise the TVE, the main part of this network
        self.tve = TransformerVectorEncoder(**tve_kwargs)
        self.model_dim = self.tve.model_dim

        ## Initialise all embedding networks
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_dim,
            **node_embd_kwargs
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **outp_embd_kwargs
        )

        ## Initialise the edge embedding network (optional)
        if self.edge_dim:
            self.edge_embd = DenseNetwork(
                inpt_dim=self.edge_dim,
                outp_dim=self.tve.sa_layers[0].self_attn.num_heads,
                ctxt_dim=self.ctxt_dim,
                **edge_embd_kwargs
            )

    def forward(
        self,
        nodes: T.Tensor,
        mask: T.BoolTensor = None,
        ctxt: T.Tensor = None,
        edges: T.Tensor = None,
        return_seq: bool = False,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially"""

        ## Embed the nodes
        nodes = pass_with_mask(nodes, self.node_embd, mask, ctxt)

        ## Embed the edges (optional)
        if edges is not None:
            edges = self.edge_embd(edges, ctxt)

        ## If we want the sequence and the output then return both
        if return_seq:
            outp, nodes = self.tve(nodes, mask, edges, return_seq)
            outp = self.outp_embd(outp, ctxt)
            return outp, nodes

        ## If we only want the output, then overwrite the nodes to save space
        nodes = self.tve(nodes, mask, edges, return_seq)
        nodes = self.outp_embd(nodes, ctxt)
        return nodes


class FullTransformerVectorDecoder(nn.Module):
    """A TVD with added input and output embedding networks
    Vector to Sequence

    1)  Embeds the input vector into a higher dimensional space based on model_dim
        using a dense network.
    2)  Passes this through a TVG to get a sequence output
    3)  Passes the sequence through an embedding dense network with vector as context
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        tvd_kwargs: dict = None,
        vect_embd_kwargs: dict = None,
        outp_embd_kwargs: dict = None,
    ) -> None:
        """Init function for the TransformerVectorEncoder

        args:
            inpt_dim: Dim. of the input vector
            outp_dim: Dim. of each element of the output sequence
        kwargs:
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            tvd_kwargs: Keyword arguments to pass to the TVD constructor
            vec_embd_kwargs: Keyword arguments for vector ff embedder
            out_embd_kwargs: Keyword arguments for output node ff embedder
        """
        super().__init__()

        ## Safe default dict arguments
        tvd_kwargs = tvd_kwargs or {}
        vect_embd_kwargs = vect_embd_kwargs or {}
        outp_embd_kwargs = outp_embd_kwargs or {}

        ## Create the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        ## Initialise the TVE, the main part of this network
        self.tvg = TransformerVectorDecoder(**tvd_kwargs)
        self.model_dim = self.tvg.model_dim

        ## Initialise all embedding networks
        self.vec_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_dim,
            **vect_embd_kwargs
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **outp_embd_kwargs
        )

    def forward(
        self, vec: T.Tensor, mask: T.BoolTensor, ctxt: T.Tensor = None
    ) -> T.Tensor:
        """Pass the input through all layers sequentially"""
        vec = self.vec_embd(vec, ctxt=ctxt)
        nodes = self.tvg(vec, mask)
        nodes = pass_with_mask(nodes, self.outp_embd, mask, context=ctxt)
        return nodes
