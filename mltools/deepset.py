"""Code for a simple deep set."""

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .torch_utils import masked_pool, smart_cat


class DeepSet(nn.Module):
    """A deep set network that can provide attention pooling."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        pool_type: str = "mean",
        attn_type: str = "mean",
        feat_net_kwargs=None,
        attn_net_kwargs=None,
        post_net_kwargs=None,
    ) -> None:
        """
        args:
            inpt_dim: The number of input features
            outp_dim: The number of desired output featues
        kwargs:
            ctxt_dim: Dimension of the context information for all networks
            pool_type: The type of set pooling applied; mean, sum, max or attn
            attn_type: The type of attention; mean, sum, raw
            feat_net_kwargs: Keyword arguments for the feature network
            attn_net_kwargs: Keyword arguments for the attention network
            post_net_kwargs: Keyword arguments for the post network
        """
        super().__init__()

        # Dict default arguments
        feat_net_kwargs = feat_net_kwargs or {}
        attn_net_kwargs = attn_net_kwargs or {}
        post_net_kwargs = post_net_kwargs or {}

        # For the attention network the default output must be set to 1
        # The dense network default output is the same as the input
        if "outp_dim" not in attn_net_kwargs:
            attn_net_kwargs["outp_dim"] = 1

        # Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.pool_type = pool_type
        self.attn_type = attn_type

        # Create the feature extraction network
        self.feat_net = MLP(self.inpt_dim, ctxt_dim=self.ctxt_dim, **feat_net_kwargs)

        # For an attention deepset
        if self.pool_type == "attn":
            # Create the attention network
            self.attn_net = MLP(
                self.inpt_dim, ctxt_dim=self.ctxt_dim, **attn_net_kwargs
            )

            # Check that the dimension of each head makes internal sense
            self.n_heads = self.attn_net.outp_dim
            assert self.feat_net.outp_dim % self.n_heads == 0
            self.head_dim = self.feat_net.outp_dim // self.n_heads

        # Create the post network to update the pooled features of the set
        self.post_net = MLP(
            self.feat_net.outp_dim, outp_dim, ctxt_dim=self.ctxt_dim, **post_net_kwargs
        )

    def forward(
        self,
        inpt: T.tensor,
        mask: T.BoolTensor,
        ctxt: T.Tensor | list | None = None,
    ):
        """Forward pass for deep set."""

        # Combine the context information if it is a list
        if isinstance(ctxt, list):
            ctxt = smart_cat(ctxt)

        # Pass the values through the feature network
        feat_outs = self.feat_net(inpt, ctxt)

        # For attention
        if self.pool_type == "attn":
            attn_outs = self.attn_net(inpt, ctxt)

            # Change the attention weights of the padded elements
            attn_outs[~mask] = 0 if self.attn_type == "raw" else -T.inf

            # Apply either a softmax for weighted mean or softplus for weighted sum
            if self.attn_type == "mean":
                attn_outs = F.softmax(attn_outs, dim=-2)
            elif self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)

            # Kill the nans introduced by the empty sets
            attn_outs = T.nan_to_num(attn_outs, 0)
            # attn_outs[~mask] = 0

            # Broadcast the attention to get the multiple poolings and sum
            attn_outs = (
                attn_outs.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(2)
            )
            feat_outs = (feat_outs * attn_outs).sum(dim=-2)

        # For the other types of pooling use the masked pool method
        else:
            feat_outs = masked_pool(self.pool_type, feat_outs, mask)

        # Pass the pooled information through post network and return
        return self.post_net(feat_outs, ctxt)
