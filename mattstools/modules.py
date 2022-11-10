"""
Collection of pytorch modules that make up the common networks used in my projects
"""

from multiprocessing.sharedctypes import Value
from typing import Optional, Union

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

from .bayesian import BayesianLinear
from .torch_utils import get_act, get_nrm, pass_with_mask, masked_pool, smart_cat


class MLPBlock(nn.Module):
    """
    A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        do_bayesian: bool = False,
    ) -> None:
        """Init method for MLPBlock

        Parameters
        ----------
        inpt_dim : int
            The number of features for the input layer
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        n_layers : int, optional
            A string indicating the name of the activation function, by default 1
        act : str, optional
            A string indicating the name of the normalisation, by default "lrlu"
        nrm : str, optional
            The dropout probability, 0 implies no dropout, by default "none"
        drp : float, optional
            Add to previous output, only if dim does not change, by default 0
        do_res : bool, optional
            The number of transform layers in this block, by default False
        do_bayesian : bool, optional
            If to fill the block with bayesian linear layers, by default False
        """
        super().__init__()

        ## Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        ## If this layer includes an additive residual connection
        self.do_res = do_res and (inpt_dim == outp_dim)

        ## Initialise the block layers as a module list
        self.block = nn.ModuleList()
        for n in range(n_layers):

            ## Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            ## Linear transform, activation, normalisation, dropout
            self.block.append(
                BayesianLinear(lyr_in, outp_dim)
                if do_bayesian
                else nn.Linear(lyr_in, outp_dim)
            )
            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        ## Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError(
                "Was expecting contextual information but none has been provided!"
            )
        temp = T.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt

        ## Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        ## Add the original inputs again for the residual connection
        if self.do_res:
            temp = temp + inpt

        return temp

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the block"""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and context
    injection layers"""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        do_bayesian: bool = False,
    ) -> None:
        """Initialise the DenseNetwork.

        Parameters
        ----------
        inpt_dim : int
            The number of input neurons
        outp_dim : int, optional
            The number of output neurons. If none it will take from inpt or hddn,
            by default 0
        ctxt_dim : int, optional
            The number of context features. The context feature use is determined by
            ctxt_type, by default 0
        hddn_dim : Union[int, list], optional
            The width of each hidden block. If a list it overides depth, by default 32
        num_blocks : int, optional
            The number of hidden blocks, can be overwritten by hddn_dim, by default 1
        n_lyr_pbk : int, optional
            The number of transform layers per hidden block, by default 1
        act_h : str, optional
            The name of the activation function to apply in the hidden blocks,
            by default "lrlu"
        act_o : str, optional
            The name of the activation function to apply to the outputs,
            by default "none"
        do_out : bool, optional
            If the network has a dedicated output block, by default True
        nrm : str, optional
            Type of normalisation (layer or batch) in each hidden block, by default "none"
        drp : float, optional
            Dropout probability for hidden layers (0 means no dropout), by default 0
        do_res : bool, optional
            Use resisdual-connections between hidden blocks (only if same size),
            by default False
        ctxt_in_inpt : bool, optional
            Include the ctxt tensor in the input block, by default True
        ctxt_in_hddn : bool, optional
            Include the ctxt tensor in the hidden blocks, by default False
        do_bayesian : bool, optional
            Create the network with bayesian linear layers, by default False

        Raises
        ------
        ValueError
            If the network was given a context input but both ctxt_in_inpt and
            ctxt_in_hddn were False
        """
        super().__init__()

        ## Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_hddn and not ctxt_in_inpt:
                raise ValueError("Network has context inputs but nowhere to use them!")

        ## We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if isinstance(hddn_dim, list):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        ## Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        ## Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h,
            nrm=nrm,
            drp=drp,
            do_bayesian=do_bayesian,
        )

        ## All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm,
                        drp=drp,
                        do_res=do_res,
                        do_bayesian=do_bayesian,
                    )
                )

        ## Output block (optional and there is no normalisation, dropout or context)
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
                outp_dim=self.outp_dim,
                act=act_o,
                do_bayesian=do_bayesian,
            )

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor = Optional[None]) -> T.Tensor:
        """Pass through all layers of the dense network"""

        ## Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        ## Pass through each hidden block
        for h_block in self.hidden_blocks:  ## Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  ## block was initialised with a ctxt dim

        ## Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i+1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure"""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.outp_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.outp_dim)
        return string


class DeepSet(nn.Module):
    """A deep set network that can provide attention pooling"""

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

        ## Dict default arguments
        feat_net_kwargs = feat_net_kwargs or {}
        attn_net_kwargs = attn_net_kwargs or {}
        post_net_kwargs = post_net_kwargs or {}

        ## For the attention network the default output must be set to 1
        ## The dense network default output is the same as the input
        if "outp_dim" not in attn_net_kwargs:
            attn_net_kwargs["outp_dim"] = 1

        ## Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.pool_type = pool_type
        self.attn_type = attn_type

        ## Create the feature extraction network
        self.feat_net = DenseNetwork(
            self.inpt_dim, ctxt_dim=self.ctxt_dim, **feat_net_kwargs
        )

        ## For an attention deepset
        if self.pool_type == "attn":

            ## Create the attention network
            self.attn_net = DenseNetwork(
                self.inpt_dim, ctxt_dim=self.ctxt_dim, **attn_net_kwargs
            )

            ## Check that the dimension of each head makes internal sense
            self.n_heads = self.attn_net.outp_dim
            assert self.feat_net.outp_dim % self.n_heads == 0
            self.head_dim = self.feat_net.outp_dim // self.n_heads

        ## Create the post network to update the pooled features of the set
        self.post_net = DenseNetwork(
            self.feat_net.outp_dim, outp_dim, ctxt_dim=self.ctxt_dim, **post_net_kwargs
        )

    def forward(
        self, inpt: T.tensor, mask: T.BoolTensor, ctxt: Union[T.Tensor, list] = None
    ):
        """The expected shapes of the inputs are
        - tensor: batch x setsize x features
        - mask: batch x setsize
        - ctxt: batch x features
        """

        ## Combine the context information if it is a list
        if isinstance(ctxt, list):
            ctxt = smart_cat(ctxt)

        ## Pass the non_zero values through the feature network
        feat_outs = pass_with_mask(inpt, self.feat_net, mask, context=ctxt)

        ## For attention
        if self.pool_type == "attn":
            attn_outs = pass_with_mask(
                inpt,
                self.attn_net,
                mask,
                context=ctxt,
                padval=0 if self.attn_type == "raw" else -T.inf,
            )

            ## Apply either a softmax for weighted mean or softplus for weighted sum
            if self.attn_type == "mean":
                attn_outs = F.softmax(attn_outs, dim=-2)
            elif self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)

            ## Broadcast the attention to get the multiple poolings and sum
            attn_outs = (
                attn_outs.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(2)
            )
            feat_outs = (feat_outs * attn_outs).sum(dim=-2)

        ## For the other types of pooling use the masked pool method
        else:
            feat_outs = masked_pool(self.pool_type, feat_outs, mask)

        ## Pass the pooled information through post network and return
        return self.post_net(feat_outs, ctxt)


class GRF(Function):
    """A gradient reversal function
    - The forward pass is the identity function
    - The backward pass multiplies the upstream gradients by -1
    """

    @staticmethod
    def forward(ctx, inpt, alpha):
        """Pass inputs without chaning them"""
        ctx.alpha = alpha
        return inpt.clone()

    @staticmethod
    def backward(ctx, grads):
        """Inverse the gradients"""
        alpha = ctx.alpha
        neg_grads = -alpha * grads
        return neg_grads, None


class GRL(nn.Module):
    """A gradient reversal layer.
    This layer has no parameters, and simply reverses the gradient
    in the backward pass.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = T.tensor(alpha, requires_grad=False)

    def forward(self, inpt):
        """Pass to the GRF"""
        return GRF.apply(inpt, self.alpha)
