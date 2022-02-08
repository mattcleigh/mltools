from typing import Union

import torch as T
import torch.nn as nn

from torch.autograd import Function

from mattstools.torch_utils import get_act, get_nrm


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense networks

    Depending on the configuration it can apply:
    - linear map
    - activation function
    - layer normalisation
    - dropout

    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        act: str = "lrlu",
        nrm: str = None,
        drp: float = 0,
    ) -> None:
        """
        args:
            inpt_dim: The number of features for the input layer
            outp_dim: The number of output features
        kwargs:
            act: A string indicating the name of the activation function
            nrm: A string indicating the name of the normalisation
            drp: The dropout probability, 0 implies no dropout
        """
        ## Applies the
        super().__init__()

        ## Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim

        ## Initialise the block as a sequential module
        block = [nn.Linear(inpt_dim, outp_dim)]
        if act:
            block += [get_act(act)]
        if nrm:
            block += [get_nrm(nrm, outp_dim)]
        if drp:
            block += [nn.Dropout(drp)]
        self.block = nn.Sequential(*block)

    def forward(self, tensor: T.Tensor) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
        """
        return self.block(tensor)


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks"""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = None,
        depth: int = 2,
        width: Union[int, list] = 32,
        act_h: str = "lrlu",
        act_o: str = None,
        do_out: bool = True,
        nrm: str = None,
        drp: float = 0,
    ) -> None:
        """
        args:
            inpt_dim: The number of input neurons
        kwargs:
            outp_dim: The number of output neurons, if none it will take inpt or hddn
            depth: The number of hidden layers
            width: The number of neurons in each hidden layer (if list, overides depth)
            act_h: The name of the activation function to apply in the hidden layers
            act_o: The name of the activation function to apply to the outputs
            do_out: If the network has a dedicated output block
            nrm: Type of normalisation, either layer or batch in each hidden block
            drp: Dropout probability for hidden layers (0 means no dropout)
        """
        super().__init__()

        ## We store the input, output dimensions to query them later
        self.inpt_dim = inpt_dim
        self.blocks = nn.ModuleList()

        ## Calculate the number of neurons in each hidden layer if not defined
        width = width if isinstance(width, list) else depth * [width]

        ## Calculating the output dimension of the network
        if do_out:
            self.outp_dim = outp_dim or inpt_dim
        else:
            self.outp_dim = width[-1]

        ## Input block
        self.blocks.append(MLPBlock(inpt_dim, width[0], act_h, nrm, drp))

        ## Hidden blocks
        for w_1, w_2 in zip(width[:-1], width[1:]):
            self.blocks.append(MLPBlock(w_1, w_2, act_h, nrm, drp))

        ## Output block (no normalisation or dropout)
        if do_out:
            self.blocks.append(MLPBlock(width[-1], self.outp_dim, act_o, False, 0))

    def forward(self, tensor: T.Tensor) -> T.Tensor:
        for module in self.blocks:
            tensor = module(tensor)
        return tensor


class GRF(Function):
    """A gradient reversal function
    - The forward pass is the identity function
    - The backward pass multiplices the upstream gradients by -1
    """

    @staticmethod
    def forward(_, inputs):
        return inputs.clone()

    @staticmethod
    def backward(_, grads):
        return grads.neg(), None
