"""
Mix of utility functions specifically for pytorch
"""

from typing import Iterable, List, Union, Tuple

import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schd
from torch.utils.data import Dataset, random_split

from geomloss import SamplesLoss

from mattstools.loss import GeomlossWrapper, MyBCEWithLogit
from mattstools.schedulers import CyclicWithWarmup


def calc_rmse(value_a: T.Tensor, value_b: T.Tensor, dim: int = 0) -> T.Tensor:
    """Calculates the RMSE without having to go through torch's warning filled mse loss
    method
    """
    return (value_a - value_b).pow(2).mean(dim=dim).sqrt()


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name"""
    if name == "relu":
        return nn.ReLU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "sigm":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax()
    raise ValueError("No activation function with name: ", name)


def empty_0dim_like(tensor: T.Tensor) -> T.Tensor:
    """Returns an empty tensor with similar size as the input but with its final
    dimension reduced to 0
    """

    ## Get all but the final dimension
    all_but_last = tensor.shape[:-1]

    ## Ensure that this is a tuple/list so it can agree with return syntax
    if isinstance(all_but_last, int):
        all_but_last = [all_but_last]

    return T.empty((*all_but_last, 0), dtype=tensor.dtype, device=tensor.device)


def get_nrm(name: str, outp_dim: int) -> nn.Module:
    """Return a 1D pytorch normalisation layer given a name and a output size"""
    if name == "batch":
        return nn.BatchNorm1d(outp_dim)
    if name == "layer":
        return nn.LayerNorm(outp_dim)
    else:
        raise ValueError("No normalistation with name: ", name)


def get_optim(optim_dict: dict, params: Iterable) -> optim.Optimizer:
    """Return a pytorch optimiser given a dict containing a name and kwargs

    args:
        optim_dict: A dictionary of kwargs used to select and configure the optimiser
        params: A pointer to the parameters that will be updated by the optimiser
    """

    ## Pop off the name and learning rate for the optimiser
    dict_copy = optim_dict.copy()
    name = dict_copy.pop("name")

    if name == "adam":
        return optim.Adam(params, **dict_copy)
    elif name == "adamw":
        return optim.AdamW(params, **dict_copy)
    elif name == "rmsp":
        return optim.RMSprop(params, **dict_copy)
    elif name == "sgd":
        return optim.SGD(params, **dict_copy)
    else:
        raise ValueError("No optimiser with name: ", name)


def get_loss_fn(name: str) -> nn.Module:
    """Return a pytorch loss function given a name"""
    if name == "none":
        return None

    ## Classification losses
    if name == "crssent":
        return nn.CrossEntropyLoss(reduction="none")
    if name == "bcewlgt":
        return MyBCEWithLogit(reduction="none")

    ## Regression losses
    if name == "huber":
        return nn.HuberLoss(reduction="none")

    ## Distribution matching losses
    if name == "engmmd":
        return GeomlossWrapper(SamplesLoss("energy"))
    if name == "sinkhorn":
        return GeomlossWrapper(SamplesLoss("sinkhorn", p=2, blur=0.01))

    else:
        raise ValueError("No standard loss function with name: ", name)


def get_sched(
    sched_dict, opt, steps_per_epoch, max_lr=None, max_epochs=None
) -> schd._LRScheduler:
    """Return a pytorch learning rate schedular given a dict containing a name and kwargs
    args:
        sched_dict: A dictionary of kwargs used to select and configure the schedular
        opt: The optimiser to apply the learning rate to
        steps_per_epoch: The number of minibatches in a single epoch
    kwargs: (only for OneCyle learning!)
        max_lr: The maximum learning rate for the one shot
        max_epochs: The maximum number of epochs to train for
    """

    ## Pop off the name and learning rate for the optimiser
    dict_copy = sched_dict.copy()
    name = dict_copy.pop("name")

    ## Pop off the number of epochs per cycle
    if "epochs_per_cycle" in dict_copy:
        epochs_per_cycle = dict_copy.pop("epochs_per_cycle")
    else:
        epochs_per_cycle = 1

    ## Use the same div_factor for cyclic with warmup
    if name == "cyclicwithwarmup":
        if "div_factor" not in dict_copy:
            dict_copy["div_factor"] = 1e4

    if name in ["", "none", "None"]:
        return None
    elif name == "cosann":
        return schd.CosineAnnealingLR(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "onecycle":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
        )
    elif name == "cyclicwithwarmup":
        return CyclicWithWarmup(
            opt, max_lr, total_steps=steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    else:
        raise ValueError("No scheduler with name: ", name)


def train_valid_split(dataset: Dataset, v_frac: float):
    """Split a pytorch dataset into a training and validation set using the random_split funciton
    args:
        dataset: The dataset to split
        v_frac: The validation fraction (0, 1)
    """
    v_size = int(v_frac * len(dataset))
    t_size = len(dataset) - v_size
    return random_split(
        dataset, [t_size, v_size], generator=T.Generator().manual_seed(42)
    )


def masked_pool(
    pool_type: str, tensor: T.Tensor, mask: T.BoolTensor, axis: int = None
) -> T.Tensor:
    """Apply pooling to a tensor based on a string argument using its mask"""

    ## Automatically get the pooling dimension from the shape of the mask
    if axis is None:
        axis = len(mask.shape) - 1
    ## Or at least ensure that the axis is a positive number
    elif axis < 0:
        axis = len(tensor.shape) - axis

    if pool_type == "max":
        tensor[~mask] = -T.inf
        return tensor.max(dim=axis)
    if pool_type == "sum":
        tensor[~mask] = 0
        return tensor.sum(dim=axis)
    if pool_type == "mean":
        tensor[~mask] = 0
        return tensor.sum(dim=axis) / (mask.sum(dim=axis, keepdim=True) + 1e-8)

    raise ValueError(f"Unknown pooling type: {pool_type}")


def smart_cat(inputs: Iterable, dim=-1):
    """A concatenation option that ensures no memory is copied if tensors are empty"""

    ## Check number of non-empty tensors in the dimension for pooling
    n_nonempt = [bool(inpt.size(dim=dim)) for inpt in inputs]

    ## If there is only one non-empty tensor then we just return it directly
    if sum(n_nonempt) == 1:
        return inputs[np.argmax(n_nonempt)]

    ## Otherwise concatenate the rest
    return T.cat(inputs, dim=dim)


def ctxt_from_mask(context: Union[list, T.Tensor], mask: T.BoolTensor) -> T.Tensor:
    """Concatenates and returns conditional information expanded and then sampled
    using a mask. The returned tensor is compressed but repeated the appropriate number
    of times for each sample. Method uses pytorch's expand function so is light on
    memory usage.

    Primarily used for repeating conditional information for deep sets or
    graph networks.

    For example, given a deep set with feature tensor [batch, nodes, features],
    a mask tensor [batch, nodes], and context information for each sample in the batch
    [batch, c_features], then this will repeat the context information the appropriate
    number of times such that new context tensor will align perfectly with tensor[mask].

    Context and mask must have the same batch dimension

    args:
        context: A tensor or a list of tensors containing the sample context info
        mask: A mask which determines the size and sampling of the context
    """

    ## Get the expanded veiw sizes
    b_size = len(mask)
    new_dims = (len(mask.shape) - 1) * [1]
    veiw_size = (b_size, *new_dims, -1)  ## Must be: (b, 1, ..., 1, features)
    ex_size = (*mask.shape, -1)

    ## If there is only one context tensor
    if not isinstance(context, list):
        return context.view(veiw_size).expand(ex_size)[mask]

    ## For multiple context tensors
    all_context = []
    for ctxt in context:
        all_context.append(ctxt.view(veiw_size).expand(ex_size)[mask])
    return smart_cat(all_context)


def pass_with_mask(
    inputs: T.Tensor,
    module: nn.Module,
    mask: T.BoolTensor,
    context: List[T.Tensor] = None,
    padval: float = 0.0,
) -> T.Tensor:
    """Pass a collection of padded tensors through a module without wasting computation
    - Only confirmed tested with mattstools DenseNet

    args:
        data: The padded input tensor
        module: The pytorch module to apply to the inputs
        mask: A boolean tensor showing the real vs padded elements of the inputs
        context: A list of context tensor per sample to be repeated for the mask
        padval: A value to pad the outputs with
    """

    ## Get the output dimension from the passed module
    if hasattr(module, "outp_dim"):
        outp_dim = module.outp_dim
    elif hasattr(module, "out_features"):
        outp_dim = module.out_features
    else:
        raise ValueError("Dont know how to infer the output dimension from the model")

    ## Create an output of the correct shape on the right device using the padval

    outputs = T.full(
        (*inputs.shape[:-1], outp_dim),
        padval,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    ## Pass only the masked elements through the network and use mask to place in out
    if context is None:
        outputs[mask] = module(inputs[mask])

    ## My networks can take in conditional information, pytorch's can not
    else:
        outputs[mask] = module(
            inputs[mask],
            ctxt = ctxt_from_mask(context, mask)
        )

    return outputs


def sel_device(dev: Union[str, T.device]) -> T.device:
    """Returns a pytorch device given a string (or a device)
    - giving cuda or gpu will run a hardware check first
    """
    ## Not from config, but when device is specified already
    if isinstance(dev, T.device):
        return dev

    ## Tries to get gpu if available
    if dev in ["cuda", "gpu"]:
        print("Trying to select cuda based on available hardware")
        dev = "cuda" if T.cuda.is_available() else "cpu"
        print(f" - {dev} selected")

    return T.device(dev)


def move_dev(
    tensor: Union[T.Tensor, tuple, list, dict], dev: Union[str, T.device]
) -> Union[T.Tensor, tuple, list, dict]:
    """Returns a copy of a tensor on the targetted device.
    This function calls pytorch's .to() but allows for values to be a
    - list of tensors
    - tuple of tensors
    - dict of tensors
    """

    ## Select the pytorch device object if dev was a string
    if isinstance(dev, str):
        dev = sel_device(dev)

    if isinstance(tensor, tuple):
        return tuple(t.to(dev) for t in tensor)
    elif isinstance(tensor, list):
        return [t.to(dev) for t in tensor]
    elif isinstance(tensor, dict):
        return {t: tensor[t].to(dev) for t in tensor}
    else:
        return tensor.to(dev)


def to_np(tensor: T.Tensor) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a
    pytorch tensor to numpy array
    - Includes gradient deletion, and device migration
    """
    return tensor.detach().cpu().numpy()


def print_gpu_info(dev=0):
    total = T.cuda.get_device_properties(dev).total_memory / 1024**3
    reser = T.cuda.memory_reserved(dev) / 1024**3
    alloc = T.cuda.memory_allocated(dev) / 1024**3
    print(f"\nTotal = {total:.2f}\nReser = {reser:.2f}\nAlloc = {alloc:.2f}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a pytorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module, norm_type: float = 2.0):
    """Return the norm of the gradients of a given model"""
    return to_np(
        T.norm(
            T.stack([T.norm(p.grad.detach(), norm_type) for p in model.parameters()]),
            norm_type,
        )
    )


def reparam_trick(tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Apply the reparameterisation trick to split a tensor into means and devs
    - Returns a sample, the means and devs as a tuple
    - Splitting is along the final dimension
    - Used primarily in variational autoencoders
    """
    means, lstds = T.chunk(tensor, 2, dim=-1)
    latents = means + T.randn_like(means) * lstds.exp()
    return latents, means, lstds


def apply_residual(rsdl_type: str, res: T.Tensor, outp: T.Tensor) -> T.Tensor:
    """Apply a residual connection by either adding or concatenating"""
    if rsdl_type == "cat":
        return smart_cat([res, outp], dim=-1)
    if rsdl_type == "add":
        return outp + res
    raise ValueError(f"Unknown residual type: {rsdl_type}")


@T.jit.script
def falling_sigmoid(x):
    return T.sigmoid(-x - 3)


def aggr_via_sparse(cmprsed: T.Tensor, mask: T.BoolTensor, reduction: str, dim: int):
    """Aggregate a compressed tensor by first blowing up to a sparse representation

    The tensor is blown up to full size such that: full[mask] = cmprsed

    Supports sum, mean, and softmax
    - mean is not supported by torch.sparse, so we use sum and the mask
    - softmax does not reduce the size of the tensor, but applies softmax along dim

    args:
        cmprsed: The nonzero elements of the compressed tensor
        mask: A mask showing where the nonzero elements should go
        reduction: A string indicating the type of reduction
        dim: Which dimension to apply the reduction
    """

    ## Create a sparse representation of the tensor
    sparse_rep = sparse_from_mask(cmprsed, mask, is_compressed=True)

    ## Apply the reduction
    if reduction == "sum":
        return T.sparse.sum(sparse_rep, dim).values()
    if reduction == "mean":
        reduced = T.sparse.sum(sparse_rep, dim)
        mask_sum = mask.sum(dim)
        mask_sum = mask_sum.unsqueeze(-1).expand(reduced.shape)[mask_sum>0]
        return reduced.values() / mask_sum
    if reduction == "softmax":
        return T.sparse.softmax(sparse_rep, dim).coalesce().values()
    else:
        raise ValueError(f"Unknown sparse reduction method: {reduction}")


def sparse_from_mask(input: T.Tensor, mask: T.BoolTensor, is_compressed: bool = False):
    """Create a pytorch sparse matrix given a tensor and a mask.
    - Shape is infered from the mask, meaning the final dim will be dense
    """
    return T.sparse_coo_tensor(
        T.nonzero(mask).t(),
        input if is_compressed else input[mask],
        device=input.device,
        dtype=input.dtype,
        requires_grad=input.requires_grad,
    ).coalesce()

def decompress(cmprsed: T.Tensor, mask: T.BoolTensor):
    """Take a compressed input and use the mask to blow it to its original shape
    such that full[mask] = cmprsed
    """
    ## We first create the zero padded tensor of the right size then replace
    full = T.zeros(
        (*mask.shape, cmprsed.shape[-1]),
        dtype=cmprsed.dtype,
        device=cmprsed.device,
    )

    ## Place the nonpadded samples into the full shape
    full[mask] = cmprsed

    return full