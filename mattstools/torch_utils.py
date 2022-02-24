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

from mattstools.loss import GANLoss, KLD2NormLoss, MyBCEWithLogit


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
        return (nn.SiLU(),)
    if name == "selu":
        return (nn.SELU(),)
    if name == "sigm":
        return (nn.Sigmoid(),)
    if name == "tanh":
        return (nn.Tanh(),)
    raise ValueError("No activation function with name: ", name)


def empty_0dim_like(tensor: T.Tensor) -> T.Tensor:
    """Returns an empty tensor BUT with its final dimension reduced to 0"""

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
    if name == "bcewlgt":
        return MyBCEWithLogit()
    if name == "crssent":
        return nn.CrossEntropyLoss()
    if name == "kld2nrm":
        return KLD2NormLoss()
    if name == "ganloss":
        return GANLoss()
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
    kwargs: (only for one shot learning!)
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

    if name in ["", "none", "None"]:
        return None
    elif name == "cosann":
        return schd.CosineAnnealingLR(
            opt, epochs_per_cycle * steps_per_epoch, **dict_copy
        )
    elif name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(
            opt, epochs_per_cycle * steps_per_epoch, **dict_copy
        )
    elif name == "onecycle":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
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

    ## Sum uses mean operation to keep values small but ignores the padding
    ## It is the same as sum / pad_length
    if pool_type == "sum":
        return tensor.mean(dim=axis)

    ## Mean pooling takes into account the mask
    if pool_type == "mean":
        return tensor.sum(dim=axis) / (mask.sum(dim=axis, keepdim=True) + 1e-8)

    raise ValueError(f"Unknown pooling type: {pool_type}")


def repeat_cat(src: T.Tensor, context: List[T.Tensor], mask: T.BoolTensor):
    """Returns a source tensor combined with a list of contextual information which is repeated
    over every dimension except the first
    - Only acts on dim=-1
    """

    ## First combine the context information together
    context = smart_cat(context, dim=-1)

    ## If it is non emtpy then we combine repeat across the batch dimension
    if T.numel(context):
        context = T.repeat_interleave(
            context,
            mask.sum(
                dim=list(range(1, len(mask.shape)))
            ),  ## Sum over all dims exept batch
        )

    ## Return the combined information
    return smart_cat([src, context], dim=-1)


def smart_cat(inputs: Iterable, dim=-1):
    """A concatenation option that ensures no memory is copied if tensors are empty"""

    ## Check number of non-empty tensors in the dimension for pooling
    n_nonempt = [bool(inpt.size(dim=dim)) for inpt in inputs]

    ## If there is only one non-empty tensor then we just return it directly
    if sum(n_nonempt) == 1:
        return inputs[np.argmax(n_nonempt)]

    ## Otherwise concatenate the rest
    return T.cat(inputs, dim=dim)


def pass_with_mask(
    data: T.Tensor,
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

    ## Create an output of the correct shape on the right device using the padval
    outputs = T.full(
        (*data.shape[:-1], module.outp_dim),
        padval,
        device=data.device,
        dtype=data.dtype,
    )

    ## Pass only the masked elements through the network and use mask to place in out
    outputs[mask] = module(data[mask])

    ## Add the contextual information (repeat according to sample multiplicity)
    ## TODO fix this
    # if context is not None:
    #    data = repeat_cat(data, context, mask)

    return outputs


def sel_device(dev: Union[str, T.device]) -> T.device:
    """Returns a pytorch device given a string (or a device)
    - includes auto option
    """
    if isinstance(dev, T.device):
        return dev
    if dev == "auto":
        return T.device("cuda" if T.cuda.is_availabel() else "cpu")
    elif dev in ["cuda", "gpu"]:
        dev = "cuda"
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
