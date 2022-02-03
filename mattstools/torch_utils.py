"""
Mix of utility functions specifically for pytorch
"""

from typing import Iterable, Tuple, Union

import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schd


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name"""
    if name == "relu":
        return nn.ReLU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu":
        return nn.SiLU(),
    if name == "selu":
        return nn.SELU(),
    if name == "sigm":
        return nn.Sigmoid(),
    if name == "tanh":
        return nn.Tanh(),
    raise ValueError("No activation function with name: ", name)


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

    if name in ["", "none", "None"]:
        return None
    elif name == "cosann":
        return schd.CosineAnnealingLR(opt, steps_per_epoch, **dict_copy)
    elif name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(opt, steps_per_epoch, **dict_copy)
    elif name == "oneshot":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
        )
    else:
        raise ValueError("No scheduler with name: ", name)


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


def reparam_trick(tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Apply the reparam trick to split a tensor into means and devs take a sample
    - Used primarily in variational autoencoders
    """
    means, lstds = T.chunk(tensor, 2, dim=-1)
    latents = means + T.randn_like(means) * lstds.exp()
    return latents, means, lstds
