"""
General utility functions used in this module
"""

from typing import Iterable, Tuple, Union
import argparse

import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schd


class RunningAverage:
    """A class which tracks the sum and data count so can calculate
    the running average on demand
    """

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        """Resets all statistics"""
        self.__init__()

    def update(self, val: float, quant: int = 1) -> None:
        """Updates the running average with a new batched average"""
        self.sum += val * quant
        self.count += quant

    @property
    def avg(self) -> float:
        """Calculate the current average"""
        return self.sum / self.count


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name"""
    return {
        "relu": nn.ReLU(),
        "lrlu": nn.LeakyReLU(0.1),
        "silu": nn.SiLU(),
        "selu": nn.SELU(),
        "sigm": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }[name]


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
    if name == "adamw":
        return optim.AdamW(params, **dict_copy)
    if name == "rmsp":
        return optim.RMSprop(params, **dict_copy)
    if name == "sgd":
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
    if name == "cosann":
        return schd.CosineAnnealingLR(opt, steps_per_epoch, **dict_copy)
    if name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(opt, steps_per_epoch, **dict_copy)
    if name == "oneshot":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
        )
    raise ValueError(f"No scheduler with name {name}")


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


def get_stats(inpt: T.Tensor, dim=0) -> T.Tensor:
    """Calculate the mean and dev of a sample and return the concatenated results"""
    mean = T.mean(inpt, dim=dim)
    stdv = T.std(inpt, dim=dim)
    return T.cat([mean, stdv])


def standardise(data, means, stds):
    """Standardise data by using mean subraction and std division"""
    return (data - means) / (stds + 1e-8)


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


def merge_dict(source: dict, update: dict) -> dict:
    """Merges two deep dictionaries recursively, only apply to small dictionaries please!
    args:
        source: The source dict, will be updated in place
        update: Will be used to overwrite and append values to the source
    """
    ## Make a copy of the source dictionary
    merged = source.copy()

    ## Cycle through all of the keys in the update
    for key in update:
        ## If the key not in the source then add move on
        if key not in merged:
            merged[key] = update[key]
            continue

        ## Check type of variable
        dict_in_upt = isinstance(update[key], dict)
        dict_in_src = isinstance(source[key], dict)

        ## If neither are a dict, then simply replace the leaf variable
        if not dict_in_upt and not dict_in_src:
            merged[key] = update[key]

        ## If both are dicts, then impliment recursion
        elif dict_in_upt and dict_in_src:
            merged[key] = merge_dict(source[key], update[key])

        ## Otherwise one is a dict and the other is a leaf, so fail!
        else:
            raise ValueError(
                f"Trying to merge dicts but {key} is a leaf node in one not other"
            )

    return merged


def interweave(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    """Combine two arrays by alternating along the first dimension
    args:
        a: array to take even indices
        b: array to take odd indices
    returns:
        combined array
    """
    arr_comb = np.empty(
        (arr_1.shape[0] + arr_2.shape[0], *arr_1.shape[1:]), dtype=arr_1.dtype
    )
    arr_comb[0::2] = arr_1
    arr_comb[1::2] = arr_2
    return arr_comb


def str2bool(mystring: str) -> bool:
    """Convert a string object into a boolean"""
    if isinstance(mystring, bool):
        return mystring
    if mystring.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if mystring.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_np(tensor: T.Tensor) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a
    pytorch tensor to numpy array
    - Includes gradient deletion, and device migration
    """
    return tensor.detach().cpu().numpy()


def reparam_trick(tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Apply the reparam trick to split a tensor into means and devs take a sample
    - Used primarily in variational autoencoders
    """
    means, lstds = T.chunk(tensor, 2, dim=-1)
    latents = means + T.randn_like(means) * lstds.exp()
    return latents, means, lstds


def print_gpu_info(dev=0):
    total = T.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    reser = T.cuda.memory_reserved(dev) / 1024 ** 3
    alloc = T.cuda.memory_allocated(dev) / 1024 ** 3
    print(f"\nTotal = {total:.2f}\nReser = {reser:.2f}\nAlloc = {alloc:.2f}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a pytorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def signed_angle_diff(angle1, angle2):
    """Calculate diff between two angles reduced to the interval of [-pi, pi]"""
    return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
