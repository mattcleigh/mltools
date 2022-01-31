"""
General mix of utility functions
"""

import yaml
import argparse
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np


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


def standardise(data, means, stds):
    """Standardise data by using mean subraction and std division"""
    return (data - means) / (stds + 1e-8)


def merge_dict(source: dict, update: dict) -> dict:
    """Merges two deep dictionaries recursively
    - Apply to small dictionaries please!
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


def signed_angle_diff(angle1, angle2):
    """Calculate diff between two angles reduced to the interval of [-pi, pi]"""
    return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi


def load_yaml_files(files: Union[list, tuple]) -> tuple:
    """Loads a list of files using yaml and returns a tuple of dictionaries"""
    opened = []

    ## Load each file using yaml
    for fnm in files:
        with open(fnm, encoding="utf-8") as f:
            opened.append(yaml.full_load(f))

    return tuple(opened)


def save_yaml_files(
    path: str, file_names: Union[list, tuple], dicts: Union[list, tuple]
) -> None:
    """Saves a collection of yaml files in a folder
    - Makes the folder if it does not exist
    """

    ## Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    ## Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w") as f:
            yaml.dump(dic, f)
