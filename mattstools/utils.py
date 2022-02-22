"""
General mix of utility functions
"""

from functools import reduce
import operator
import json
import yaml
import argparse
from pathlib import Path
from typing import Any, Union
import numpy as np

from pickle import load, dump

from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
)


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
        """Return the current average"""
        return self.sum / self.count


def standardise(data, means, stds):
    """Standardise data by using mean subraction and std division"""
    return (data - means) / (stds + 1e-8)


def unstandardise(data, means, stds):
    """Undo a standardisation operation by multiplying by std and adding mean"""
    return data * stds + means


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

        ## If both are dicts, then implement recursion
        elif dict_in_upt and dict_in_src:
            merged[key] = merge_dict(source[key], update[key])

        ## Otherwise one is a dict and the other is a leaf, so fail!
        else:
            raise ValueError(
                f"Trying to merge dicts but {key} is a leaf node in one not other"
            )

    return merged


def print_dict(dic: dict, indent: int = 1) -> None:
    """Recursively print a dictionary using json

    args:
        dic: The dictionary
        indent: The spacing/indent to do for nested dicts
    """
    print(json.dumps(dic, indent=indent))


def get_from_dict(data_dict: dict, key_list: list) -> Any:
    """Returns a value from a nested dictionary using list of keys"""
    return reduce(operator.getitem, key_list, data_dict)


def set_in_dict(data_dict: dict, key_list: list, value: Any):
    """Sets a value in a nested dictionary using a list of keys"""
    get_from_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def key_add(pref: str, dic: dict) -> dict:
    """Adds a prefix to each key in a dictionary"""
    return {f"{pref}{key}": val for key, val in dic.items()}


def key_change(dict, old_key, new_key, new_value=None):
    """Changes the key used in a dictionary inplace only if it exists"""

    ## If the original key is not present, nothing changes
    if old_key not in dict:
        return

    ## Use the old value and pop, really just a jey change
    if new_value is None:
        dict[new_key] = dict.pop(old_key)

    ## Both a key change AND value change! Essentially a replacement
    else:
        dict[new_key] = new_value
        del dict[old_key]


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


def load_yaml_files(files: Union[list, tuple, str]) -> tuple:
    """Loads a list of files using yaml and returns a tuple of dictionaries"""

    ## If the input is not a list then it returns a dict
    if isinstance(files, (str, Path)):
        with open(files, encoding="utf-8") as f:
            return yaml.full_load(f)

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

    ## If the input is not a list then one file is saved
    if isinstance(file_names, (str, Path)):
        with open(f"{path}/{file_names}.yaml", "w") as f:
            yaml.dump(dic, f)

    ## Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    ## Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w") as f:
            yaml.dump(dic, f)


def get_scaler(name: str):
    """Return a sklearn scaler object given a name"""
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "power":
        return PowerTransformer()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal")
    raise ValueError(f"No sklearn scaler with name: {name}")


def args_into_conf(
    argp: object,
    conf: dict,
    inpt_name: str,
    dest_keychains: Union[list, str] = None,
) -> None:
    """Takes an input string and collects the attribute with that name from an object,
    then it places that value within a dictionary at certain locations defined by
    a list of destination keys chained together

    This function is specifically designed for placing commandline arguments collected
    via argparse into to certain locations within a configuration dictionary

    There are some notable behaviours:
    - The dictionary is updated INPLACE!
    - If the input is not found on the obj or it is None, then the dict is not updated
    - If the keychain is a list the value is placed in multiple locations in the dict
    - If the keychain is None, then the input is placed in the first layer of the conf
      using its name as the key

    args:
        argp: The object from which to retrive the attribute using input_name
        conf: The dictionary to be updated with this new value
        input_name: The name of the value to retrive from the argument object
        dest_keychains: A string or list of strings for desinations in the dict
        (The keychain should show breaks in keys using '/')
    """

    ## Exit if the input is not in the argp or if its value is None
    if not hasattr(argp, inpt_name) or getattr(argp, inpt_name) is None:
        return

    ## Get the value from the argparse
    val = getattr(argp, inpt_name)

    ## Do a simple replacement if the dest keychains is None
    if dest_keychains is None:
        conf[inpt_name] = val
        return

    ## For a complex keychain we use a list for consistancy
    if isinstance(dest_keychains, str):
        dest_keychains = [dest_keychains]

    ## Cycle through all of the destinations and place in the dictionary
    for dest in dest_keychains:
        set_in_dict(conf, dest.split("/"), val)
