"""
General mix of utility functions
"""

import argparse
from pathlib import Path
from typing import Union, Tuple, Any
from functools import reduce
import operator
import json
import yaml
import numpy as np

from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
)


def get_standard_configs(
    def_train: str = "config/data.yaml",
    def_net: str = "config/net.yaml",
    def_data: str = "config/train.yaml",
) -> Tuple[dict, dict, dict]:
    """Loads, modifies, and returns three configuration dictionaries using command
    line arguments for a basic training setup
    - For configuring the dataset, network and training scheme
    - One can set the names of the config files to load
    - One can then modify the returned dictionaries using additional arguments

    Any extra arguments can be manually added in each project
    args:
        def_train: Path to default train config file
        def_net: Path to default net config file
        def_data: Path to default data config file
    """
    print("Loading config files")

    parser = argparse.ArgumentParser()

    ## Config files
    parser.add_argument(
        "--data_conf",
        type=str,
        default=def_train,
        help="Path to config file to use as a template for data preperation",
    )
    parser.add_argument(
        "--net_conf",
        type=str,
        default=def_net,
        help="Path to config file to use as a template for network construction",
    )
    parser.add_argument(
        "--train_conf",
        type=str,
        default=def_data,
        help="Path to config file to use as a template for training scheme",
    )

    ## Network base kwargs
    parser.add_argument(
        "--name", type=str, help="The name to use for saving the network"
    )
    parser.add_argument(
        "--save_dir", type=str, help="The directory to use for saving the network"
    )

    ## Learning scheme
    parser.add_argument(
        "--lr", type=float, help="Learning rate given to optimiser (max for cyclic)"
    )
    parser.add_argument(
        "--sched_name", type=float, help="Name of the learning rate scheduler"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Number of samples per training batch"
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of parellel threads to load the batches"
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        help="Resume the latest training checkpoint",
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--tqdm_quiet",
        type=str2bool,
        help="Silences the tqdm output for logging",
        nargs="?",
        const=True,
        default=False,
    )

    ## Load the arguments
    args, _ = parser.parse_known_args()

    ## Change the paths to previous configs if resuming, otherwise keep defaults
    if args.resume:
        args.data_conf = Path(args.save_dir, args.name, "config/data.yaml")
        args.net_conf = Path(args.save_dir, args.name, "config/net.yaml")
        args.train_conf = Path(args.save_dir, args.name, "config/train.yaml")

    ## Load the config dictionaries
    data_conf, net_conf, train_conf = load_yaml_files(
        [args.data_conf, args.net_conf, args.train_conf]
    )

    ## Remove all yaml anchors
    print("- Removing yaml anchors")
    data_conf = remove_keys_starting_with(data_conf, "__")
    net_conf = remove_keys_starting_with(net_conf, "__")
    train_conf = remove_keys_starting_with(train_conf, "__")

    ## Most args need manual placement in the configuration dicts
    args_into_conf(args, net_conf, "name", "base_kwargs/name")
    args_into_conf(args, net_conf, "save_dir", "base_kwargs/save_dir")
    args_into_conf(args, train_conf, "batch_size", "loader_kwargs/batch_size")
    args_into_conf(args, train_conf, "num_workers", "loader_kwargs/num_workers")
    args_into_conf(args, train_conf, "sched_name", "trainer_kwargs/sched_dict/name")
    args_into_conf(args, train_conf, "lr", "trainer_kwargs/optim_dict/lr")
    if args.resume:
        args_into_conf(args, train_conf, "resume", "trainer_kwargs/resume")
    if args.tqdm_quiet:
        args_into_conf(args, train_conf, "tqdm_quiet", "trainer_kwargs/tqdm_quiet")

    return data_conf, net_conf, train_conf


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
        source: The source dict, will be copied (not modified)
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


def get_from_dict(data_dict: dict, key_list: list, default=None) -> Any:
    """Returns a value from a nested dictionary using list of keys"""
    try:
        return reduce(operator.getitem, key_list, data_dict)
    except KeyError:
        return default


def set_in_dict(data_dict: dict, key_list: list, value: Any):
    """Sets a value in a nested dictionary using a list of keys"""
    get_from_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def key_prefix(pref: str, dic: dict) -> dict:
    """Adds a prefix to each key in a dictionary"""
    return {f"{pref}{key}": val for key, val in dic.items()}


def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Changes the key used in a dictionary inplace only if it exists"""

    ## If the original key is not present, nothing changes
    if old_key not in dic:
        return

    ## Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    ## Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]


def remove_keys_starting_with(dic: dict, match: str) -> dict:
    """Removes all keys from the dictionary if they start with
    - Returns a copy of the dictionary
    """
    return {key: val for key, val in dic.items() if key[: len(match)] != match}


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


def sum_other_axes(array: np.ndarray, axis: int) -> np.ndarray:
    """Applies numpy sum to all axes except one in an array"""
    axes_for_sum = [i for i in range(len(array.shape))]
    axes_for_sum.pop(axis)
    return array.sum(axis=tuple(axes_for_sum))


def mid_points(array: np.ndarray) -> np.ndarray:
    """Return the midpoints of an array, one smaller"""
    return (array[1:] + array[:-1]) / 2


def undo_mid(array: np.ndarray) -> np.ndarray:
    """Undo the midpoints, trying to get the bin boundaries"""
    array = np.array(array)  ## Have to include this because of pandas
    half_bw = (array[1] - array[0]) / 2  ## Assumes constant bin widths
    array = np.insert(array + half_bw, 0, array[0] - half_bw)
    return array


def chunk_given_size(a: np.ndarray, size: int, axis: int = 0) -> np.ndarray:
    """Split an array into chunks along an axis, the final chunk will be smaller"""
    return np.split(a, np.arange(size, a.shape[axis], size), axis=axis)


def apply_mask(arrays: list, mask: np.ndarray):
    """Applies a mask to a list of arrays"""
    return [a[mask] for a in arrays]


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
            return yaml.safe_load(f)

    opened = []

    ## Load each file using yaml
    for fnm in files:
        with open(fnm, encoding="utf-8") as f:
            opened.append(yaml.safe_load(f))

    return tuple(opened)


def save_yaml_files(
    path: str, file_names: Union[str, list, tuple], dicts: Union[dict, list, tuple]
) -> None:
    """Saves a collection of yaml files in a folder
    - Makes the folder if it does not exist
    """

    ## If the input is not a list then one file is saved
    if isinstance(file_names, (str, Path)):
        with open(f"{path}/{file_names}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(dicts, f, sort_keys=False)
        return

    ## Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    ## Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(dic, f, sort_keys=False)


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
    if name == "none":
        return None
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
