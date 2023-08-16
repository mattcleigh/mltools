"""General mix of utility functions not related to numpy or pytorch."""

import argparse
import json
import math
import operator
from functools import reduce
from pathlib import Path
from typing import Any, Mapping, Union

import yaml
from dotmap import DotMap
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def standard_job_array(
    job_name: str,
    work_dir: str,
    image_path: str,
    command: str,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
    time_hrs: int,
    mem_gb: int,
    opt_dict: Mapping,
    use_dashes: bool = True,
    extra_slurm: str = "",
) -> None:
    """Create a standard gridjob array using the expected settings for the UNIGE HPC."""
    # Calculate the total number of jobs to perform
    n_jobs = 1
    for key, vals in opt_dict.items():
        if not isinstance(vals, list):
            vals = [vals]
            opt_dict[key] = vals
        n_jobs *= len(vals)
    print(f"Generating gridsearch with {n_jobs} subjobs")

    # Creating the slurm submision file
    f = open(f"{job_name}.sh", "w", newline="\n", encoding="utf-8")
    f.write("#!/bin/sh\n\n")
    f.write(f"#SBATCH --cpus-per-task={n_cpus}\n")
    f.write(f"#SBATCH --mem={mem_gb}GB\n")
    f.write(f"#SBATCH --time={time_hrs}:00:00\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write(f"#SBATCH --output={log_dir}/%A_%a.out\n")
    if n_gpus:
        f.write(f"#SBATCH --gpus={n_gpus}\n")
        f.write("#SBATCH --partition=shared-gpu,private-dpnc-gpu\n")
    else:
        f.write("#SBATCH --partition=shared-cpu,private-dpnc-cpu\n")

    # Include the extra slurm here
    f.write(extra_slurm + "\n")

    # The job array setup using the number of jobs
    f.write(f"\n#SBATCH -a 0-{n_jobs-1}\n\n")

    # Creating the bash lists of the job arguments
    simple_keys = [str(k).replace(".", "") for k in opt_dict]
    for i, (opt, vals) in enumerate(opt_dict.items()):
        f.write(f"{simple_keys[i]}=(")
        for v in vals:
            f.write(" " + str(v))
        f.write(" )\n")
    f.write("\n")

    # The command line arguments
    f.write('export XDG_RUNTIME_DIR=""\n')

    # Creating the base singularity execution script
    f.write(f"cd {work_dir}\n")
    f.write("srun apptainer exec --nv -B /srv,/home \\\n")
    f.write(f"   {image_path} \\\n")
    f.write(f"   {command} \\\n")

    # Now include the job array options using the bash lists
    run_tot = 1
    dashdash = "--" if use_dashes else ""
    for i, (opt, vals) in enumerate(opt_dict.items()):
        f.write(f"       {dashdash}{opt} ${{{simple_keys[i]}")
        f.write(f"[`expr ${{SLURM_ARRAY_TASK_ID}} / {run_tot} % {len(vals)}`]")
        f.write("} \\\n")
        run_tot *= len(vals)
    f.close()


def str2bool(mystring: str) -> bool:
    """Convert a string object into a boolean."""
    if isinstance(mystring, bool):
        return mystring
    if mystring.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if mystring.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def merge_dict(source: dict, update: dict) -> dict:
    """Merge two deep dictionaries recursively.

    Slow with deep dictionaries.

    Args
    ----
    source:
        The source dict, will be copied (not modified)
    update:
        Will be used to overwrite and append values to the source
    """
    # Make a copy of the source dictionary
    merged = source.copy()

    # Cycle through all of the keys in the update
    for key in update:
        # If the key not in the source then add move on
        if key not in merged:
            merged[key] = update[key]
            continue

        # Check type of variable
        dict_in_upt = isinstance(update[key], dict)
        dict_in_src = isinstance(source[key], dict)

        # If neither are a dict, then simply replace the leaf variable
        if not dict_in_upt and not dict_in_src:
            merged[key] = update[key]

        # If both are dicts, then implement recursion
        elif dict_in_upt and dict_in_src:
            merged[key] = merge_dict(source[key], update[key])

        # Otherwise one is a dict and the other is a leaf, so fail!
        else:
            raise ValueError(
                f"Trying to merge dicts but {key} is a leaf node in one not other"
            )

    return merged


def print_dict(dic: dict, indent: int = 1) -> None:
    """Recursively print a dictionary using json."""
    print(json.dumps(dic, indent=indent))


def get_from_dict(data_dict: dict, key_list: list, default=None) -> Any:
    """Return a value from a nested dictionary using list of keys."""
    try:
        return reduce(operator.getitem, key_list, data_dict)
    except KeyError:
        return default


def set_in_dict(data_dict: dict, key_list: list, value: Any):
    """Set a value in a nested dictionary using a list of keys."""
    get_from_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def key_prefix(pref: str, dic: dict) -> dict:
    """Add a prefix to each key in a dictionary."""
    return {f"{pref}{key}": val for key, val in dic.items()}


def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Change the key used in a dictionary inplace only if it exists."""

    # If the original key is not present, nothing changes
    if old_key not in dic:
        return

    # Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    # Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]


def remove_keys_starting_with(dic: dict, match: str) -> dict:
    """Remove all keys from the dictionary if they start with.

    - Returns a copy of the dictionary
    """
    return {key: val for key, val in dic.items() if key[: len(match)] != match}


def signed_angle_diff(angle1: Any, angle2: Any) -> Any:
    """Calculate diff between two angles reduced to the interval of [-pi, pi]"""
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi


def load_yaml_files(files: Union[list, tuple, str]) -> tuple:
    """Load a list of files using yaml and returns a tuple of dictionaries."""

    # If the input is not a list then it returns a dict
    if isinstance(files, (str, Path)):
        with open(files, encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.Loader)

    opened = []

    # Load each file using yaml
    for fnm in files:
        with open(fnm, encoding="utf-8") as f:
            opened.append(yaml.load(f, Loader=yaml.Loader))

    return tuple(opened)


def save_yaml_files(
    path: str, file_names: Union[str, list, tuple], dicts: Union[dict, list, tuple]
) -> None:
    """Save a collection of yaml files in a folder.

    - Makes the folder if it does not exist
    """

    # If the input is not a list then one file is saved
    if isinstance(file_names, (str, Path)):
        with open(f"{path}/{file_names}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(
                dicts.toDict() if isinstance(dicts, DotMap) else dicts,
                f,
                sort_keys=False,
            )
        return

    # Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    # Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(
                dic.toDict() if isinstance(dic, DotMap) else dic, f, sort_keys=False
            )


def get_scaler(name: str):
    """Return a sklearn scaler object given a name."""
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
