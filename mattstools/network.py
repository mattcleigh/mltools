"""
A collection of networks that all inherit from the MyNetwork base class
"""

import dill
from itertools import count
from pathlib import Path
from typing import Union

import torch as T
import torch.nn as nn
import wandb

from mattstools.torch_utils import sel_device, count_parameters
from mattstools.utils import save_yaml_files


class MyNetBase(nn.Module):
    """A base class which is used to keep consistancy and harmony between the networks defined here
    and the trainer class
    """

    def __init__(
        self,
        *,
        name: str,
        save_dir: str,
        inpt_dim: Union[int, list],
        outp_dim: Union[int, list],
        ctxt_dim: Union[int, list] = 0,
        device: str = "cpu",
        mkdir: bool = True,
        **other_info,
    ) -> None:
        """
        kwargs:
            name: The name for the network, used for saving
            save_dir: The save directory for the model
            inpt_dim: The dimension of the input data
            outp_dim: The dimension of the output data
            ctxt_dim: The dimension of the context data, optional
            device: The name of the device on which to load/save and store the network
            mkdir: If a directory for holding the model should be made
            other_kwargs: These kwargs are saved as attributes on the object
        """
        super().__init__()
        print(f"\nCreating network: {name}")

        ## Basic interfacing class attributes
        self.name = name
        self.save_dir = save_dir
        self.full_name = Path(save_dir, name)
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.device = sel_device(device)

        ## A list of all the loss names, all classes need a total loss!
        self.loss_names = ["total"]

        ## Create the folder to store the network
        if mkdir:
            self.full_name.mkdir(parents=True, exist_ok=True)

        ## Any extra information
        self.other_info = other_info

    def _setup(self) -> None:
        """Finish setting up the model this should be called at the end of the init"""
        self.to(self.device)
        print(self)

    def loss_dict_reset(self) -> dict:
        """Reset the loss dictionary
        - Returns a dictionary with 0 values for each of the loss names
        - Should be called at the beggining of each get_losses call
        """
        return {
            lsnm: T.tensor(0, dtype=T.float32, device=self.device)
            for lsnm in self.loss_names
        }

    def set_preproc(self, stat_dict):
        """Save a dictionary of data processing tensors as buffers on the network
        - Ensures they will be saved/loaded alongside the network
        """
        for key, val in stat_dict.items():
            self.register_buffer(key, val.to(self.device))

    def get_losses(self, _batch: tuple, _batch_idx: int, _epoch_num: int) -> dict:
        """The function called by the trainer class to perform gradient descent
        by defualt the forward pass should have space for the sample and a get_loss
        flag
        - This method can be overwritten if there is a quicker way to get the loss
            - This is the case for normalising flows
        """
        loss_dict = {"total": 0}
        return loss_dict

    def visualise(self, *_args, **__kwargs):
        """This method should be overwritten by any inheriting network
        - It is used to save certain plots using a batch of samples
        """
        print("This model has no visualise method")

    def save(
        self,
        file_name: str = "model",
        as_dict: bool = False,
        cust_path: Union[str, Path] = "",
    ) -> None:
        """Save a version of the model
        - Will place the model in its save_dir/name/ by default
        - Can be saved as either as fixed or as a dictionary

        kwargs:
            name: The output name of the network file
            as_dict: True if the network is to be saved as a torch dict
            cust_path: The path to save the network, if empty uses the save_dir
        """

        ## All dict saved get the dict suffix
        if as_dict:
            file_name += "_dict"

        ## Check that the folder exists
        folder = Path(cust_path or self.full_name)
        folder.mkdir(parents=True, exist_ok=True)

        ## Create the full path of the file
        full_path = Path(folder, file_name)

        ## Use the torch save method
        if as_dict:
            T.save(self.state_dict(), full_path)
        else:
            try:
                T.save(self, full_path)
            except:
                T.save(self, full_path, pickle_module=dill)

    def save_configs(self, data_conf, net_conf, train_conf, do_wandb=True):
        """Save the three config files that were used to build the network,
        supply the data and train the model
        """
        save_yaml_files(
            self.full_name / "config",
            ["data", "net", "train"],
            [data_conf, net_conf, train_conf],
        )

        ## Also save the configs to weights and biases
        if do_wandb and wandb.run is not None:
            wandb.config.update(
                {
                    "data_conf": data_conf,
                    "net_conf": net_conf,
                    "train_conf": train_conf,
                    "num_params": count_parameters(self),
                },
                allow_val_change=True,
            )

    def set_device(self, device):
        """Sets the device attribute and moves all parameters"""
        self.device = sel_device(device)
        self.to(self.device)

    def __repr__(self):
        return super().__repr__() + "\nNum params: " + str(count_parameters(self))

    def on_epoch_start(self, _epoch_num: int):
        """This method is called by the trainer when an epoch begins"""
        return

    def on_epoch_end(self, _epoch_num: int):
        """This method is called by the trainer when an epoch ends"""
        return

    def on_train_start(self, _epoch_num: int):
        """This method is called by the trainer when a training epoch begins"""
        return

    def on_valid_start(self, _epoch_num: int):
        """This method is called by the trainer when a validation epoch begins"""
        return
