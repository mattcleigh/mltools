"""
A collection of networks that all inherit from the MyNetwork base class
"""

from pathlib import Path
from typing import Union

import torch as T
import torch.nn as nn

from utils import sel_device

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
        device: str = "cpu",
    ) -> None:
        """
        kwargs:
            name: The name for the network, used for saving
            save_dir: The save directory for the model
            inpt_dim: The dimension of the input data
            outp_dim: The dimension of the output data
            device: The name of the device on which to load/save and store the network
        """
        super().__init__()

        ## Basic interfacing class attributes
        self.name = name
        self.save_dir = save_dir
        self.full_name = Path(save_dir, name)
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.device = sel_device(device)

        ## A list of all the loss names, all classes need a total loss!
        self.loss_names = ["total"]

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
        - Can be accesed as attributes with prefix = 'preproc_'
        """
        for key, val in stat_dict.items():
            self.register_buffer("preproc_" + key, val.to(self.device))

    def get_losses(self, *_, path: Path, flag: str):
        """This method should be overwritten by any inheriting network
        - It should be used to populate the dictionary of losses
        """
        print("This model has no get_losses method")

    def visualise(self, *_):
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
            T.save(self, full_path)
        else:
            T.save(self.state_dict(), full_path)
