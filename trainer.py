"""
Base class for training network
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mattsutils.utils import RunningAverage
from mattsutils.plotting import plot_multi_loss
from mattsutils.torch_utils import get_optim, get_sched, move_dev


class Trainer:
    """A class to oversee the training of a network which can handle its own losses"""

    def __init__(
        self,
        network,
        train_set: Dataset,
        valid_set: Dataset,
        b_size: int = 32,
        patience: int = 100,
        max_epochs: int = 100,
        grad_clip: float = 0.0,
        n_workers: int = 2,
        optim_dict: dict = None,
        sched_dict: dict = None,
        vis_every: int = 10,
        chkp_every: int = 10,
    ) -> None:
        """
        args:
            network:     Network with a get_losses method
            train_set:   Dataset on which to perform batched gradient descent
            valid_set:   Dataset for validation loss and early stopping conditions
        kwargs:
            b_size:      Batch size to use in the minibatch gradient descent
            patience:    Early stopping patience calculated using the validation set
            max_epochs:  Maximum number of epochs to train for
            grad_clip:   Clip value for the norm of the gradients (0 will not clip)
            n_workers:   Number of parallel threads which prepare each batch
            optim_dict:  A dict used to select and configure the optimiser
            sched_dict:  A dict used to select and configure the scheduler
            vis_every:   Run the network's visualisation function every X epochs
            chkp_every:  Save a checkpoint of the net/opt/schd/loss every X epochs
        """
        print("\nInitialising the trainer")

        ## Default dict arguments
        optim_dict = optim_dict or {"name": "adam", "lr": 1e-4}
        sched_dict = sched_dict or {"name": "none"}

        ## Save the network
        self.network = network

        ## Report on the number of files/samples used
        print(f"train set: {len(train_set):7} samples")
        if valid_set:
            print(f"valid set: {len(valid_set):7} samples")
        else:
            print("No validation set added")

        ## Create the common dataloader arguments
        loader_kwargs = {
            "batch_size": b_size,
            "num_workers": n_workers,
            "drop_last": True,
            "pin_memory": True,
        }

        ## Add a custom collate function to the kwargs if present
        if hasattr(train_set, "col_fn"):
            loader_kwargs["collate_fn"] = train_set.col_fn

        ## Initialise the loaders
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        self.valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

        ## Create a history of train and validation losses for early stopping
        self.loss_hist = {
            lsnm: {set: [] for set in ["train", "valid"]}
            for lsnm in self.network.loss_names
        }

        ## A running average tracker for each loss during an epoch
        self.run_loss = {lsnm: RunningAverage() for lsnm in self.network.loss_names}

        ## Gradient clipping settings and saving settings
        self.grad_clip = grad_clip
        self.vis_every = vis_every
        self.chkp_every = chkp_every

        ## Load the optimiser and scheduler
        self.optimiser = get_optim(optim_dict, self.network.parameters())
        self.scheduler = get_sched(
            sched_dict,
            self.optimiser,
            len(self.train_loader),
            optim_dict["lr"],
            max_epochs,
        )

        ## Variables to keep track of stopping conditions
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_epochs = 0
        self.bad_epochs = 0
        self.best_epoch = 0

    def run_training_loop(self) -> None:
        """The main loop which cycles epochs of train and test
        - After each epochs it calls the save function and checks for early stopping
        """
        print("\nStarting the training process")

        for epc in np.arange(self.num_epochs, self.max_epochs):
            print(f"\nEpoch: {epc}")

            ## Run the test/train cycle, update stats, and save
            self.epoch(is_train=True)
            self.epoch(is_train=False)
            self.count_epochs()
            self.save_checkpoint()

            ## Check if we have exceeded the patience
            if self.bad_epochs > self.patience:
                print("Patience Exceeded: Stopping training!")
                return 0

        ## If we have reached the maximum number of epochs
        print("Maximum number of epochs exceeded")
        return 0

    def epoch(self, is_train: bool = False) -> None:
        """Perform a single epoch on either the train loader or the validation loader
        - Will update average loss during epoch
        - Will add average loss to loss history at end of epoch

        kwargs:
            is_train: Effects gradient tracking, network state, and data loader
        """

        ## Select the correct mode for the epoch
        if is_train:
            mode = "train"
            self.network.train()
            loader = self.train_loader
            T.set_grad_enabled(True)
        else:
            mode = "valid"
            self.network.eval()
            loader = self.valid_loader
            T.set_grad_enabled(False)

        ## Cycle through the batches provided by the selected loader
        for sample in tqdm(loader, desc=mode, ncols=80):

            ## Move the sample to the network device
            sample = move_dev(sample, self.network.device)

            ## Pass through the network and get the loss dictionary
            losses = self.network.get_losses(sample)

            ## For training epochs we perform gradient descent
            if is_train:

                ## Zero and calculate gradients using total loss (from dict)
                self.optimiser.zero_grad(set_to_none=False)
                losses["total"].backward()

                ## Apply gradient clipping
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)

                ## Step the optimiser
                self.optimiser.step()

                ## Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

            ## Update the each of the running losses using the dictionary
            for lnm, running in self.run_loss.items():
                running.update(losses[lnm].item())

        ## Use the running losses to update the total history, then reset
        for lnm, running in self.run_loss.items():
            self.loss_hist[lnm][mode].append(running.avg)
            running.reset()

    def count_epochs(self) -> None:
        """Update attributes counting number of bad and total epochs"""
        self.num_epochs = len(self.loss_hist["total"]["train"])
        self.best_epoch = np.argmin(self.loss_hist["total"]["valid"]) + 1
        self.bad_epochs = self.num_epochs - self.best_epoch

    def save_checkpoint(self) -> None:
        """Add folders to the network's save directory containing
        - losses -> Loss history as json and plotted as pngs
        - checkpoints -> Checkpoints of network/optimiser/scheduler/loss states
        - visual -> Output of network's visualisation method on first valid batch
        """

        print("Saving...")

        ## Save best network model and dict (easier to reload) in main directory
        if self.bad_epochs == 0:
            self.network.save("best")
            self.network.save("best", as_dict=True)

        ## Create loss folder
        loss_folder = Path(self.network.full_name, "losses")
        loss_folder.mkdir(parents=True, exist_ok=True)

        ## Save a plot and a json of the loss history
        loss_file_name = Path(loss_folder, "losses.json")
        plot_multi_loss(loss_file_name, self.loss_hist)
        with open(loss_file_name, "w", encoding="utf-8") as l_file:
            json.dump(self.loss_hist, l_file, indent=2)

        ## For checkpointing
        if self.chkp_every > 0 and self.num_epochs % self.chkp_every == 0:

            ## Create checkpoint folder
            chckpnt_folder = Path(self.network.full_name, "checkpoints")
            chckpnt_folder.mkdir(parents=True, exist_ok=True)

            ## Save a checkpoint of the network/scheduler/optimiser (for reloading)
            checkpoint = {
                "network": self.network.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "losses": self.loss_hist,
            }
            if self.scheduler is not None:
                checkpoint["scheduler"] = self.scheduler.state_dict()
            T.save(checkpoint, Path(chckpnt_folder, f"checkpoint_{self.num_epochs}"))

        ## For visualisation
        if self.vis_every > 0 and self.num_epochs % self.vis_every == 0:

            ## Set evaluation mode
            self.network.eval()
            T.set_grad_enabled(False)

            ## Create the vis folder
            vis_folder = Path(self.network.full_name, "visual")
            vis_folder.mkdir(parents=True, exist_ok=True)

            ## Use the first batch of the valid date as the sample
            sample = next(iter(self.valid_loader))
            self.network.visualise(sample, path=vis_folder, flag=str(self.num_epochs))

    def load_checkpoint(self, flag="latest") -> None:
        """Loads the latest instance of a saved network to continue training"""

        print("Loading checkpoint...")

        ## Find the latest checkpoint in the folder (this could be written better)
        checkpoint_file = "NoFilesFound"
        if flag == "latest":
            for i in range(self.max_epochs):
                test_file = Path(
                    self.network.full_name, "checkpoints", f"checkpoint_{i}"
                )
                if test_file.is_file():
                    checkpoint_file = test_file
        else:
            checkpoint_file = Path(
                self.network.full_name, "checkpoints", f"checkpoint_{flag}"
            )

        ## Load the and unpack checkpoint object
        checkpoint = T.load(checkpoint_file)
        self.network.load_state_dict(checkpoint["network"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.loss_hist = checkpoint["losses"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            checkpoint["scheduler"] = self.scheduler.state_dict()

        ## Update the epoch count
        self.count_epochs()
