from typing import Iterable

import torch as T
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

from mattstools.network import MyNetBase
from mattstools.transformers import FullTransformerVectorEncoder

def contrastive_loss(aug_1, aug_2):
    return 0

def augmentation_collation(batch: Iterable) -> tuple:
    """A custom collation function which augments batches of jets together
    """
    batch = default_collate(batch)

    # ... do all the augmentations
    batch_1 = batch*2
    batch_2 = batch*3

    return batch_1, batch_2


class CLRTrans(MyNetBase):
    def __init__(self, base_kwargs: dict, ftve_kwargs: dict) -> None:
        super().__init__(**base_kwargs)

        ## Load the module itself
        self.ftve = FullTransformerVectorEncoder(
            inpt_dim=self.inpt_dim,
            outp_dim=self.outp_dim,
            **ftvs_kwargs
        )
        self.to(self.device)

    def forward(self, nodes: T.Tensor) -> T.Tensor:
        return self.ftve(nodes)

    def get_losses(self, sample: tuple, _batch_idx: int) -> dict:

        ## Break up the sample into it's two augmented versions
        aug_1, aug_2 = sample

        ## Pass each one through the network to get the embeddings
        aug_1 = self.forward(aug_1)
        aug_2 = self.forward(aug_2)

        ## Update the loss dictionary
        return {"total", contrastive_loss(aug_1, aug_2)}

base_dict = {
  "name": "JetCLR",
  "save_dir": "/home/users/l/leighm/scratch/Saved_Networks/CLR",
  "device": "gpu",
  "inpt_dim": 3,
  "outp_dim": 64,
}

config_dict = {
    "tve_kwargs": {
        "model_dim":128,
        "num_sa_blocks":3,
        "num_ca_blocks":2,
        "mha_kwargs":{
            "num_heads": 4,
            "drp": 0.1
        },
        "trans_ff_kwargs": {
            "hddn_dim": 128,
            "nrm": "layer",
            "drp": 0.1,
        },
    },
    "node_embd_kwargs": {
        "hddn_dim": 128,
        "nrm": "layer",
        "drp": 0.1,
    },
    "outp_embd_kwargs": {
        "hddn_dim": 256,
        "nrm": "layer",
        "drp": 0.1,
    },
}

network = CLRTrans(base_dict, config_dict)

example_inpt = T.rand((10, 100, 3))
data = network.forward(example_inpt)
data = data / T.linalg.norm(data, dim=-1, keepdims=True)
print(data)


