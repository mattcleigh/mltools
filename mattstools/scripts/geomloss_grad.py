import torch as T

from geomloss import SamplesLoss
from mattstools.torch_utils import get_loss_fn


def main():

    ## Define dummy outputs and target tensors
    outputs = T.randn(3, 100)
    targets = T.randn(3, 100)

    ## Define the loss functions to test
    loss_fns = [
        SamplesLoss("energy"),
        SamplesLoss("sinkhorn", p=1, blur=0.01),
        get_loss_fn("engmmd"),
        get_loss_fn("sinkhorn"),
    ]

    for loss_fn in loss_fns:

        with T.no_grad():
            loss = loss_fn(outputs, targets)
            print(loss, T.is_grad_enabled())


if __name__ == "__main__":
    main()
