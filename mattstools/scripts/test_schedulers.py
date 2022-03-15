from mattstools.torch_utils import get_sched, get_optim


import torch.nn as nn

import matplotlib.pyplot as plt

def main():

    ## Define the learning bounds
    lr = 1
    epochs = 3
    batches = 100

    ## Define dummy network and an optimiser
    net = nn.Linear(3, 3)
    opt = get_optim({"name": "sgd", "lr": lr}, net.parameters())

    ## Define the schedulers
    schd_nms = [
        "cosann",
        "cosannwr",
        "onecycle",
        "cyclicwithwarmup"
    ]

    ## Cycle through the different schedulers
    for schd_nm in schd_nms:

        ## Build the scheduler
        schd = get_sched({"name": schd_nm}, opt, batches, max_lr=lr, max_epochs=epochs)

        ## Simulate the learning loop
        lrs = []
        opt.step() ## To prevent the warnings
        for epoch in range(epochs):
            for batch in range(batches):
                lrs.append(schd.get_last_lr()[0])
                schd.step()

        plt.plot(lrs, label = schd_nm)

    plt.legend()
    plt.xlabel("batch passes")
    plt.ylabel("learning rate")
    plt.savefig("schedulers.png")

if __name__ == "__main__":
    main()
