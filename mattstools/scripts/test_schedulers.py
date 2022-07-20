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

    ## Define the schedulers
    scheds = [
        {
            "name": "onecycle",
            "div_factor": 10,
            "final_div_factor": 10,
            "epochs_per_cycle": 20,
            "pct_start": 0.05,
            "anneal_strategy": "linear",
            "three_phase": True,
        },
        {
            "name": "onecycle",
            "div_factor": 10,
            "final_div_factor": 1,
            "epochs_per_cycle": 20,
            "pct_start": 0.05,
            "anneal_strategy": "linear",
            "three_phase": True,
        },
    ]

    ## Cycle through the different schedulers
    for schd_dict in scheds:

        ## Build the scheduler
        opt = get_optim({"name": "sgd", "lr": lr}, net.parameters())
        schd = get_sched(schd_dict, opt, batches, max_lr=lr, max_epochs=epochs)

        ## Simulate the learning loop
        lrs = []
        opt.step()  ## To prevent the warnings
        for epoch in range(epochs):
            for batch in range(batches):
                lrs.append(schd.get_last_lr()[0])
                schd.step()

        plt.plot(lrs, label=schd_dict["name"])

    plt.legend()
    plt.xlabel("batch passes (100 per epoch)")
    plt.ylabel("learning rate")
    plt.savefig("schedulers.png")


if __name__ == "__main__":
    main()
