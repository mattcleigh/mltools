"""
A collection of plotting scripts for standard uses
"""

from typing import Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_multi_loss(path: Path, loss_hist: dict, xlabel: str = "epoch") -> None:
    """Plot the contents of a loss history with epoch on the x-axis
    args:
        path: Where to save the output images
        loss_hist: A dictionary containing lists of loss histories
            dict[loss_name][dataset][epoch_number]
    kwargs
        xlabel: The label for the shared x-axis in the plots
    """

    ## Create the main figure and subplots
    fig, axes = plt.subplots(
        len(loss_hist), 1, sharex=True, figsize=(4, 4 * len(loss_hist))
    )

    ## Account for the fact that there may be a single loss
    if len(loss_hist) == 1:
        axes = [axes]

    ## Cycle though the different loss types, each on their own axis
    for ax, lnm in zip(axes, loss_hist.keys()):
        ax.set_ylabel(lnm)
        ax.set_xlabel(xlabel)

        ## Plot each dataset's history ontop of each other
        for dset, vals in loss_hist[lnm].items():
            ax.plot(vals, label=dset)

    ## Put the legend only on the top plot and save
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"))
    plt.close(fig)


def plot_multi_hists(
    path: Union[Path, str],
    data_list: list,
    type_labels: list,
    col_labels: list,
    bins: int = 100,
    logy: bool = False,
    hor: bool = True,
    scale: int = 4,
    leg: bool = True,
):
    """Plot multiple histograms given a list of 2D tensors/arrays
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis

    args:
        path: The save location of the plots
        data_list: A list of tensors or numpy arrays
        type_labels: A list of labels for each tensor in data_list
        col_labels: A list of labels for each column/histogram
        bins: The number of bins to use for the histograms
        logy: If we should use the log in the y-axis
        hor: If the multiplot should be horizontal or vertical
        scale: The size in inches for each subplot
        leg: If the legend should be plotted
    """

    ## Make sure we are using a pathlib type variable
    path = Path(path)

    ## Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = len(data_list[0][0])

    ## Create the figure and axes listss
    dims = np.array([n_axis, 1]) if hor else np.array([1, n_axis])
    fig, axes = plt.subplots(*dims[::-1], figsize=tuple(scale * dims))

    ## Cycle through each axis
    for i in range(n_axis):

        ## Save the exact bins for superimposing the different arrays
        b = bins

        ## Cycle through the different data arrays
        for j in range(n_data):

            ## Calculate histogram of the column and remember the bins
            histo, b = np.histogram(data_list[j][:, i], b, density=True)

            ## Plot the histogram as a step graph
            axes[i].step(b, [0] + histo.tolist(), label=type_labels[j])

        ## Set the x_axis label and limits
        axes[i].set_xlabel(col_labels[i])
        axes[i].set_xlim(b[0], b[-1])

        ## Set the y scale to be logarithmic
        if logy:
            axes[i].set_yscale("log")

        ## Set the y axis, only on first if horizontal
        if not hor or i == 0:
            axes[i].set_ylabel("Normalised Entries")

    ## Only do legend on the first axis
    if leg:
        axes[0].legend()

    ## Save the image as a png
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"))
    plt.close(fig)
