"""
A collection of plotting scripts for standard uses
"""

from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.interpolate import make_interp_spline


def plot_multi_loss(
    path: Path,
    loss_hist: dict,
    xvals: list = None,
    xlabel: str = "epoch",
    logx: bool = False,
) -> None:
    """Plot the contents of a loss history with epoch on the x-axis
    args:
        path: Where to save the output images
        loss_hist: A dictionary containing lists of loss histories
            dict[loss_name][dataset][epoch_number]
    kwargs
        xvals: A list of values for the x-axis, if None it uses arrange
        xlabel: The label for the shared x-axis in the plots
        logx: Using a log scale on the x-axis
    """

    ## Get the x-axis values using the length of the total loss in the trainset
    ## This should always be present
    if xvals is None:
        xvals = np.arange(1, len(loss_hist["total"]["train"]) + 1)

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

        if logx:
            ax.set_xscale("log")

        ## Plot each dataset's history ontop of each other
        for dset, vals in loss_hist[lnm].items():

            ## Skip empty loss dictionaries (sometimes we dont have valid loss)
            if not vals:
                continue

            ax.plot(xvals, vals, label=dset)

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
    normed: bool = True,
    bins: str = "auto",
    logy: bool = False,
    ylim: list = None,
    hor: bool = True,
    scale: int = 4,
    leg: bool = True,
    incl_zeros: bool = True,
):
    """Plot multiple histograms given a list of 2D tensors/arrays
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis

    args:
        path: The save location of the plots
        data_list: A list of tensors or numpy arrays
        type_labels: A list of labels for each tensor in data_list
        col_labels: A list of labels for each column/histogram
        normed: If the histograms are to be a density plot
        bins: The number of bins to use for the histograms, can use numpy's strings
        logy: If we should use the log in the y-axis
        hor: If the multiplot should be horizontal or vertical
        scale: The size in inches for each subplot
        leg: If the legend should be plotted
        incl_zeros: If zero values should be included in the histograms or ignored
    """

    ## Make sure we are using a pathlib type variable
    path = Path(path)

    ## Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = len(data_list[0][0])

    ## Create the figure and axes listss
    dims = np.array([n_axis, 1]) if hor else np.array([1, n_axis])
    fig, axes = plt.subplots(*dims[::-1], figsize=tuple(scale * dims))

    ## To support a single plot
    if n_axis == 1:
        axes = [axes]

    ## Replace the zeros
    if not incl_zeros:
        for d in data_list:
            d[d == 0] = np.nan

    ## Cycle through each axis
    for i in range(n_axis):

        ## Reduce bins based on number of unique datapoints
        b = bins
        if isinstance(bins, str):
            n_unique = len(np.unique(data_list[0][:, i]))
            if n_unique < 10:
                b = np.linspace(
                    data_list[0][:, i].min() - 0.5,
                    data_list[0][:, i].max() + 0.5,
                    n_unique + 1,
                )

        ## Cycle through the different data arrays
        for j in range(n_data):

            ## Calculate histogram of the column and remember the bins
            histo, b = np.histogram(data_list[j][:, i], b, density=normed)

            ## Plot the histogram as a step graph
            axes[i].step(b, [0] + histo.tolist(), label=type_labels[j])

        ## Set the x_axis label and limits
        axes[i].set_xlabel(col_labels[i])
        axes[i].set_xlim(b[0], b[-1])

        if ylim is not None:
            axes[i].set_ylim(*ylim)

        ## Set the y scale to be logarithmic
        if logy:
            axes[i].set_yscale("log")

        ## Set the y axis, only on first if horizontal
        if not hor or i == 0:
            if normed:
                axes[i].set_ylabel("Normalised Entries")
            else:
                axes[i].set_ylabel("Entries")

    ## Only do legend on the first axis
    if leg:
        axes[0].legend()

    ## Save the image as a png
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"))
    plt.close(fig)


def parallel_plot(
    path: str,
    df: pd.DataFrame,
    cols: list,
    rank_attr: str,
    cmap: str="Spectral",
    curved: bool=True,
    curvedextend=0.1,

):
    """
    Create a parallel coordinates plot from pandas dataframe
    args:
        path: Location of output plot
        df: dataframe
        cols: columns to use for axes
        rank_attr: attribute to use for colour ranking
    kwargs:
        cmap: Colour palette to use for ranking of lines
        curved: Spline interpolation along lines
        curvedextend: Fraction extension in y axis, adjust to contain curvature
    """

    ## Load the colourmap
    colmap = matplotlib.cm.get_cmap(cmap)

    ## Add the column to use for the ranking
    cols = cols + [rank_attr]

    ## Create the plot
    fig, axes = plt.subplots(
        1, len(cols) - 1, sharey=False, figsize=(3 * len(cols) + 3, 5)
    )
    valmat = np.ndarray(shape=(len(cols), len(df)))
    x = np.arange(0, len(cols), 1)
    ax_info = {}
    for i, col in enumerate(cols):
        vals = df[col]
        if (vals.dtype == float) & (len(np.unique(vals)) > 10):
            minval = np.min(vals)
            maxval = np.max(vals)
            rangeval = maxval - minval
            vals = np.true_divide(vals - minval, maxval - minval)
            nticks = 5
            tick_labels = [
                round(minval + i * (rangeval / nticks), 4) for i in range(nticks + 1)
            ]
            ticks = [0 + i * (1.0 / nticks) for i in range(nticks + 1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels, ticks]
        else:
            vals = vals.astype("category")
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = 0
            maxval = len(cats) - 1
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval - minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels, ticks]
            valmat[i] = c_vals

    extendfrac = curvedextend if curved else 0.05
    for i, ax in enumerate(axes):
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x) * 20)
                a_BSpline = make_interp_spline(
                    x, valmat[:, idx], k=3, bc_type="clamped"
                )
                y_new = a_BSpline(x_new)
                ax.plot(x_new, y_new, color=colmap(valmat[-1, idx]), alpha=0.3)
            else:
                ax.plot(x, valmat[:, idx], color=colmap(valmat[-1, idx]), alpha=0.3)
        ax.set_ylim(0 - extendfrac, 1 + extendfrac)
        ax.set_xlim(i, i + 1)

    for dim, (ax, col) in enumerate(zip(axes, cols)):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))

        ## Formatting the tick labels to make them readable
        tick_labels = []
        for a in ax_info[col][0]:
            if isinstance(a, float):
                tick_labels.append("{:.5}".format(a))
            else:
                tick_labels.append(a)

        ax.set_yticklabels(tick_labels)
        ax.set_xticklabels([cols[dim]])

    plt.subplots_adjust(wspace=0)
    norm = matplotlib.colors.Normalize(0, 1)  # *axes[-1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        pad=0,
        ticks=ax_info[rank_attr][1],
        extend="both",
        extendrect=True,
        extendfrac=extendfrac,
    )
    if curved:
        cbar.ax.set_ylim(0 - curvedextend, 1 + curvedextend)

    ## Change the plot labels to be the configuration, not value
    labels = [str(row) for row in df[cols[:-3]].values.tolist()]
    if len(labels) > 10:
        labels = ax_info[rank_attr][0]
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_xlabel(rank_attr)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, left=0.05, right=0.85)
    plt.savefig(Path(path).with_suffix(".png"))
