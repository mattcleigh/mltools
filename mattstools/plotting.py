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
    data_list: Union[list, np.ndarray],
    type_labels: Union[list, str],
    col_labels: Union[list, str],
    normed: bool = True,
    bins: Union[list, str] = "auto",
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
        bins: The bins to use for each axis, can use numpy's strings
        logy: If we should use the log in the y-axis
        hor: If the multiplot should be horizontal or vertical
        scale: The size in inches for each subplot
        leg: If the legend should be plotted
        incl_zeros: If zero values should be included in the histograms or ignored
    """

    ## Make sure we are using a pathlib type variable
    path = Path(path)

    ## Make the arguments lists for generality
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(type_labels, list):
        type_labels = [type_labels]
    if not isinstance(col_labels, list):
        col_labels = [col_labels]
    if not isinstance(bins, list):
        bins = len(data_list[0][0]) * [bins]

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
        ## If the number of datapoints is less than 10 then we assume interger types
        b = bins[i]
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
    rank_col: str = None,
    cmap: str = "Spectral",
    curved: bool = True,
    curved_extend: float = 0.1,
    groupby_methods: list = None,
    highlight_best: bool = False,
    do_sort: bool = True,
) -> None:
    """
    Create a parallel coordinates plot from pandas dataframe
    args:
        path: Location of output plot
        df: dataframe
        cols: columns to use for axes, final column will be used for colour ranking
    kwargs:
        rank_col: The name of the column to use for ranking, otherwise takes last
        cmap: Colour palette to use for ranking of lines
        curved: Use spline interpolation along lines
        curved_extend: Fraction extension in y axis, adjust to contain curvature
        groupby_methods: List of aggr methods to include for each categorical column
        highlight_best: Highlight the best row with a darker line
        do_sort: Sort the dataframe by rank column, best configs are more visible
    """

    ## Make sure that the rank column is the final column in the list
    if rank_col is not None:
        if rank_col in cols:
            cols.append(cols.pop(cols.index(rank_col)))
        else:
            cols.append(rank_col)
    rank_col = cols[-1]

    ## Sort the dataframe by the rank column
    if do_sort:
        df.sort_values(by=rank_col, ascending=False, inplace=True)

    ## Load the colourmap
    colmap = matplotlib.cm.get_cmap(cmap)

    ## Create a value matrix for the y intercept points on each column for each line
    y_matrix = np.zeros((len(cols), len(df)))
    x_values = np.arange(len(cols))
    ax_info = {}  ## Dict which will contain tick labels and values for each col

    ## Cycle through each column
    for i, col in enumerate(cols):

        ## Pull the column data from the dataframe
        col_data = df[col]

        ## For continuous data (more than 10 unique values)
        if (col_data.dtype == float) & (len(np.unique(col_data)) > 10):

            ## Scale the range of data to [0,1] and save to matrix
            y_min = np.min(col_data)
            y_max = np.max(col_data)
            y_range = y_max - y_min
            y_matrix[i] = (col_data - y_min) / y_range

            ## Create the ticks and tick labels for the axis
            nticks = 5  ## Good number for most cases
            tick_labels = np.linspace(y_min, y_max, nticks, endpoint=True)
            tick_labels = [f"{s:.2f}" for s in tick_labels]
            tick_values = np.linspace(0, 1, nticks, endpoint=True)
            ax_info[col] = [tick_labels, tick_values]

        ## For categorical data (less than 10 unique values)
        else:

            ## Set the type for the data to categorical to pull out stats using pandas
            col_data = col_data.astype("category")
            cats = col_data.cat.categories
            cat_vals = col_data.cat.codes

            ## Scale to the range [0,1] (special case for data with only one cat)
            if len(cats) == 1:
                y_matrix[i] = 0.5
            else:
                y_matrix[i] = cat_vals / cat_vals.max()

            ## The tick labels include average performance using groupby
            if groupby_methods is not None and col != rank_col:
                groups = (
                    df[[col, rank_col]].groupby([col]).agg(groupby_methods)[rank_col]
                )

                ## Create the tick labels by using all groupy results
                tick_labels = [
                    str(cat)
                    + "".join(
                        [
                            f"\n{meth}={groups[meth].loc[cat]:.3f}"
                            for meth in groupby_methods
                        ]
                    )
                    for cat in list(cats)
                ]

            ## Or they simply use the cat names
            else:
                tick_labels = cats

            ## Create the tick locations and save in dict
            tick_values = np.unique(y_matrix[i])
            ax_info[col] = [tick_labels, tick_values]

    ## Get the index of the best row
    best_idx = np.argmin(y_matrix[-1]) if highlight_best else -1

    ## Create the plot
    fig, axes = plt.subplots(
        1, len(cols) - 1, sharey=False, figsize=(3 * len(cols) + 3, 5)
    )

    ## Amount by which to extend the y axis ranges above the data range
    y_ax_ext = curved_extend if curved else 0.05

    ## Cycle through each line (singe row in the original dataframe)
    for lne in range(len(df)):

        ## Calculate spline function to use across all axes
        if curved:
            spline_fn = make_interp_spline(
                x_values, y_matrix[:, lne], k=3, bc_type="clamped"
            )

        ## Keyword arguments for drawing the line
        lne_kwargs = {
            "color": colmap(y_matrix[-1, lne]),
            "alpha": 1 if lne == best_idx else 0.3,
            "linewidth": 4 if lne == best_idx else None,
        }

        ## Cycle through each axis (bridges one column to the next)
        for i, ax in enumerate(axes):

            ## For splines
            if curved:

                ## Plot the spline using a more dense x space spanning the axis window
                x_space = np.linspace(i, i + 1, 20)
                ax.plot(x_space, spline_fn(x_space), **lne_kwargs)

            ## For simple line connectors
            else:
                ax.plot(
                    x_values[[i, i + 1]],
                    y_matrix[[i, i + 1], lne],
                    **lne_kwargs,
                )

            ## Set the axis limits, y included extensions, x is limited to window
            ax.set_ylim(0 - y_ax_ext, 1 + y_ax_ext)
            ax.set_xlim(i, i + 1)

    ## For setting the axis ticklabels
    for dim, (ax, col) in enumerate(zip(axes, cols)):

        ## Reduce the x axis ticks to the start of the plot for column names
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.set_xticklabels([cols[dim]])

        ## The y axis ticks were calculated and saved in the info dict
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])

    ## Create the colour bar on the far right side of the plot
    norm = matplotlib.colors.Normalize(0, 1)  ## Map data into the colour range [0, 1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  ## Required for colourbar
    cbar = plt.colorbar(
        sm,
        pad=0,
        ticks=ax_info[rank_col][1],  ## Uses ranking attribute
        extend="both",  ## Extending to match the y extension passed 0 and 1
        extendrect=True,
        extendfrac=y_ax_ext,
    )

    ## The colour bar also needs tick labels, x labels and y limits extended
    cbar.ax.set_yticklabels(ax_info[rank_col][0])
    cbar.ax.set_xlabel(rank_col)
    if curved:
        cbar.ax.set_ylim(0 - curved_extend, 1 + curved_extend)

    ## Change the plot layout and save
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, left=0.05, right=0.95)
    plt.savefig(Path(path).with_suffix(".png"))
