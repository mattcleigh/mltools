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
from matplotlib.colors import LogNorm

from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mattstools.utils import mid_points, undo_mid

## Some defaults for my plots to make them look nicer
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.edgecolor"] = "1"
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["legend.fontsize"] = 11


def gaussian(x_data, mu=0, sig=1):
    """Return the value of the gaussian distribution"""
    return (
        1
        / np.sqrt(2 * np.pi * sig**2)
        * np.exp(-((x_data - mu) ** 2) / (2 * sig**2))
    )


def plot_corr_heatmaps(
    path: Path,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bins: list,
    xlabel: str,
    ylabel: str,
    weights: np.ndarray = None,
    do_log: bool = True,
    cmap: str = "coolwarm",
    incl_line: bool = True,
    incl_cbar: bool = True,
    title: str = "",
    figsize=(6, 5),
    do_pearson=False,
    do_pdf: bool = False,
) -> None:
    """
    Plot and save a 2D heatmap, usually for correlation plots

    args:
        path: Location of the output file
        x_vals: The values to put along the x-axis, usually truth
        y_vals: The values to put along the y-axis, usually reco
        bins: The bins to use, must be [xbins, ybins]
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    kwargs:
        weights: The weight value for each x, y pair
        do_log: If the z axis should be the logarithm
        cmap: The name of the cmap to use for z values
        incl_line: If a y=x line should be included to show ideal correlation
        incl_cbar: Add the colour bar to the axis
        figsize: The size of the output figure
        title: Title for the plot
        do_pearson: Add the pearson correlation coeficient to the plot
        do_pdf: If the output should also contain a pdf version
    """

    ## Create the histogram
    if len(bins) != 2:
        bins = [bins, bins]
    hist, xedges, yedges = np.histogram2d(x_vals, y_vals, weights=weights, bins=bins)

    ## Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    imshow = ax.imshow(
        np.log(hist.T) if do_log else hist.T,
        origin="lower",
        cmap=cmap,
        extent=[min(xedges), max(xedges), min(yedges), max(yedges)],
    )

    ## Add line
    if incl_line:
        ax.plot([min(xedges), max(xedges)], [min(yedges), max(yedges)], "k--")

    ## Add colourbar
    if incl_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(imshow, cax=cax, orientation="vertical")

    ## Axis labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title != "":
        ax.set_title(title)

    ## Correlation coeficient
    if do_pearson:
        ax.text(
            0.05,
            0.92,
            f"r = {pearsonr(x_vals, y_vals)[0]:.3f}",
            transform=ax.transAxes,
            fontsize="large",
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    ## Save the image
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    if do_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


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
    normed: bool = False,
    bins: Union[list, str] = "auto",
    logy: bool = False,
    ylim: list = None,
    rat_ylim=(0, 2),
    rat_label=None,
    scale: int = 5,
    leg: bool = True,
    incl_zeros: bool = True,
    already_hists: bool = False,
    hist_fills: list = None,
    hist_colours: list = None,
    hist_kwargs: dict = None,
    hist_scale: float = 1,
    incl_overflow: bool = False,
    incl_underflow: bool = True,
    do_step: bool = True,
    do_ratio_to_first: bool = False,
    as_pdf: bool = False,
):
    """Plot multiple histograms given a list of 2D tensors/arrays
    - Performs the histogramming here
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
        ylim: The y limits for all plots
        rat_ylim: The y limits of the ratio plots
        rat_label: The label for the ratio plot
        scale: The size in inches for each subplot
        leg: If the legend should be plotted
        incl_zeros: If zero values should be included in the histograms or ignored
        already_hists: If the data is already histogrammed and doesnt need to be binned
        hist_fills: Bool for each histogram in data_list, if it should be filled
        hist_colours: Color for each histogram in data_list
        hist_kwargs: Additional keyword arguments for the line for each histogram
        hist_scale: Amount to scale all histograms
        incl_overflow: Have the final bin include the overflow
        incl_underflow: Have the first bin include the underflow
        do_step: If the data should be represented as a step plot
        do_ratio_to_first: Include a ratio plot to the first histogram in the list
        as_pdf: Also save an additional image in pdf format
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
    if not isinstance(hist_colours, list):
        hist_colours = len(data_list) * [hist_colours]

    ## Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = len(data_list[0][0])

    ## Make sure the there are not too many subplots
    if n_axis > 20:
        raise RuntimeError("You are asking to create more than 20 subplots!")

    ## Create the figure and axes listss
    dims = np.array([n_axis, 1])
    size = np.array([n_axis, 1.0])
    if do_ratio_to_first:
        dims *= np.array([1, 2])
        size *= np.array([1, 1.2])
    fig, axes = plt.subplots(
        *dims[::-1],
        figsize=tuple(scale * size),
        gridspec_kw={"height_ratios": [3, 1] if do_ratio_to_first else {1}},
    )
    if n_axis == 1 and not do_ratio_to_first:
        axes = np.array([axes])
    axes = axes.reshape(dims)

    ## Replace the zeros
    if not incl_zeros:
        for d in data_list:
            d[d == 0] = np.nan

    ## Cycle through each axis
    for i in range(n_axis):
        b = bins[i]

        ## Reduce bins based on number of unique datapoints
        ## If the number of datapoints is less than 10 then we assume interger types
        if isinstance(b, str) and not already_hists:
            unq = np.unique(data_list[0][:, i])
            n_unique = len(unq)
            if 1 < n_unique < 10:
                b = (unq[1:] + unq[:-1]) / 2  ## Use midpoints
                b = np.append(b, unq.max() + unq.max() - b[-1])  ## Add final bin
                b = np.insert(b, 0, unq.min() + unq.min() - b[0])  ## Add initial bin

        ## Cycle through the different data arrays
        for j in range(n_data):

            ## Read the binned data from the array
            if already_hists:
                histo = data_list[j][:, i]

            ## Calculate histogram of the column and remember the bins
            else:

                ## Get the bins for the histogram based on the first plot
                if j == 0:
                    b = np.histogram_bin_edges(data_list[j][:, i], bins=b)

                ## Apply overflow and underflow (make a copy)
                data = np.copy(data_list[j][:, i])
                if incl_overflow:
                    data = np.minimum(data, b[-1])
                if incl_underflow:
                    data = np.maximum(data, b[0])

                ## Calculate the histogram
                histo, _ = np.histogram(data, b, density=normed)

            ## Apply the scaling factor
            histo = histo * hist_scale

            ## Save the first histogram for the ratio plots
            if j == 0:
                denom_hist = histo

            ## Get the additional keywork arguments
            if hist_kwargs is not None:
                kwargs = {key: val[j] for key, val in hist_kwargs.items()}
            else:
                kwargs = {}

            ## Plot the fill
            ydata = histo.tolist()
            ydata = [ydata[0]] + ydata
            if hist_fills is not None and hist_fills[j]:
                axes[i, 0].fill_between(
                    b,
                    ydata,
                    label=type_labels[j],
                    step="pre" if do_step else None,
                    alpha=0.4,
                    color=hist_colours[j],
                )

            ## Plot the histogram as a step graph
            elif do_step:
                axes[i, 0].step(
                    b, ydata, label=type_labels[j], color=hist_colours[j], **kwargs
                )

            else:
                axes[i, 0].plot(
                    b, ydata, label=type_labels[j], color=hist_colours[j], **kwargs
                )

            ## Plot the ratio plot
            if do_ratio_to_first:
                ydata = (histo / denom_hist).tolist()
                ydata = [ydata[0]] + ydata
                axes[i, 1].step(
                    b,
                    ydata,
                    color=hist_colours[j],
                    **kwargs,
                )

        ## Set the x_axis label
        if do_ratio_to_first:
            axes[i, 0].set_xticklabels([])
            axes[i, 1].set_xlabel(col_labels[i])
        else:
            axes[i, 0].set_xlabel(col_labels[i])

        ## Set the limits
        axes[i, 0].set_xlim(b[0], b[-1])
        if ylim is not None:
            axes[i, 0].set_ylim(*ylim)

        if do_ratio_to_first:
            axes[i, 1].set_xlim(b[0], b[-1])
            axes[i, 1].set_ylim(rat_ylim)

        ## Set the y scale to be logarithmic
        if logy:
            axes[i, 0].set_yscale("log")

        ## Set the y axis
        if normed:
            axes[i, 0].set_ylabel("Normalised Entries")
        elif hist_scale != 1:
            axes[i, 0].set_ylabel("a.u.")
        else:
            axes[i, 0].set_ylabel("Entries")
        if do_ratio_to_first:
            if rat_label is not None:
                axes[i, 1].set_ylabel(rat_label)
            else:
                axes[i, 1].set_ylabel(f"Ratio to {type_labels[0]}")

    ## Only do legend on the first axis
    if leg:
        axes[0, 0].legend()

    ## Save the image as a png
    fig.tight_layout()

    ## For ratio plots minimise the h_space
    if do_ratio_to_first:
        fig.subplots_adjust(hspace=0.08)

    fig.savefig(Path(path).with_suffix(".png"))
    if as_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_and_save_hists(
    path: str,
    hist_list: list,
    labels: list,
    ax_labels: list,
    bins: np.ndarray,
    do_csv: bool = False,
    stack: bool = False,
    is_mid: bool = False,
) -> None:
    """Plot a list of hitograms on the same axis and save the results to a csv file
    args:
        path: The path to the output file, will get png and csv suffix
        hist_list: A list of histograms to plot
        labels: List of labels for each histogram
        ax_labels: Name of the x and y axis
        bins: Binning used to create the histograms
        do_csv: If the histograms should also be saved as csv files
        stack: If the histograms are stacked or overlayed
        is_mid: If the bins provided are already the midpoints
    """

    ## Make the arguments lists for generality
    if not isinstance(hist_list, list):
        hist_list = [hist_list]

    ## Get the midpoints of the bins
    mid_bins = bins if is_mid else mid_points(bins)
    bins = undo_mid(mid_bins) if is_mid else bins

    ## Save the histograms to text
    if do_csv:
        df = pd.DataFrame(
            np.vstack([mid_bins] + hist_list).T, columns=["bins"] + labels
        )
        df.to_csv(path.with_suffix(".csv"), index=False)

    ## Create the plot of the histograms
    fig, ax = plt.subplots()
    base = np.zeros_like(hist_list[0])
    for i, h in enumerate(hist_list):
        if stack:
            ax.fill_between(mid_bins, base, base + h, label=labels[i])
            base += h
        else:
            ax.step(bins, [0] + h.tolist(), label=labels[i])

    ## Add the axis labels, set limits and save
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def parallel_plot(
    path: str,
    df: pd.DataFrame,
    cols: list,
    rank_col: str = None,
    cmap: str = "viridis",
    curved: bool = True,
    curved_extend: float = 0.1,
    groupby_methods: list = None,
    highlight_best: bool = False,
    do_sort: bool = True,
    alpha: float = 0.3,
    class_thresh=10,
) -> None:
    """
    Create a parallel coordinates plot from pandas dataframe
    args:
        path: Location of output plot
        df: dataframe
        cols: columns to use along the x axis
    kwargs:
        rank_col: The name of the column to use for ranking, otherwise takes last
        cmap: Colour palette to use for ranking of lines
        curved: Use spline interpolation along lines
        curved_extend: Fraction extension in y axis, adjust to contain curvature
        groupby_methods: List of aggr methods to include for each categorical column
        highlight_best: Highlight the best row with a darker line
        do_sort: Sort dataframe by rank column, best are drawn last -> more visible
        alpha: Opacity of each line
        class_thresh: Minimum unique values before ticks are treated as classes
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

        ## For continuous data (more than class_thresh unique values)
        if (col_data.dtype == float) & (len(np.unique(col_data)) > class_thresh):

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

        ## For categorical data (less than class_thresh unique values)
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
            "alpha": 1 if lne == best_idx else alpha,
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
    cbar = fig.colorbar(
        sm,
        pad=0,
        ticks=ax_info[rank_col][1],  ## Uses ranking attribute
        extend="both",  ## Extending to match the y extension passed 0 and 1
        extendrect=True,
        extendfrac=y_ax_ext,
    )

    ## The colour bar also needs axis labels
    cbar.ax.set_yticklabels(ax_info[rank_col][0])
    cbar.ax.set_xlabel(rank_col)  # For some reason this is not showing up now?
    cbar.set_label(rank_col)

    ## Change the plot layout and save
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, right=0.95)
    plt.savefig(Path(path + "_" + rank_col).with_suffix(".png"))


def plot_2d_hists(path, hist_list, hist_labels, ax_labels, bins):
    """Given a list of 2D histograms, plot them side by side as imshows"""

    ## Calculate the axis limits from the bins
    limits = (min(bins[0]), max(bins[0]), min(bins[1]), max(bins[1]))
    mid_bins = [(b[1:] + b[:-1]) / 2 for b in bins]

    ## Create the subplots
    fig, axes = plt.subplots(1, len(hist_list), figsize=(8, 4))

    ## For each histogram to be plotted
    for i in range(len(hist_list)):
        axes[i].set_xlabel(ax_labels[0])
        axes[i].set_title(hist_labels[i])
        axes[i].imshow(
            hist_list[i], cmap="viridis", origin="lower", extent=limits, norm=LogNorm()
        )
        axes[i].contour(*mid_bins, np.log(hist_list[i] + 1e-4), colors="k", levels=10)

    axes[0].set_ylabel(ax_labels[1])
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_latent_space(path, latents, labels=None, n_classes=None):
    """Plot the latent space marginal distributions of a VAE"""

    ## If there are labels then we do multiple lines per datapoint
    if labels is not None and n_classes is None:
        unique_lab = np.unique(labels)
    elif n_classes is not None:
        unique_lab = np.arange(n_classes)
    else:
        unique_lab = [-1]

    ## Get the number of plots based on the dimension of the latents
    lat_dim = min(8, latents.shape[-1])

    ## Create the figure with the  correct number of plots
    fig, axis = plt.subplots(2, int(np.ceil(lat_dim / 2)), figsize=(8, 4))
    axis = axis.flatten()

    ## Plot the distributions of the marginals
    for dim in range(lat_dim):

        ## Make a seperate plot for each of the unique labels
        for lab in unique_lab:

            ## If the lab is -1 then it means use all
            if lab == -1:
                mask = np.ones((len(latents))).astype("bool")
            else:
                mask = labels == lab

            ## Use the selected info for making the histogram
            x_data = latents[mask, dim]
            hist, edges = np.histogram(x_data, bins=30, density=True)
            hist = np.insert(hist, 0, hist[0])
            axis[dim].step(edges, hist, label=lab)

        ## Plot the standard gaussian which should be the latent distribution
        x_space = np.linspace(-4, 4, 100)
        axis[dim].plot(x_space, gaussian(x_space), "--k")

        ## Remove the axis ticklabels
        axis[dim].set_xticklabels([])
        axis[dim].set_yticklabels([])

    axis[0].legend()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
