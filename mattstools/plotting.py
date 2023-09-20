"""A collection of plotting scripts for standard uses."""

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib
import matplotlib.axes._axes as axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline
from scipy.stats import binned_statistic, pearsonr

# from .numpy_utils import mid_points, undo_mid

# Some defaults for my plots to make them look nicer
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.edgecolor"] = "1"
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["legend.fontsize"] = 11


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Return only a portion of a matplotlib colormap."""
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def gaussian(x_data, mu=0, sig=1):
    """Return the value of the gaussian distribution."""
    return (
        1
        / np.sqrt(2 * np.pi * sig**2)
        * np.exp(-((x_data - mu) ** 2) / (2 * sig**2))
    )


def plot_profiles(
    x_list: np.ndarray,
    y_list: np.ndarray,
    data_labels: list,
    ylabel: str,
    xlabel: str,
    central_statistic: str | Callable = "mean",
    up_statistic: str | Callable = "std",
    down_statistic: str | Callable = "std",
    bins: int | list | np.ndarray = 50,
    figsize: tuple = (5, 4),
    hist_kwargs: list | None = None,
    err_kwargs: list | None = None,
    legend_kwargs: dict | None = None,
    path: Path | None = None,
    return_fig: bool = False,
    return_img: bool = False,
) -> None:
    """Plot and save a profile plot."""

    assert len(x_list) == len(y_list)

    # Make sure the kwargs are lists too
    if not isinstance(hist_kwargs, list):
        hist_kwargs = len(x_list) * [hist_kwargs]
    if not isinstance(err_kwargs, list):
        err_kwargs = len(x_list) * [err_kwargs]

    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        # Get the basic histogram to setup the counts and edges
        hist, bin_edges = np.histogram(x, bins)

        # Get the central values for the profiles
        central = binned_statistic(x, y, central_statistic, bin_edges)
        central_vals = central.statistic

        # Get the up and down values for the statistic
        up_vals = binned_statistic(x, y, up_statistic, bin_edges).statistic
        if not (up_statistic == "std" and down_statistic == "std"):
            down_vals = binned_statistic(x, y, down_statistic, bin_edges).statistic
        else:
            down_vals = up_vals

        # Correct based on the uncertainty of the mean
        if up_statistic == "std":
            up_vals = central_vals + up_vals / np.sqrt(hist + 1e-8)
        if down_statistic == "std":
            down_vals = central_vals - down_vals / np.sqrt(hist + 1e-8)

        # Get the additional keyword arguments for the histograms
        if hist_kwargs[i] is not None and bool(hist_kwargs[i]):
            h_kwargs = deepcopy(hist_kwargs[i])
        else:
            h_kwargs = {}

        # Use the stairs function to plot the histograms
        line = ax.stairs(central_vals, bin_edges, label=data_labels[i], **h_kwargs)

        # Get the additional keyword arguments for the histograms
        if err_kwargs[i] is not None and bool(err_kwargs[i]):
            e_kwargs = deepcopy(err_kwargs[i])
        else:
            e_kwargs = {"color": line._edgecolor, "alpha": 0.2, "fill": True}

        # Include the uncertainty in the plots as a shaded region
        ax.stairs(up_vals, bin_edges, baseline=down_vals, **e_kwargs)

    # Limits
    ylim1, ylim2 = ax.get_ylim()
    ax.set_ylim(top=ylim2 + 0.5 * (ylim2 - ylim1))
    ax.set_xlim([bin_edges[0], bin_edges[-1]])

    # Axis labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(**(legend_kwargs or {}))
    ax.grid(visible=True)

    # Final figure layout
    fig.tight_layout()

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)


def plot_corr_heatmaps(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bins: list,
    xlabel: str,
    ylabel: str,
    path: Optional[Path] = None,
    weights: np.ndarray | None = None,
    do_log: bool = True,
    equal_aspect: bool = True,
    cmap: str = "coolwarm",
    incl_line: bool = True,
    incl_cbar: bool = True,
    title: str = "",
    figsize=(6, 5),
    do_pearson=False,
    return_fig: bool = False,
    return_img: bool = False,
) -> None:
    """Plot and save a 2D heatmap, usually for correlation plots.

    Parameters
    ----------
    path : str
        Location of the output file.
    x_vals : array_like
        The values to put along the x-axis, usually truth.
    y_vals : array_like
        The values to put along the y-axis, usually reco.
    bins : array_like
        The bins to use, must be [xbins, ybins].
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    weights : array_like, optional
        The weight value for each x, y pair.
    do_log : bool, optional
        If the z axis should be the logarithm.
    equal_aspect : bool, optional
        Force the sizes of the axes' units to match.
    cmap : str, optional
        The name of the cmap to use for z values.
    incl_line : bool, optional
        If a y=x line should be included to show ideal correlation.
    incl_cbar : bool, optional
        Add the colour bar to the axis.
    figsize : tuple, optional
        The size of the output figure.
    title : str, optional
        Title for the plot.
    do_pearson : bool, optional
        Add the pearson correlation coeficient to the plot.
    do_pdf : bool, optional
        If the output should also contain a pdf version.
    """

    # Define the bins for the data
    if isinstance(bins, partial):
        bins = bins()
    if len(bins) != 2:
        bins = [bins, bins]

    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hist = ax.hist2d(
        x_vals.flatten(),
        y_vals.flatten(),
        bins=bins,
        weights=weights,
        cmap=cmap,
        norm="log" if do_log else None,
    )
    if equal_aspect:
        ax.set_aspect("equal")

    # Add line
    if incl_line:
        ax.plot([min(hist[1]), max(hist[1])], [min(hist[2]), max(hist[2])], "k--", lw=1)

    # Add colourbar
    if incl_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        try:  # Hacky solution to fix this sometimes failing if the values are shit
            fig.colorbar(hist[3], cax=cax, orientation="vertical", label="frequency")
        except Exception:
            pass

    # Axis labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title != "":
        ax.set_title(title)

    # Correlation coeficient
    if do_pearson:
        ax.text(
            0.05,
            0.92,
            f"r = {pearsonr(x_vals, y_vals)[0]:.3f}",
            transform=ax.transAxes,
            fontsize="large",
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    # Save the image
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    if return_fig:
        return fig
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    plt.close(fig)


def add_hist(
    ax: axes.Axes,
    data: np.ndarray,
    bins: np.ndarray,
    do_norm: bool = False,
    label: str = "",
    scale_factor: float | None = None,
    hist_kwargs: dict | None = None,
    err_kwargs: dict | None = None,
    do_err: bool = True,
) -> None:
    """Plot a histogram on a given axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot the histogram on.
    data : numpy.ndarray
        The data to plot as a histogram.
    bins : int
        The bin edges to use for the histogram
    do_norm : bool, optional
        Whether to normalize the histogram, by default False.
    label : str, optional
        The label to use for the histogram, by default "".
    scale_factor : float, optional
        A scaling factor to apply to the histogram, by default None.
    hist_kwargs : dict, optional
        Additional keyword arguments to pass to the histogram function, by default None.
    err_kwargs : dict, optional
        Additional keyword arguments to pass to the errorbar function, by default None.
    do_err : bool, optional
        Whether to include errorbars, by default True.

    Returns
    -------
    None
        The function only has side effects.
    """

    # Compute the histogram
    hist, _ = np.histogram(data, bins)
    hist_err = np.sqrt(hist)

    # Normalise the errors
    if do_norm:
        divisor = np.array(np.diff(bins), float) / hist.sum()
        hist = hist * divisor
        hist_err = hist_err * divisor

    # Apply the scale factors
    if scale_factor is not None:
        hist *= scale_factor
        hist_err *= scale_factor

    # Get the additional keyword arguments for the histograms
    if hist_kwargs is not None and bool(hist_kwargs):
        h_kwargs = hist_kwargs
    else:
        h_kwargs = {}

    # Use the stairs function to plot the histograms
    line = ax.stairs(hist, bins, label=label, **h_kwargs)

    # Get the additional keyword arguments for the error bars
    if err_kwargs is not None and bool(err_kwargs):
        e_kwargs = err_kwargs
    else:
        e_kwargs = {"color": line._edgecolor, "alpha": 0.5, "fill": True}

    # Include the uncertainty in the plots as a shaded region
    if do_err:
        ax.stairs(hist + hist_err, bins, baseline=hist - hist_err, **e_kwargs)


def quantile_bins(
    data: np.ndarray,
    bins: int = 50,
    low: float = 0.001,
    high: float = 0.999,
    axis: int | None = None,
) -> np.ndarray:
    """Return bin edges between quantile values of a dataset."""
    return np.linspace(*np.quantile(data, [low, high], axis=axis), bins)


def plot_multi_correlations(
    data_list: list | np.ndarray,
    data_labels: list,
    col_labels: list,
    n_bins: int = 50,
    bins: list | None = None,
    fig_scale: float = 1,
    n_kde_points: int = 50,
    do_err: bool = True,
    do_norm: bool = True,
    hist_kwargs: list | None = None,
    err_kwargs: list | None = None,
    legend_kwargs: dict | None = None,
    path: Path | str | None = None,
    return_img: bool = False,
    return_fig: bool = False,
) -> plt.Figure | None:
    """Plot multiple correlations in a matrix format.

    Parameters
    ----------
    data_list : list | np.ndarray
        List of data arrays to be plotted.
    data_labels : list
        List of labels for the data.
    col_labels : list
        List of column labels for the data.
    n_bins : int, optional
        Number of bins for the histogram, by default 50. Superseeded by bins
    bins : list | None, optional
        List of bin edges, by default None.
    fig_scale : float, optional
        Scaling factor for the figure size, by default 1.
    n_kde_points : int, optional
        Number of points for the KDE plot, by default 50.
    do_err : bool, optional
        If True, add error bars to the histogram, by default True.
    do_norm : bool, optional
        If True, normalize the histogram, by default True.
    hist_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the plotting function,
        by default None.
    err_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the error function,
        by default None.
    legend_kwargs : dict | None, optional
        Dictionary with keyword arguments for the legend function, by default None.
    path : Path | str, optional
        Path where to save the figure, by default None.
    return_img : bool, optional
        If True, return the image as a PIL.Image object instead of saving it to a file,
        by default False.
    return_fig : bool, optional
        If True, return the figure and axes objects instead of saving it to a file or
        returning an image object.

    Returns
    -------
    plt.Figure | None
        The figure object if `return_fig` is True. Otherwise returns None.
    """

    # Make sure the kwargs are lists too
    if not isinstance(hist_kwargs, list):
        hist_kwargs = len(data_list) * [hist_kwargs]
    if not isinstance(err_kwargs, list):
        err_kwargs = len(data_list) * [err_kwargs]

    # Create the figure with the many sub axes
    n_features = len(col_labels)
    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=((2 * n_features + 3) * fig_scale, (2 * n_features + 1) * fig_scale),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )

    # Define the binning as auto or not
    all_bins = []
    for n in range(n_features):
        if bins is None or bins[n] == "auto":
            all_bins.append(quantile_bins(data_list[0][:, n], bins=n_bins))
        else:
            all_bins.append(np.array(bins[n]))

    # Cycle through the rows and columns and set the axis labels
    for row in range(n_features):
        axes[0, 0].set_ylabel("A.U.", loc="top")
        if row != 0:
            axes[row, 0].set_ylabel(col_labels[row])
        for column in range(n_features):
            axes[-1, column].set_xlabel(col_labels[column])
            if column != 0:
                axes[row, column].set_yticklabels([])

            # Remove all ticks
            if row != n_features - 1:
                axes[row, column].tick_params(
                    axis="x", which="both", direction="in", labelbottom=False
                )
            if row == column == 0:
                axes[row, column].yaxis.set_ticklabels([])
            elif column > 0:
                axes[row, column].tick_params(
                    axis="y", which="both", direction="in", labelbottom=False
                )

            # For the diagonals they become histograms
            # Bins are based on the first datapoint in the list
            if row == column:
                b = all_bins[column]
                for i, d in enumerate(data_list):
                    add_hist(
                        axes[row, column],
                        d[:, row],
                        bins=b,
                        hist_kwargs=hist_kwargs[i],
                        err_kwargs=err_kwargs[i],
                        do_err=do_err,
                        do_norm=do_norm,
                    )
                    axes[row, column].set_xlim(b[0], b[-1])

            # If we are in the lower triange  fill using a contour plot
            elif row > column:
                x_bounds = np.quantile(data_list[0][:, column], [0.001, 0.999])
                y_bounds = np.quantile(data_list[0][:, row], [0.001, 0.999])
                for i, d in enumerate(data_list):
                    color = None
                    if hist_kwargs[i] is not None and "color" in hist_kwargs[i].keys():
                        color = hist_kwargs[i]["color"]
                    sns.kdeplot(
                        x=d[:, column],
                        y=d[:, row],
                        ax=axes[row, column],
                        alpha=0.4,
                        levels=3,
                        color=color,
                        fill=True,
                        clip=[x_bounds, y_bounds],
                        gridsize=n_kde_points,
                    )
                    axes[row, column].set_xlim(x_bounds)
                    axes[row, column].set_ylim(y_bounds)

            # If we are in the upper triangle we set visibility off
            else:
                axes[row, column].set_visible(False)

    # Create some invisible lines which will be part of the legend
    for i, d in enumerate(data_list):
        color = None
        if hist_kwargs[i] is not None and "color" in hist_kwargs[i].keys():
            color = hist_kwargs[i]["color"]
        axes[row, column].plot([], [], label=data_labels[i], color=color)
    fig.legend(**(legend_kwargs or {}))

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)
    return


def plot_multi_hists_2(
    data_list: Union[list, np.ndarray],
    data_labels: Union[list, str],
    col_labels: Union[list, str],
    path: Optional[Union[Path, str]] = None,
    scale_factors: Optional[list] = None,
    do_err: bool = False,
    do_norm: bool = False,
    bins: Union[list, str, partial] = "auto",
    logy: bool = False,
    y_label: Optional[str] = None,
    ylims: list | tuple | None = None,
    ypad: float = 1.5,
    rat_ylim: tuple = (0, 2),
    rat_label: Optional[str] = None,
    scale: int = 5,
    do_legend: bool = True,
    hist_kwargs: Optional[list] = None,
    err_kwargs: Optional[list] = None,
    legend_kwargs: Optional[list] = None,
    extra_text: Optional[list] = None,
    incl_overflow: bool = True,
    incl_underflow: bool = True,
    do_ratio_to_first: bool = False,
    axis_callbacks: list[Callable] | None = None,
    return_fig: bool = False,
    return_img: bool = False,
) -> Union[plt.Figure, None]:
    """Plot multiple histograms given a list of 2D tensors/arrays.

    - Performs the histogramming here
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis
    - If the tensor being passed is 3D it will average them and combine the uncertainty

    Parameters
    ----------
    data_list:
        A list of tensors or numpy arrays, each col will be a seperate axis
    data_labels:
        A list of labels for each tensor in data_list
    col_labels:
        A list of labels for each column/axis
    path:
        The save location of the plots (include img type)
    scale_factors:
        List of scalars to be applied to each histogram
    do_err:
        If the statistical errors should be included as shaded regions
    do_norm:
        If the histograms are to be a density plot
    bins:
        List of bins to use for each axis, can use numpy's strings
    logy:
        If we should use the log in the y-axis
    y_label:
        Label for the y axis of the plots
    ylims:
        The y limits for all plots, should be a tuple per plot
    ypad:
        The amount by which to pad the whitespace above the plots
    rat_ylim:
        The y limits of the ratio plots
    rat_label:
        The label for the ratio plot
    scale:
        The size in inches for each subplot
    do_legend:
        If the legend should be plotted
    hist_kwargs:
        Additional keyword arguments for the line for each histogram
    legend_kwargs:
        Extra keyword arguments to pass to the legend constructor
    extra_text:
        Extra text to put on each axis (same length as columns)
    incl_overflow:
        Have the final bin include the overflow
    incl_underflow:
        Have the first bin include the underflow
    do_ratio_to_first:
        Include a ratio plot to the first histogram in the list
    as_pdf:
        Also save an additional image in pdf format
    return_fig:
        Return the figure (DOES NOT CLOSE IT!)
    return_img:
        Return a PIL image (will close the figure)
    """

    # Make the arguments lists for generality
    if not isinstance(data_list, list):
        data_list = [data_list]
    if isinstance(data_labels, str):
        data_labels = [data_labels]
    if isinstance(col_labels, str):
        col_labels = [col_labels]
    if not isinstance(bins, list):
        bins = data_list[0].shape[-1] * [bins]
    if not isinstance(scale_factors, list):
        scale_factors = len(data_list) * [scale_factors]
    if not isinstance(hist_kwargs, list):
        hist_kwargs = len(data_list) * [hist_kwargs]
    if not isinstance(err_kwargs, list):
        err_kwargs = len(data_list) * [err_kwargs]
    if not isinstance(extra_text, list):
        extra_text = len(col_labels) * [extra_text]
    if not isinstance(legend_kwargs, list):
        legend_kwargs = len(col_labels) * [legend_kwargs]
    if not isinstance(ylims, list):
        ylims = len(col_labels) * [ylims]
    if not isinstance(axis_callbacks, list):
        axis_callbacks = len(col_labels) * [axis_callbacks]

    # Cycle through the datalist and ensure that they are 2D, as each column is an axis
    for data_idx in range(len(data_list)):
        if data_list[data_idx].ndim < 2:
            data_list[data_idx] = np.expand_dims(data_list[data_idx], -1)

    # Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = data_list[0].shape[-1]

    # Make sure that all the list lengths are consistant
    assert len(data_labels) == n_data
    assert len(col_labels) == n_axis
    assert len(bins) == n_axis

    # Make sure the there are not too many subplots
    if n_axis > 20:
        raise RuntimeError("You are asking to create more than 20 subplots!")

    # Create the figure and axes lists
    dims = np.array([1, n_axis])  # Subplot is (n_rows, n_columns)
    size = np.array([n_axis, 1.0])  # Size is (width, height)
    if do_ratio_to_first:
        dims *= np.array([2, 1])  # Double the number of rows
        size *= np.array([1, 1.2])  # Increase the height
    fig, axes = plt.subplots(
        *dims,
        figsize=tuple(scale * size),
        gridspec_kw={"height_ratios": [3, 1] if do_ratio_to_first else {1}},
        squeeze=False,
    )

    # Cycle through each axis and determine the bins that should be used
    # Automatic/Interger bins are replaced using the first item in the data list
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]
        if isinstance(ax_bins, partial):
            ax_bins = ax_bins()

        # If the axis bins was specified to be 'auto' or another numpy string
        if isinstance(ax_bins, str):
            unq = np.unique(data_list[0][:, ax_idx])
            n_unique = len(unq)

            # If the number of datapoints is less than 10 then use even spacing
            if 1 < n_unique < 10:
                ax_bins = (unq[1:] + unq[:-1]) / 2  # Use midpoints, add final, initial
                ax_bins = np.append(ax_bins, unq.max() + unq.max() - ax_bins[-1])
                ax_bins = np.insert(ax_bins, 0, unq.min() + unq.min() - ax_bins[0])

            elif ax_bins == "quant":
                ax_bins = quantile_bins(data_list[0][:, ax_idx])

        # Numpy function to get the bin edges, catches all other cases (int, etc)
        ax_bins = np.histogram_bin_edges(data_list[0][:, ax_idx], bins=ax_bins)

        # Replace the element in the array with the edges
        bins[ax_idx] = ax_bins

    # Cycle through each of the axes
    for ax_idx in range(n_axis):
        # Get the bins for this axis
        ax_bins = bins[ax_idx]

        # Cycle through each of the data arrays
        for data_idx in range(n_data):
            # Apply overflow and underflow (make a copy)
            data = np.copy(data_list[data_idx][..., ax_idx]).squeeze()
            if incl_overflow:
                data = np.minimum(data, ax_bins[-1])
            if incl_underflow:
                data = np.maximum(data, ax_bins[0])

            # If the data is still a 2D tensor treat it as a collection of histograms
            if data.ndim > 1:
                h = []
                for dim in range(data.shape[-1]):
                    h.append(np.histogram(data[:, dim], ax_bins)[0])

                # Nominal and err is based on chi2 of same value, mult measurements
                hist = 1 / np.mean(1 / np.array(h), axis=0)
                hist_err = np.sqrt(1 / np.sum(1 / np.array(h), axis=0))

            # Otherwise just calculate a single histogram
            else:
                hist, _ = np.histogram(data, ax_bins)
                hist_err = np.sqrt(hist)

            # Manually do the density so that the error can be scaled
            if do_norm:
                divisor = np.array(np.diff(ax_bins), float) / hist.sum()
                hist = hist * divisor
                hist_err = hist_err * divisor

            # Apply the scale factors
            if scale_factors[data_idx] is not None:
                hist *= scale_factors
                hist_err *= scale_factors

            # Save the first histogram for the ratio plots
            if data_idx == 0:
                denom_hist = hist
                denom_err = hist_err

            # Get the additional keyword arguments for the histograms and errors
            if hist_kwargs[data_idx] is not None and bool(hist_kwargs[data_idx]):
                h_kwargs = deepcopy(hist_kwargs[data_idx])
            else:
                h_kwargs = {}

            # Use the stair function to plot the histograms
            line = axes[0, ax_idx].stairs(
                hist, ax_bins, label=data_labels[data_idx], **h_kwargs
            )

            if err_kwargs[data_idx] is not None and bool(err_kwargs[data_idx]):
                e_kwargs = deepcopy(err_kwargs[data_idx])
            else:
                e_kwargs = {"color": line._edgecolor, "alpha": 0.2, "fill": True}

            # Include the uncertainty in the plots as a shaded region
            if do_err:
                axes[0, ax_idx].stairs(
                    hist + hist_err, ax_bins, baseline=hist - hist_err, **e_kwargs
                )

            # Add a ratio plot
            if do_ratio_to_first:
                if hist_kwargs[data_idx] is not None and bool(hist_kwargs[data_idx]):
                    ratio_kwargs = deepcopy(hist_kwargs[data_idx])
                else:
                    ratio_kwargs = {
                        "color": line._edgecolor,
                        "linestyle": line._linestyle,
                    }
                ratio_kwargs["fill"] = False  # Never fill a ratio plot

                # Calculate the new ratio values with their errors
                rat_hist = hist / denom_hist
                rat_err = rat_hist * np.sqrt(
                    (hist_err / hist) ** 2 + (denom_err / denom_hist) ** 2
                )

                # Plot the ratios
                axes[1, ax_idx].stairs(rat_hist, ax_bins, **ratio_kwargs)

                # Use a standard shaded region for the errors
                if do_err:
                    axes[1, ax_idx].stairs(
                        rat_hist + rat_err,
                        ax_bins,
                        baseline=rat_hist - rat_err,
                        **e_kwargs,
                    )

    # Cycle again through each axis and apply editing
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]

        # X axis
        axes[0, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])
        if do_ratio_to_first:
            axes[0, ax_idx].set_xticklabels([])
            axes[1, ax_idx].set_xlabel(col_labels[ax_idx])
            axes[1, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])
        else:
            axes[0, ax_idx].set_xlabel(col_labels[ax_idx])

        # Y axis
        if logy:
            axes[0, ax_idx].set_yscale("log")
        if ylims[ax_idx] is not None:
            axes[0, ax_idx].set_ylim(*ylims[ax_idx])
        else:
            _, ylim2 = axes[0, ax_idx].get_ylim()
            if logy:
                axes[0, ax_idx].set_ylim(top=np.exp(np.log(ylim2) + ypad))
            else:
                axes[0, ax_idx].set_ylim(top=ylim2 * ypad)
        if y_label is not None:
            axes[0, ax_idx].set_ylabel(y_label)
        elif do_norm:
            axes[0, ax_idx].set_ylabel("Normalised Entries")
        else:
            axes[0, ax_idx].set_ylabel("Entries")

        # Ratio Y axis
        if do_ratio_to_first:
            if rat_ylim is not None:
                axes[1, ax_idx].set_ylim(rat_ylim)
            if rat_label is not None:
                axes[1, ax_idx].set_ylabel(rat_label)
            else:
                axes[1, ax_idx].set_ylabel(f"Ratio to {data_labels[0]}")

            # Ratio X line:
            axes[1, ax_idx].hlines(
                1, *axes[1, ax_idx].get_xlim(), colors="k", zorder=-9999
            )

        # Extra text
        if extra_text[ax_idx] is not None:
            axes[0, ax_idx].text(**extra_text[ax_idx])

        # Legend
        if do_legend:
            lk = legend_kwargs[ax_idx] or {}
            axes[0, ax_idx].legend(**lk)

        # Any final callbacks to execute on the axis
        if axis_callbacks[ax_idx] is not None:
            axis_callbacks[ax_idx](fig, axes[0, ax_idx])

    # Final figure layout
    fig.tight_layout()
    if do_ratio_to_first:
        fig.subplots_adjust(hspace=0.08)  # For ratio plots minimise the h_space

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)


def parallel_plot(
    path: str,
    df: pd.DataFrame,
    cols: list,
    rank_col: str | None = None,
    cmap: str = "viridis",
    curved: bool = True,
    curved_extend: float = 0.1,
    groupby_methods: list | None = None,
    highlight_best: bool = False,
    do_sort: bool = True,
    alpha: float = 0.3,
    class_thresh=10,
) -> None:
    """Create a parallel coordinates plot from pandas dataframe.

    Parameters
    ----------
    path:
        Location of output plot
    df:
        dataframe
    cols:
        columns to use along the x axis
    rank_col:
        The name of the column to use for ranking, otherwise takes last
    cmap:
        Colour palette to use for ranking of lines
    curved:
        Use spline interpolation along lines
    curved_extend:
        Fraction extension in y axis, adjust to contain curvature
    groupby_methods:
        List of aggr methods to include for each categorical column
    highlight_best:
        Highlight the best row with a darker line
    do_sort:
        Sort dataframe by rank column, best are drawn last -> more visible
    alpha:
        Opacity of each line
    class_thresh:
        Minimum unique values before ticks are treated as classes
    """

    # Make sure that the rank column is the final column in the list
    if rank_col is not None:
        if rank_col in cols:
            cols.append(cols.pop(cols.index(rank_col)))
        else:
            cols.append(rank_col)
    rank_col = cols[-1]

    # Sort the dataframe by the rank column
    if do_sort:
        df.sort_values(by=rank_col, ascending=False, inplace=True)

    # Load the colourmap
    colmap = matplotlib.cm.get_cmap(cmap)

    # Create a value matrix for the y intercept points on each column for each line
    y_matrix = np.zeros((len(cols), len(df)))
    x_values = np.arange(len(cols))
    ax_info = {}  # Dict which will contain tick labels and values for each col

    # Cycle through each column
    for i, col in enumerate(cols):
        # Pull the column data from the dataframe
        col_data = df[col]

        # For continuous data (more than class_thresh unique values)
        if (col_data.dtype == float) & (len(np.unique(col_data)) > class_thresh):
            # Scale the range of data to [0,1] and save to matrix
            y_min = np.min(col_data)
            y_max = np.max(col_data)
            y_range = y_max - y_min
            y_matrix[i] = (col_data - y_min) / y_range

            # Create the ticks and tick labels for the axis
            nticks = 5  # Good number for most cases
            tick_labels = np.linspace(y_min, y_max, nticks, endpoint=True)
            tick_labels = [f"{s:.2f}" for s in tick_labels]
            tick_values = np.linspace(0, 1, nticks, endpoint=True)
            ax_info[col] = [tick_labels, tick_values]

        # For categorical data (less than class_thresh unique values)
        else:
            # Set the type for the data to categorical to pull out stats using pandas
            col_data = col_data.astype("category")
            cats = col_data.cat.categories
            cat_vals = col_data.cat.codes

            # Scale to the range [0,1] (special case for data with only one cat)
            if len(cats) == 1:
                y_matrix[i] = 0.5
            else:
                y_matrix[i] = cat_vals / cat_vals.max()

            # The tick labels include average performance using groupby
            if groupby_methods is not None and col != rank_col:
                groups = (
                    df[[col, rank_col]].groupby([col]).agg(groupby_methods)[rank_col]
                )

                # Create the tick labels by using all groupy results
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

            # Or they simply use the cat names
            else:
                tick_labels = cats

            # Create the tick locations and save in dict
            tick_values = np.unique(y_matrix[i])
            ax_info[col] = [tick_labels, tick_values]

    # Get the index of the best row
    best_idx = np.argmin(y_matrix[-1]) if highlight_best else -1

    # Create the plot
    fig, axes = plt.subplots(
        1, len(cols) - 1, sharey=False, figsize=(3 * len(cols) + 3, 5)
    )

    # Amount by which to extend the y axis ranges above the data range
    y_ax_ext = curved_extend if curved else 0.05

    # Cycle through each line (singe row in the original dataframe)
    for lne in range(len(df)):
        # Calculate spline function to use across all axes
        if curved:
            spline_fn = make_interp_spline(
                x_values, y_matrix[:, lne], k=3, bc_type="clamped"
            )

        # Keyword arguments for drawing the line
        lne_kwargs = {
            "color": colmap(y_matrix[-1, lne]),
            "alpha": 1 if lne == best_idx else alpha,
            "linewidth": 4 if lne == best_idx else None,
        }

        # Cycle through each axis (bridges one column to the next)
        for i, ax in enumerate(axes):
            # For splines
            if curved:
                # Plot the spline using a more dense x space spanning the axis window
                x_space = np.linspace(i, i + 1, 20)
                ax.plot(x_space, spline_fn(x_space), **lne_kwargs)

            # For simple line connectors
            else:
                ax.plot(x_values[[i, i + 1]], y_matrix[[i, i + 1], lne], **lne_kwargs)

            # Set the axis limits, y included extensions, x is limited to window
            ax.set_ylim(0 - y_ax_ext, 1 + y_ax_ext)
            ax.set_xlim(i, i + 1)

    # For setting the axis ticklabels
    for dim, (ax, col) in enumerate(zip(axes, cols)):
        # Reduce the x axis ticks to the start of the plot for column names
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.set_xticklabels([cols[dim]])

        # The y axis ticks were calculated and saved in the info dict
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])

    # Create the colour bar on the far right side of the plot
    norm = matplotlib.colors.Normalize(0, 1)  # Map data into the colour range [0, 1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Required for colourbar
    cbar = fig.colorbar(
        sm,
        pad=0,
        ticks=ax_info[rank_col][1],  # Uses ranking attribute
        extend="both",  # Extending to match the y extension passed 0 and 1
        extendrect=True,
        extendfrac=y_ax_ext,
    )

    # The colour bar also needs axis labels
    cbar.ax.set_yticklabels(ax_info[rank_col][0])
    cbar.ax.set_xlabel(rank_col)  # For some reason this is not showing up now?
    cbar.set_label(rank_col)

    # Change the plot layout and save
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, right=0.95)
    plt.savefig(Path(path + "_" + rank_col).with_suffix(".png"))
    return


def plot_2d_hists(path, hist_list, hist_labels, ax_labels, bins):
    """Given a list of 2D histograms, plot them side by side as imshows."""

    # Calculate the axis limits from the bins
    limits = (min(bins[0]), max(bins[0]), min(bins[1]), max(bins[1]))
    mid_bins = [(b[1:] + b[:-1]) / 2 for b in bins]

    # Create the subplots
    fig, axes = plt.subplots(1, len(hist_list), figsize=(8, 4))

    # For each histogram to be plotted
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


def plot_latent_space(
    path,
    latents,
    labels=None,
    n_classes=None,
    return_fig: bool = False,
    return_img=False,
):
    """Plot the latent space marginal distributions of a VAE."""

    # If there are labels then we do multiple lines per datapoint
    if labels is not None and n_classes is None:
        unique_lab = np.unique(labels)
    elif n_classes is not None:
        unique_lab = np.arange(n_classes)
    else:
        unique_lab = [-1]

    # Get the number of plots based on the dimension of the latents
    lat_dim = min(8, latents.shape[-1])

    # Create the figure with the  correct number of plots
    fig, axis = plt.subplots(2, int(np.ceil(lat_dim / 2)), figsize=(8, 4))
    axis = axis.flatten()

    # Plot the distributions of the marginals
    for dim in range(lat_dim):
        # Make a seperate plot for each of the unique labels
        for lab in unique_lab:
            # If the lab is -1 then it means use all
            if lab == -1:
                mask = np.ones(len(latents)).astype("bool")
            else:
                mask = labels == lab

            # Use the selected info for making the histogram
            x_data = latents[mask, dim]
            hist, edges = np.histogram(x_data, bins=30, density=True)
            hist = np.insert(hist, 0, hist[0])
            axis[dim].step(edges, hist, label=lab)

        # Plot the standard gaussian which should be the latent distribution
        x_space = np.linspace(-4, 4, 100)
        axis[dim].plot(x_space, gaussian(x_space), "--k")

        # Remove the axis ticklabels
        axis[dim].set_xticklabels([])
        axis[dim].set_yticklabels([])

    axis[0].legend()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(Path(path).with_suffix(".png"))
    if return_fig:
        return fig
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    plt.close(fig)
