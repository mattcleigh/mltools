"""
A collection of plotting scripts specifically for graph objects and networks
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_bar_graphs(
    path: Path,
    real_nodes: list,
    real_mask: list,
    gen_nodes: list,
    gen_mask: list,
    labels: list = None,
):
    """Plot a series of scatter plots using batch of generated point clouds
    - Creates a multiplot with n_samples rows and n_stages columns
    - Shows the generated point clouds after each stage.
    - New nodes added each stage are shown in red while the existing are shown in blue
    - Existing nodes have arrows from their previous position to highlight node drift
    - Target point cloud is shown in each plot as gray

    args:
        path: The location to save the image
        real_nodes: A list of real point clouds [(num_samples), n_nodes, node_dim]
        real_mask: A list of real masks of shape [(num_samples), n_nodes, node_dim]
        gen_nodes: A list of generated point clouds [(num_stages), num_samples, n_nodes]
        gen_mask: A list of generated masksof shape [(num_stages), num_samples, n_nodes]
    kwargs:
        labels: The labels for each of the num_samples
    """

    ## The dimensions of all of the subplots
    n_cols = len(real_nodes)
    n_rows = len(gen_nodes)
    n_dim = real_nodes[0].shape[-1]

    ## Only works for 2, 3 dimensional point clouds
    if n_dim not in [2, 3]:
        raise ValueError("Can only do 2 and 3D reconstructions!")

    ## For a 3 dimensional plot only plot the top, middle and final
    if n_dim == 3 and n_rows > 3:
        gen_nodes = [gen_nodes[0], gen_nodes[n_rows // 2], gen_nodes[-1]]
        gen_mask = [gen_mask[0], gen_mask[n_rows // 2], gen_mask[-1]]
        n_rows = 3

    ## Create the figure and axes
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3 * n_dim / 2, n_rows * 3 * n_dim / 2),
        subplot_kw=dict(projection="3d") if n_dim == 3 else {},
    )
    axes = np.reshape(axes, (n_rows, n_cols))

    ## Style for the node drift arrows
    arr_style = "Simple, tail_width=0.5, head_width=4, head_length=8"

    ## Add in a fake empty mask so first nodes are new
    if n_dim == 2:
        gen_mask.insert(0, np.zeros_like(gen_mask[0], dtype=bool))

    ## Loop through all of the subplots axes
    for i in range(n_rows):
        for j in range(n_cols):
            axis = axes[i, j]

            ## Plot the real nodes
            axis.plot(
                *real_nodes[j][(real_mask[j])].T, "o", color="grey", label=labels[i]
            )

            if n_dim == 2:
                ## Plot the unchanged nodes in blue
                axis.plot(
                    *gen_nodes[i][j][gen_mask[i + 1][j] * gen_mask[i][j]].T,
                    "o",
                    color="blue"
                )

                ## Plot the arrows showing the node transitions
                starts = gen_nodes[i - 1][j][gen_mask[i][j]]
                fins = gen_nodes[i][j][gen_mask[i][j]]
                for s, f in zip(starts, fins):
                    arrow = FancyArrowPatch(posA=s, posB=f, arrowstyle=arr_style)
                    axis.add_patch(arrow)

                ## Plot the new nodes in red
                axis.plot(
                    *gen_nodes[i][j][(gen_mask[i + 1][j] != gen_mask[i][j])].T,
                    "o",
                    color="red"
                )

            if n_dim == 3:
                axis.scatter(*real_nodes[j][(real_mask[j])].T, "o", color="grey")
                axis.scatter(*gen_nodes[i][j][(gen_mask[i][j])].T, "o", color="red")

            ## Set the axis parameters and limits
            axis.set_xlim([-0.5, 0.5])
            axis.set_ylim([-0.5, 0.5])
            if n_dim == 3:
                axis.set_xlim([-1, 1])
                axis.set_ylim([-1, 1])
                axis.set_zlim([-1, 1])
                axis.zaxis.set_ticklabels([])
                axis.zaxis.set_ticks_position("none")
            else:
                axis.set_aspect("equal", adjustable="box")
            axis.xaxis.set_ticklabels([])
            axis.yaxis.set_ticklabels([])
            axis.xaxis.set_ticks_position("none")
            axis.yaxis.set_ticks_position("none")

            ## Add the legend to the first row only
            if i == 0:
                axis.legend()

    fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
