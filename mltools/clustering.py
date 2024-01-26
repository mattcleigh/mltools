"""Clustering algorithms for pytorch tensors."""

import torch as T
from tqdm import tqdm


def kmeans(
    x: T.Tensor,
    num_clusters: int,
    tol_per_dimension=1e-4,
    max_iter=1000,
) -> T.Tensor:
    """Perform kmeans and return the cluster centers."""

    # Make sure that the input is a float
    x = x.float()

    # Initialise the cluster centers by randomly sampling from the input
    indices = T.multinomial(T.ones(x.shape[0]), num_clusters, replacement=False)
    cluster_centers = x[indices]

    # Iterate until convergence
    tqdm_meter = tqdm(desc="Running kmeans")
    for iteration in range(max_iter):
        # Calculate the distance between the inputs and the cluster centers
        dist = T.cdist(x, cluster_centers)

        # Assign each input to the closest cluster center
        idxes = T.argmin(dist, dim=-1).unsqueeze(-1)

        # For each cluster, calculate the mean of the inputs assigned to it
        cluster_means = T.zeros_like(cluster_centers)
        cluster_means.scatter_add_(0, idxes.expand_as(x), x)
        cluster_means /= T.bincount(idxes.squeeze()).unsqueeze(-1)

        # Calculate the total shift per dimension of all cluster centers
        shift = T.norm(cluster_means - cluster_centers, dim=-1).sum()
        shift /= x.shape[-1]

        # Replace the cluster centers with the new means
        cluster_centers = cluster_means

        # Update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f"{iteration}",
            center_shift=f"{shift:0.6f}",
            tol_per_dimension=f"{tol_per_dimension:0.6f}",
        )
        tqdm_meter.update()
        if shift < tol_per_dimension:
            break

    tqdm_meter.close()

    # Print a warning if the iteration limit was reached
    if iteration == max_iter - 1:
        print("Warning: kmeans reached maximum iterations without convergence.")

    return cluster_means
