import torch as T
import torch.nn as nn

def get_timesteps(n_steps: int, t_min: float, p: float) -> T.Tensor:
    """Generate variable timesteps working back from 1 to t_min

    Args:
        n_steps: The number of time steps
        t_min: The minimum time
        p: The degree of curvature, p=1 equal step size, recommened 7 for diffusion

    """
    idx = T.arange(0, n_steps).float()
    times = (1 + idx / (n_steps - 1) * (t_min**(1/p) - 1))**p
    return times

def heun_sampler(
    model,
    initial_noise: T.Tensor,
    time_steps: T.Tensor,
    keep_all: bool = False,
    mask: T.Tensor | None = None,
    ctxt: T.BoolTensor | None = None,
    clip_predictions: tuple | None = None,
) -> None:

    # Get the initial noise for generation and the number of sammples
    batch_size = initial_noise.shape[0]
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)
    all_stages = [initial_noise]
    num_steps = len(time_steps)

    # Start with the initial noise
    x = initial_noise

    # Start iterating through each timestep
    for i in range(num_steps-1):

        # Expancd the diffusion times for the number of samples in the batch
        diff_times = T.full((batch_size,1), time_steps[i], device=model.device)
        diff_times_next = T.full((batch_size,1), time_steps[i+1], device=model.device)

        # Calculate the derivative and apply the euler step
        # Note that this is the same as a single DDIM step! Triple checked!
        d = (x - model.denoise(x, diff_times, mask, ctxt)) / time_steps[i]
        x_next = x + (diff_times_next-diff_times).view(expanded_shape) * d

        # Apply the second order correction as long at the time doesnt go to zero
        if time_steps[i+1] > 0:
            d_next = (x_next - model.denoise(x_next, diff_times_next, mask, ctxt)) / time_steps[i+1]
            x_next = x + (diff_times_next-diff_times).view(expanded_shape) * (d + d_next) / 2

        # Update the track
        x = x_next
        if keep_all:
            all_stages.append(x)

    return x, all_stages


def stochastic_sampler(
    model,
    initial_noise: T.Tensor,
    time_steps: T.Tensor,
    keep_all: bool = False,
    mask: T.Tensor | None = None,
    ctxt: T.BoolTensor | None = None,
    clip_predictions: tuple | None = None,
) -> None:

    # Get the initial noise for generation and the number of sammples
    batch_size = initial_noise.shape[0]
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)
    all_stages = [initial_noise]
    num_steps = len(time_steps)

    # Start with the initial noise
    x = initial_noise

    # Start iterating through each timestep
    for i in range(num_steps-1):

        # Expancd the diffusion times for the number of samples in the batch
        diff_times = T.full((batch_size,1), time_steps[i], device=model.device)
        diff_times_next = T.full((batch_size,1), time_steps[i+1], device=model.device)

        # Calculate the derivative and apply the euler step
        d = (x - model.denoise(x, diff_times, mask, ctxt)) / time_steps[i]
        x_next = x + (diff_times_next-diff_times).view(expanded_shape) * d

        # Apply the second order correction as long at the time doesnt go to zero
        if time_steps[i+1] > 0:
            d_next = (x_next - model.denoise(x_next, diff_times_next, mask, ctxt)) / time_steps[i+1]
            x_next = x + (diff_times_next-diff_times).view(expanded_shape) * (d + d_next) / 2

        # Update the track
        x = x_next
        if keep_all:
            all_stages.append(x)

    return x, all_stages
