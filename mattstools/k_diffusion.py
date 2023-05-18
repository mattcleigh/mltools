import math

import torch as T
import torch.nn as nn
from pyparsing import Mapping


@T.no_grad()
def multistep_consistency_sampling(
    model: nn.Module, sigmas: T.Tensor, min_sigma: float, x: T.Tensor, extra_args
) -> T.Tensor:
    """Perform multistep consistency sampling from a consistency model.

    Parameters
    ----------
    model : callable
        The consistency model.
    sigmas : torch.Tensor
        A sequence of sigma values to iterate through.
    initial_noise : torch.Tensor
        The initial noise.

    Returns
    -------
    x : torch.Tensor
        The final sample.
    """
    extra_args = extra_args or {}
    sigma_shape = x.new_ones([x.shape[0], 1])

    x = model(x, sigmas[0] * sigma_shape, **extra_args)
    for sigma in sigmas:
        x_t = x + (sigma**2 - min_sigma**2).sqrt() * T.randn_like(x)
        x = model(x_t, sigma * sigma_shape)
    return x


def gaussian(x: T.Tensor, mu: T.Tensor, var: T.Tensor) -> T.Tensor:
    return T.exp(-((x - mu) ** 2) / (2 * var))


def ideal_denoise(noisy_data, data, sigma):
    gaus_term = gaussian(
        noisy_data.unsqueeze(1),
        data.unsqueeze(0),
        append_dims(sigma, noisy_data.dim() + 1) ** 2,
    )
    numerator = (gaus_term * data.unsqueeze(0)).sum(1)
    denoised = numerator / gaus_term.sum(1)

    return denoised


@T.no_grad()
def one_step_ideal_heun(x, data, sigma_start, sigma_end):
    """Apply just one step of the heun-solver to get two adjacent points on the
    PF-ODE."""

    # Denoise the sample, and calculate derivative and the time step
    denoised = ideal_denoise(x, data, sigma_start)
    d = (x - denoised) / append_dims(sigma_start, x.dim())
    dt = append_dims((sigma_end - sigma_start), x.dim())
    x_2 = x + d * dt

    # Heun's 2nd order method
    # denoised_2 = ideal_denoise(x_2, data, sigma_end)
    # d_2 = (x_2 - denoised_2) / append_dims(sigma_end, x.dim())
    # d_prime = (d + d_2) / 2
    # x = x + d_prime * dt

    return x_2


@T.no_grad()
def one_step_heun(model, x, sigma_start, sigma_end, extra_args):
    """Apply just one step of the heun-solver to get two adjacent points on the
    PF-ODE."""

    # Initial setup
    extra_args = extra_args or {}

    # Denoise the sample, and calculate derivative and the time step
    denoised = model(x, sigma_start, **extra_args)
    d = (x - denoised) / append_dims(sigma_start, x.dim())
    dt = append_dims((sigma_end - sigma_start), x.dim())
    x_2 = x + d * dt

    # Heun's 2nd order method
    denoised_2 = model(x_2, sigma_end, **extra_args)
    d_2 = (x_2 - denoised_2) / append_dims(sigma_end, x.dim())
    d_prime = (d + d_2) / 2
    x = x + d_prime * dt

    return x


def append_dims(x: T.Tensor, target_dims: int) -> T.Tensor:
    """Appends dimensions of 1 to the end of a tensor until it has target_dims
    dimensions."""
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))


def get_sigmas_karras(
    t_max: float, t_min: float, n_steps: int = 100, rho: float = 7
) -> T.Tensor:
    """Constructs the noise schedule of Karras et al. (2022)

    Args:
        t_max: The maximum/starting time
        t_min: The minimum/final time
        n_steps: The number of time steps
        p: The degree of curvature, p=1 equal step size, recommened 7 for diffusion
    """
    ramp = T.linspace(0, 1, n_steps)
    inv_rho = 1 / rho
    max_inv_rho = t_max**inv_rho
    min_inv_rho = t_min**inv_rho
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


@T.no_grad()
def sample_heun(
    model,
    x: T.Tensor,
    sigmas: T.Tensor,
    do_heun_step: bool = True,
    keep_all: bool = False,
    extra_args: Mapping | None = None,
) -> None:
    """Deterministic sampler using Heun's second order method.

    Parameters
    ----------
    model : nn.Module
        The model to generate samples from.
    x : Tensor
        The initial noise for generation.
    sigmas : Tensor
        The sequence of noise levels to generate samples.
    do_heun_step : bool, optional
        Whether to use Heun's 2nd order method or not. Default is True.
    keep_all : bool, optional
        Whether to store the samples at each step or not. Default is False.
    extra_args : Mapping[str, Any] or None, optional
        Extra arguments to pass to the model. Default is None.

    Returns
    -------
    tuple
        A tuple containing two elements.
        - The generated samples.
        - All the intermediate samples (if keep_all is True), otherwise None.

    Notes
    -----
    Hard coded such that t = sigma and s(t) = 1.
    Alg. 1 from the https://arxiv.org/pdf/2206.00364.pdf.
    """

    # Initial setup
    num_steps = len(sigmas) - 1
    all_stages = [x] if keep_all else None
    sigma_shape = x.new_ones([x.shape[0], 1])
    extra_args = extra_args or {}

    # Start iterating through each timestep
    for i in range(num_steps):
        # Denoise the sample, and calculate derivative and the time step
        denoised = model(x, sigmas[i] * sigma_shape, **extra_args)
        d = (x - denoised) / sigmas[i]
        dt = sigmas[i + 1] - sigmas[i]

        # Apply the integration step
        if not do_heun_step or sigmas[i + 1] == 0:
            # Euler step (=DDIM with this noise schedule)
            x = x + d * dt
        else:
            # Heun's 2nd order method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * sigma_shape, **extra_args)
            d_2 = (x_2 - denoised_2) / sigmas[i + 1]
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        # Update the track
        if keep_all:
            all_stages.append(x)

    return x, all_stages


@T.no_grad()
def sample_stochastic_heun(
    model,
    x: T.Tensor,
    sigmas: T.Tensor,
    do_heun_step: bool = True,
    keep_all: bool = False,
    s_churn: float = 40.0,
    s_tmin: float = 0.05,
    s_tmax: float = 50.0,
    s_noise: float = 1.003,
    extra_args: Mapping | None = None,
) -> None:
    """Stochastic sampler using Heun's second order method.

    Parameters
    ----------
    model : nn.Module
        The model to generate samples from.
    x : Tensor
        The initial noise for generation.
    sigmas : Tensor
        The sequence of noise levels to generate samples.
    do_heun_step : bool, optional
        Whether to use Heun's 2nd order method or not. Default is True.
    keep_all : bool, optional
        Whether to store the samples at each step or not. Default is False.
    s_churn : float, optional (default=40.0)
        Changes the time for the iteration by a small amount
    s_tmin : float, optional (default=0.05)
        The lower bound of sigma where the stochasticity is allowed
    s_tmax : float, optional (default=50.0)
        The upper bound of sigma where the stochasticity is allowed
    s_noise : float, optional (default=1.003)
        The std of the noise which is added to the sample
    extra_args : Mapping[str, Any] or None, optional
        Extra arguments to pass to the model. Default is None.

    Returns
    -------
    tuple
        A tuple containing two elements.
        - The generated samples.
        - All the intermediate samples (if keep_all is True), otherwise None.

    Notes:
    ------
    - Equivalent to the deterministic case if s_churn = 0
    - Alg. 2 from the https://arxiv.org/pdf/2206.00364.pdf
    - Hard coded such that t = sigma and s(t) = 1
    - Default s values are taken from emiprical results in the paper
    """

    # Initial setup
    num_steps = len(sigmas) - 1
    all_stages = [x] if keep_all else None
    sigma_shape = x.new_ones([x.shape[0], 1])
    extra_args = extra_args or {}

    # Start iterating through each timestep
    for i in range(num_steps):
        # Get gamma factor (time perturbation)
        gamma = (
            min(s_churn / num_steps, math.sqrt(2.0) - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )

        # Shift the sigma value and the sample using the noise based gamma
        sigma_hat = sigmas[i] * (1 + gamma)

        # Inject noise into x if the gamma value is above zero
        if gamma > 0:
            eps = T.randn_like(x) * s_noise
            x = x + eps * math.sqrt(sigma_hat**2 - sigmas[i] ** 2)

        # Denoise the sample, and calculate derivative and the time step
        denoised = model(x, sigma_hat * sigma_shape, **extra_args)
        d = (x - denoised) / sigma_hat
        dt = sigmas[i + 1] - sigma_hat

        # Apply the integration step
        if not do_heun_step or sigmas[i + 1] == 0:
            # Euler step (=DDIM with this noise schedule)
            x = x + d * dt
        else:
            # Heun's 2nd order method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * sigma_shape, **extra_args)
            d_2 = (x_2 - denoised_2) / sigmas[i + 1]
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        # Update the track
        if keep_all:
            all_stages.append(x)

    return x, all_stages
