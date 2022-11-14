from typing import Tuple
import math
import torch as T


def cosine_encoding(
    x: T.Tensor,
    outp_dim: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> T.Tensor:
    """Computes a positional cosine encodings with an increasing series of frequencies

    The frequencies either increase linearly or exponentially (default).
    The latter is good for when max_value is large and extremely high sensitivity to the
    input is required.
    If inputs greater than the max value are provided, the outputs become degenerate.
    If inputs smaller than the min value are provided, the inputs the the cosine will
    be both positive and negative, which may lead degenerate outputs.

    Always make sure that the min and max bounds are not exceeded!

    Args:
        x: The input, the final dimension is encoded. If 1D then it will be unqueezed
        out_dim: The dimension of the output encoding
        min_value: Added to x (and max) as cosine embedding works with positive inputs
        max_value: The maximum expected value, sets the scale of the lowest frequency
        frequency_scaling: Either 'linear' or 'exponential'

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if T.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if T.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = T.arange(outp_dim, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = T.arange(1, outp_dim + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    return T.cos((x + min_value) * freqs * math.pi / (max_value + min_value))


def diffusion_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-2
) -> Tuple[T.Tensor, T.Tensor]:
    """Calculates the signal and noise rate for any point in the diffusion processes

    Using continuous diffusion times between 0 and 1 which make switching between
    different numbers of diffusion steps between training and testing much easier.
    Returns only the values needed for the jump forward diffusion step and the reverse
    DDIM step.
    These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
    and noise_rate respectively.

    The jump forward diffusion process is simply a weighted sum of:
        input * signal_rate + eps * noise_rate

    Uses a cosine annealing schedule as proposed in
    Proposed in https://arxiv.org/abs/2102.09672

    Args:
        diff_time: The time used to sample the diffusion scheduler
            Output will match the shape
            Must be between 0 and 1
        max_sr: The initial rate at the first step
        min_sr: How much signal is preserved at end of diffusion
            (can't be zero due to log)
    """

    ## Use cosine annealing, which requires switching from times -> angles
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    signal_rates = T.cos(diffusion_angles)
    noise_rates = T.sin(diffusion_angles)
    return signal_rates, noise_rates


class CosineEncoding:
    def __init__(
        self,
        outp_dim: int = 32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        frequency_scaling: str = "exponential",
    ) -> None:
        self.outp_dim = outp_dim
        self.min_value = min_value
        self.max_value = max_value
        self.frequency_scaling = frequency_scaling

    def __call__(self, inpt: T.Tensor) -> T.Tensor:
        return cosine_encoding(
            inpt, self.outp_dim, self.min_value, self.max_value, self.frequency_scaling
        )


class DiffusionSchedule:
    def __init__(self, max_sr: float = 1, min_sr: float = 1e-2) -> None:
        self.max_sr = max_sr
        self.min_sr = min_sr

    def __call__(self, time: T.Tensor) -> T.Tensor:
        return diffusion_shedule(time, self.max_sr, self.min_sr)
