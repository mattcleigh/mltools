"""Functions and classes used to define invertible transformations."""

from functools import partial
from typing import Any, Callable, Literal

import normflows as nf
import numpy as np
import torch as T
import torch.nn as nn
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.utils.masks import create_alternating_binary_mask
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE

from .mlp import MLP
from .torch_utils import base_modules, get_act


class PermuteEvenOdd(nf.flows.Flow):
    """Permutation features along the channel dimension swapping even and odd values."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, z, context=None) -> tuple:
        z1 = z[:, 0::2]
        z2 = z[:, 1::2]
        z = T.stack((z2, z1), dim=2).view(z.shape[0], -1)
        log_det = T.zeros(z.shape[0], device=z.device)
        return z, log_det

    def inverse(self, z, context=None) -> tuple:
        return self.forward(z, context)


class LULinear(nf.flows.Flow):
    """Invertible linear layer using LU decomposition."""

    def __init__(self, num_channels: int, identity_init: bool = True):
        super().__init__()
        self.linear = nf.flows.mixing._LULinear(
            num_channels, identity_init=identity_init
        )

    def use_cache(self, use_cache: bool = True) -> None:
        self.linear.use_cache(use_cache)

    def forward(self, z, context=None) -> tuple[T.Tensor, T.Tensor]:
        z, log_det = self.linear.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None) -> tuple:
        z, log_det = self.linear(z, context=context)
        return z, log_det.view(-1)


class CoupledRationalQuadraticSpline(nf.flows.Flow):
    """Overloaded class from normflows which allow init_identity."""

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        init_identity=True,
    ) -> None:
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            net = MLP(  # I find that my MLPs use context information better!
                inpt_dim=in_features,
                outp_dim=out_features,
                ctxt_dim=num_context_channels or 0,
                hddn_dim=num_hidden_channels,
                num_blocks=num_blocks,
                act_h=partial(activation),
                drp=dropout_probability,
                ctxt_in_hddn=True,
                ctxt_in_inpt=False,
            )
            if init_identity:
                nn.init.constant_(net.output_block.layers[0].weight, 0.0)
                nn.init.constant_(
                    net.output_block.layers[0].bias,
                    np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1),
                )
            return net

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=True,
        )

    def forward(self, z, context=None) -> tuple:
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None) -> tuple:
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


def rqs_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    num_stacks: int = 3,
    mlp_width: int = 32,
    mlp_depth: int = 2,
    mlp_act: Callable = nn.LeakyReLU,
    tail_bound: float = 4.0,
    dropout: float = 0.0,
    num_bins: int = 8,
    do_lu: bool = True,
    init_identity: bool = True,
    do_norm: bool = False,
    flow_type: Literal["made", "coupling"] = "coupling",
) -> nf.NormalizingFlow | nf.ConditionalNormalizingFlow:
    """Return a rational quadratic spline normalising flow."""

    # Normflows wants the activation function as a class
    if isinstance(mlp_act, str):
        mlp_act = get_act(mlp_act).__class__

    # Set the kwargs for the flow as expected by normflows
    kwargs = {
        "num_input_channels": xz_dim,
        "num_blocks": mlp_depth,
        "num_hidden_channels": mlp_width,
        "num_context_channels": ctxt_dim if ctxt_dim else None,
        "num_bins": num_bins,
        "tail_bound": tail_bound,
        "activation": mlp_act,
        "dropout_probability": dropout,
        "init_identity": init_identity,
    }

    # Determine the type of layers to be used in the flow
    if flow_type == "made":
        fn = nf.flows.AutoregressiveRationalQuadraticSpline
        perm = nf.flows.LULinearPermute if do_lu else nf.flows.Permute
    elif flow_type == "coupling":
        fn = CoupledRationalQuadraticSpline
        perm = LULinear if do_lu else None
    else:
        raise ValueError("Unrecognised flow type" % flow_type)

    flows = []
    for i in range(num_stacks):
        # For coupling layers we need to alternate the mask and don't need permute
        if flow_type == "coupling":
            kwargs["reverse_mask"] = i % 2 == 1

        # Add the flow
        flows += [fn(**kwargs)]

        # Add the permutation layer if required
        if perm is not None:
            flows += [perm(xz_dim)]

        # Add the normalisation layer
        if do_norm:
            flows += [nf.flows.ActNorm(xz_dim)]

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(xz_dim, trainable=False)

    # Return the full flow
    if ctxt_dim:
        return nf.ConditionalNormalizingFlow(q0=q0, flows=flows)
    return nf.NormalizingFlow(q0=q0, flows=flows)


def prepare_for_onnx(
    flowwrapper: nn.Module,
    dummy_input: Any,
    method: str = "sample",
) -> None:
    """Prepare a flow for export to ONNX primarily by filling the LU cache."""
    flowwrapper.eval()

    # Switch to cache mode
    n_changed = 0
    for module in base_modules(flowwrapper):
        try:
            module.use_cache(True)
            n_changed += 1
        except AttributeError:
            pass
    print(f"Switched {n_changed} modules to cache mode")

    # Call the method to fill the cache
    if isinstance(dummy_input, tuple):
        getattr(flowwrapper, method)(*dummy_input)
    else:
        getattr(flowwrapper, method)(dummy_input)

    # Remove gradients from the LU caches layers
    n_changed = 0
    for module in base_modules(flowwrapper):
        try:
            module.cache.inverse = module.cache.inverse.data
            module.cache.logabsdet = module.cache.logabsdet.data
            n_changed += 1
        except AttributeError:
            pass
    print(f"Removed cache gradients from {n_changed} modules")

    return
