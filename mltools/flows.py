"""Functions and classes used to define invertible transformations."""

from typing import Any, Callable, Literal

import normflows as nf
import numpy as np
import torch as T
import torch.nn as nn
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.nets.resnet import ResidualNet
from normflows.utils.masks import create_alternating_binary_mask
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE

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
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )
            if init_identity:
                nn.init.constant_(net.final_layer.weight, 0.0)
                nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
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


# class ContextSplineTransform(Transform):
#     """An invertible transform of a applied elementwise to a tensor."""

#     def __init__(
#         self,
#         inpt_dim: int,
#         ctxt_dim: int,
#         num_bins: int = 10,
#         init_identity: bool = False,
#         tails: str | None = None,
#         tail_bound: float = 1.0,
#         dense_config: Mapping | None = None,
#         min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
#         min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
#         min_derivative: float = DEFAULT_MIN_DERIVATIVE,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         inpt_dim : int
#             The input dimension.
#         ctxt_dim : int
#             The context dimension.
#         num_bins : int, optional
#             The number of bins, by default 10.
#         init_identity : bool, optional
#             Whether to initialize as identity, by default False.
#         tails : str or None, optional
#             The type of tails to use, either None or linear, by default None.
#         tail_bound : float, optional
#             The tail bound, by default 1.0.
#         dense_config : Mapping or None, optional
#             The dense network configuration, by default None.
#         min_bin_width : float, optional
#             The minimum bin width, by default DEFAULT_MIN_BIN_WIDTH.
#         min_bin_height : float, optional
#             The minimum bin height, by default DEFAULT_MIN_BIN_HEIGHT.
#         min_derivative : float, optional
#             The minimum derivative, by default DEFAULT_MIN_DERIVATIVE.
#         """

#         super().__init__()

#         self.num_bins = num_bins
#         self.min_bin_width = min_bin_width
#         self.min_bin_height = min_bin_height
#         self.min_derivative = min_derivative
#         self.tails = tails
#         self.tail_bound = tail_bound
#         self.init_identity = init_identity

#         self.net = MLP(
#             inpt_dim=ctxt_dim,
#             outp_dim=inpt_dim * self._output_dim_multiplier(),
#             **(dense_config or {})
#         )

#         # To be equally spaced with identity mapping
#         if init_identity:
#             # Cycle through the final dense block and pull out the last linear layer
#             for layer in self.net.output_block.block[::-1]:
#                 if isinstance(layer, nn.Linear):
#                     break

#             # Set the weights to be zero and change the bias
#             T.nn.init.constant_(layer.weight, 0.0)
#             T.nn.init.constant_(layer.bias, np.log(np.exp(1 - min_derivative) - 1))

#     def _output_dim_multiplier(self):
#         if self.tails == "linear":
#             return self.num_bins * 3 - 1
#         elif self.tails is None:
#             return self.num_bins * 3 + 1
#         else:
#             raise ValueError

#     def _process(
#         self, inputs: T.Tensor, context: T.Tensor | None = None, inverse: bool = False
#     ) -> tuple:
#         # Pass through the context extraction network
#         spline_params = self.net(context)

#         # Save some usefull shapes
#         batch_size, features = inputs.shape[:2]

#         # Reshape the outputs to be batch x dim x spline_params
#         transform_params = spline_params.view(
#             batch_size, features, self._output_dim_multiplier()
#         )

#         # Out of the parameters we get the widths, heights, and knot gradients
#         unnormalized_widths = transform_params[..., : self.num_bins]
#         unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
#         unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

#         # Select the appropriate function transform
#         if self.tails is None:
#             spline_fn = rational_quadratic_spline
#             spline_kwargs = {}
#         elif self.tails == "linear":
#             spline_fn = unconstrained_rational_quadratic_spline
#             spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
#         else:
#             raise ValueError

#         # Apply the spline transform
#         outputs, logabsdet = spline_fn(
#             inputs=inputs,
#             unnormalized_widths=unnormalized_widths,
#             unnormalized_heights=unnormalized_heights,
#             unnormalized_derivatives=unnormalized_derivatives,
#             inverse=inverse,
#             min_bin_width=self.min_bin_width,
#             min_bin_height=self.min_bin_height,
#             min_derivative=self.min_derivative,
#             **spline_kwargs
#         )

#         return outputs, sum_except_batch(logabsdet)

#     def forward(self, inputs: T.Tensor, context: T.Tensor) -> T.Tensor:
#         return self._process(inputs, context, inverse=False)

#     def inverse(self, inputs: T.Tensor, context: T.Tensor) -> T.Tensor:
#         return self._process(inputs, context, inverse=True)

# def stacked_ctxt_flow(xz_dim: int, ctxt_dim: int, nstacks: int, transform: partial):
#     """Return a composite transform given the config."""
#     return nf.flows.Composite([transform(xz_dim, ctxt_dim) for _ in range(nstacks)])
