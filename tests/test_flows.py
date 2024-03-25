import onnx
import onnxruntime as ort
import pytest
import torch as T
from torch import nn

from mltools.flows import (
    CoupledRationalQuadraticSpline,
    LULinear,
    PermuteEvenOdd,
    prepare_for_onnx,
    rqs_flow,
)


def test_permute_even_odd():
    flow = PermuteEvenOdd()
    z = T.randn(5, 10)
    z_permuted, _ = flow.forward(z)
    z_restored, _ = flow.inverse(z_permuted)
    z_twice, _ = flow.forward(z_permuted)
    assert T.allclose(z, z_restored)
    assert T.allclose(z, z_twice)


def test_lu_linear():
    flow = LULinear(num_channels=10)
    z = T.randn(5, 10)
    z_transformed, _ = flow.forward(z)
    z_restored, _ = flow.inverse(z_transformed)
    assert T.allclose(z, z_restored)


def test_coupled_rational_quadratic_spline():
    T.manual_seed(0)
    flow = CoupledRationalQuadraticSpline(
        num_input_channels=10,
        num_blocks=2,
        num_hidden_channels=5,
        num_context_channels=None,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation="ReLU",
        dropout_probability=0.0,
        reverse_mask=False,
        init_identity=True,
    )
    z = T.randn(5, 10)
    z_transformed, _ = flow.forward(z)
    z_restored, _ = flow.inverse(z_transformed)
    assert T.allclose(z, z_transformed)  # Identity init
    assert T.allclose(z, z_restored, atol=1e-3)  # Don't be too strict


flow_type = "coupling"
do_lu = True
ctxt_dim = 0


# Test rqs_flow function
@pytest.mark.parametrize("flow_type", ["coupling", "autoregressive"])
@pytest.mark.parametrize("do_lu", [True, False])
@pytest.mark.parametrize("ctxt_dim", [0, 5])
def test_rqs_flow(flow_type, do_lu, ctxt_dim):
    flow = rqs_flow(
        xz_dim=10,
        ctxt_dim=ctxt_dim,
        num_stacks=3,
        mlp_width=32,
        mlp_depth=2,
        mlp_act="LeakyReLU",
        tail_bound=4.0,
        dropout=0.0,
        num_bins=4,
        do_lu=do_lu,
        init_identity=True,
        do_norm=False,
        flow_type=flow_type,
    )
    x = T.randn(5, 10)
    if ctxt_dim:
        ctxt = T.randn(5, ctxt_dim)
        loss = flow.forward_kld(x, context=ctxt)
        z = flow(x, context=ctxt)
        xz = flow.inverse(z, context=ctxt)
    else:
        loss = flow.forward_kld(x)
        z = flow(x)
        xz = flow.inverse(z)
    assert T.allclose(x, xz, atol=1e-3)  # Don't be too strict
    assert isinstance(loss, T.Tensor)


@pytest.mark.filterwarnings("ignore")  # Lots of warnings with ONNX
@pytest.mark.parametrize("flow_type", ["coupling", "autoregressive"])
def test_onnx_export(tmp_path, flow_type):
    flow = rqs_flow(
        xz_dim=3,
        ctxt_dim=5,
        num_stacks=3,
        mlp_width=8,
        mlp_depth=2,
        mlp_act="LeakyReLU",
        tail_bound=4.0,
        dropout=0.0,
        num_bins=4,
        do_lu=False,
        init_identity=True,
        do_norm=False,
        flow_type=flow_type,
    )

    # Wrap the flow such that the sampling method is the forward method
    class WrappedFlow(nn.Module):
        def __init__(self, flow):
            super().__init__()
            self.flow = flow

        def forward(self, c: T.Tensor) -> T.Tensor:
            return self.flow.sample(1, c)

    # Wrap and sanitize the model for onnx
    model = WrappedFlow(flow)
    ctxt = T.randn(1, 5)
    prepare_for_onnx(model, ctxt, method="forward")

    # Export the model to onnx
    T.onnx.export(
        model=model,
        args=ctxt,
        f=tmp_path / "flow.onnx",
        export_params=True,
        verbose=True,
        input_names=["c"],
        output_names=["x"],
        opset_version=16,
    )

    # Only check that the onnx model runs
    # Can't check the output as the flow is stochastic
    onnx_model = onnx.load(tmp_path / "flow.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(tmp_path / "flow.onnx")
    ort_inputs = {"c": ctxt.detach().numpy()}
    ort_session.run(None, ort_inputs)
