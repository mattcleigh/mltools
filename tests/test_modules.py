import pytest
import torch as T

from mltools.modules import GRL, CosineEncodingLayer, IterativeNormLayer


def test_GRL_forward():
    grl = GRL(alpha=0.5)
    x = T.tensor([1, 2, 3], dtype=T.float32)
    output = grl(x)
    assert T.allclose(output, x)


def test_GRL_backward():
    grl = GRL(alpha=0.5)
    x = T.tensor([1, 2, 3], dtype=T.float32, requires_grad=True)
    output = grl(x)
    output.backward(T.ones_like(output))
    assert T.allclose(x.grad, -0.5 * T.ones_like(x))


def test_IterativeNormLayer_fit():
    T.manual_seed(0)
    inpt = T.randn(10, 4, 3)
    mask = T.randn(10, 4) > 0
    norm_layer = IterativeNormLayer(inpt_dim=3)
    norm_layer.fit(inpt, mask=mask)
    varns, means = T.var_mean(inpt[mask], dim=0, keepdim=True, unbiased=False)
    assert T.allclose(norm_layer.means, means)
    assert T.allclose(norm_layer.vars, varns)
    assert norm_layer.n == mask.sum()


def test_IterativeNormLayer_forward():
    T.manual_seed(0)
    inpt = T.randn(10, 3)
    norm_layer = IterativeNormLayer(inpt_dim=3)
    norm_layer.training = True
    for i in range(0, 10, 2):
        norm_layer(inpt[i : i + 2])
    means = inpt.mean(dim=0, keepdim=True)
    varns = inpt.var(dim=0, keepdim=True, unbiased=False)
    assert T.allclose(norm_layer.means, means)
    assert T.allclose(norm_layer.vars, varns)
    assert norm_layer.n == len(inpt)


def test_IterativeNormLayer_forward_reverse():
    T.manual_seed(0)
    inpt = T.randn(10, 4, 3)
    mask = T.randn(10, 4) > 0
    norm_layer = IterativeNormLayer(inpt_dim=3)
    norm_layer.fit(inpt, mask=mask)
    output = norm_layer(inpt, mask=mask)
    out_v, out_m = T.var_mean(output[mask], dim=0, keepdim=True, unbiased=False)
    rec = norm_layer.reverse(output, mask=mask)
    assert T.allclose(out_v, T.ones_like(out_v), atol=1e-5)
    assert T.allclose(out_m, T.zeros_like(out_m), atol=1e-5)
    assert T.allclose(rec, inpt, atol=1e-5)


@pytest.mark.parametrize("scheme", ["exp", "pow", "linear"])
@pytest.mark.parametrize("do_sin", [True, False])
def test_cosine_encoding_layer(scheme, do_sin):
    layer = CosineEncodingLayer(
        inpt_dim=4,
        encoding_dim=8,
        scheme=scheme,
        do_sin=do_sin,
    )
    inpt = T.rand(10, 4)
    output = layer(inpt)
    assert output.shape == (10, 4 * 8)
