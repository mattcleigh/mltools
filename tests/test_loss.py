import pytest
import torch as T

from mltools import loss


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
def test_contrastive_loss(temperature):
    x1 = T.randn(10, 128)
    x2 = T.randn(10, 128)

    loss_value = loss.contrastive_loss(x1, x2, temperature)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()


def test_koleo_loss():
    x = T.randn(10, 128)
    eps = 1e-8
    normed = False

    loss_value = loss.koleo_loss(x, eps, normed)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()


def test_pressure_loss():
    x = T.randn(10, 128)
    normed = False

    loss_value = loss.pressure_loss(x, normed)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()


def test_champfer_loss():
    T.manual_seed(0)
    x = T.randn(2, 5, 5)
    y = T.randn(2, 10, 5)
    x_mask = T.randn(2, 5) > 0
    y_mask = T.randn(2, 10) > 0
    loss_value = loss.champfer_loss(x, x_mask, y, y_mask)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()


def test_kld_to_norm_loss():
    means = T.randn(10, 128)
    log_stds = T.randn(10, 128)

    loss_value = loss.kld_to_norm_loss(means, log_stds)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()


def test_my_bce_with_logit():
    outputs = T.randn(10, 1)
    targets = T.randint(0, 2, (10,))

    loss_fn = loss.MyBCEWithLogit()
    loss_value = loss_fn(outputs, targets)

    assert isinstance(loss_value, T.Tensor)
    assert loss_value.shape == ()
