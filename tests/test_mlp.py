import pytest
import torch

from mltools.mlp import MLP


@pytest.mark.parametrize("num_layers_per_block", [1, 2])
@pytest.mark.parametrize("act_h", ["ReLU", "SiLU"])
@pytest.mark.parametrize("norm", [None, "LayerNorm"])
@pytest.mark.parametrize("do_bayesian", [False, True])
@pytest.mark.parametrize("init_zeros", [False, True])
def test_mlp_block(
    num_layers_per_block: int,
    act_h: str,
    norm: str,
    do_bayesian: bool,
    init_zeros: bool,
) -> None:
    x = torch.randn(10, 4)
    ctxt = torch.randn(10, 1)
    mlp = MLP(
        inpt_dim=4,
        outp_dim=3,
        hddn_dim=5,
        num_blocks=2,
        num_layers_per_block=num_layers_per_block,
        act_h=act_h,
        norm=norm,
        do_bayesian=do_bayesian,
        init_zeros=init_zeros,
    )
    out = mlp(x, ctxt=ctxt)
    assert out.shape == (10, 3)
