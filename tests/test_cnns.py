import pytest
import torch

from mltools.cnns import ConvNet, UNet


@pytest.mark.parametrize("num_blocks_per_layer", [1, 2])
@pytest.mark.parametrize("attn_resolution", [0, 8])
@pytest.mark.parametrize("channel_mult", [[1, 1], [1, 2, 2]])
@pytest.mark.parametrize("ctxt_dim", [0, 2])
@pytest.mark.parametrize("start_channels", [1, 2])
def test_cnn(
    num_blocks_per_layer: int,
    attn_resolution: int,
    channel_mult: list,
    ctxt_dim: int,
    start_channels: int,
) -> None:
    x = torch.randn(3, 3, 8, 8)
    ctxt = torch.randn(3, 2)
    cnn = ConvNet(
        inpt_size=(8, 8),
        outp_dim=2,
        inpt_channels=3,
        ctxt_dim=ctxt_dim,
        num_blocks_per_layer=num_blocks_per_layer,
        attn_resolution=attn_resolution,
        start_channels=start_channels,
        channel_mult=channel_mult,
    )
    out = cnn(x, ctxt=ctxt)
    assert out.shape == (3, 2)


@pytest.mark.parametrize("num_blocks_per_layer", [1, 2])
@pytest.mark.parametrize("attn_resolution", [0, 8])
@pytest.mark.parametrize("ctxt_dim", [0, 2])
@pytest.mark.parametrize("ctxt_img_channels", [0, 2])
@pytest.mark.parametrize("start_channels", [1, 2])
def test_unet(
    num_blocks_per_layer: int,
    attn_resolution: int,
    ctxt_dim: list,
    ctxt_img_channels: list,
    start_channels: list,
) -> None:
    x = torch.randn(2, 3, 8, 8)
    ctxt_img = torch.randn(2, 2, 8, 8)
    ctxt = torch.randn(2, 2)
    unet = UNet(
        inpt_size=x.shape[2:],
        inpt_channels=x.shape[1],
        ctxt_dim=ctxt_dim,
        num_blocks_per_layer=num_blocks_per_layer,
        attn_resolution=attn_resolution,
        start_channels=start_channels,
        ctxt_img_channels=ctxt_img_channels,
        channel_mult=[1, 1, 2],
    )
    out = unet(x, ctxt=ctxt, ctxt_img=ctxt_img)
    assert out.shape == (2, 3, 8, 8)
