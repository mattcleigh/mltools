import pytest
import torch as T
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from mltools.torch_utils import move_dev
from mltools.transformers import (
    Attention,
    ClassAttentionPooling,
    CrossAttentionEncoder,
    LayerScale,
    PreNormScaledResidual,
    Transformer,
    TransformerVectorEncoder,
    merge_masks,
    my_scaled_dot_product_attention,
    pack,
    unpack,
)


def get_transformer_inputs() -> dict:
    batch_size = 2
    seq_len = 5
    dim = 4
    T.manual_seed(0)
    return {
        "x": T.randn(batch_size, seq_len, dim),
        "mask": T.randn(batch_size, seq_len) > 0,
        "kv": T.randn(batch_size, seq_len + 1, dim),
        "kv_mask": T.randn(batch_size, seq_len + 1) > 0,
        "ctxt": T.randn(batch_size, dim),
        "attn_mask": T.randn(batch_size, seq_len, seq_len + 1) > 0,
        "attn_bias": T.randn(batch_size, seq_len, seq_len + 1, 2),
    }


def test_merge_masks() -> None:
    i = get_transformer_inputs()
    merge_masks(i["kv_mask"], None, None, i["x"])


def test_my_scaled_dot_product_attention():
    k = T.randn(2, 3, 4)
    q = T.randn(2, 3, 4)
    v = T.randn(2, 3, 4)
    attn = my_scaled_dot_product_attention(q, k, v)
    expected_attn = scaled_dot_product_attention(q, k, v)
    assert T.allclose(attn, expected_attn)


def test_pack():
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    txt = i["ctxt"]
    px, pctxt, culens, maxlen = pack(x, mask, txt)
    assert T.allclose(x[mask], px)
    assert pctxt.shape == px.shape
    assert maxlen == mask.sum(-1).max()
    assert culens[-1] == mask.sum()


def test_unpack():
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    txt = i["ctxt"]
    px = pack(x, mask, txt)[0]
    upx = unpack(px, mask)
    assert T.allclose(x * mask.unsqueeze(-1), upx)


def test_layerscale():
    dim = 4
    init_value = 1e-3
    layer_scale = LayerScale(dim, init_value)
    x = T.randn(2, dim)
    scaled_x = layer_scale(x)
    assert T.allclose(scaled_x, x * init_value)


def test_prenomscaledresidual() -> None:
    dim = 4
    ls_init = 1e-3
    x = T.randn(2, dim)
    fn = nn.Linear(dim, dim)
    layer = PreNormScaledResidual(fn, ls_init, dim)
    out = layer(x)
    expected = x + ls_init * fn(layer.norm(x))
    assert T.allclose(out, expected)


def get_attn_models(dim, num_heads) -> tuple:
    attn = Attention(dim, num_heads=num_heads)
    torch_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    attn.attn_in_w = torch_attn.in_proj_weight
    attn.attn_in_b = torch_attn.in_proj_bias
    attn.attn_out.weight = torch_attn.out_proj.weight
    attn.attn_out.bias = torch_attn.out_proj.bias
    return attn, torch_attn


def test_self_attention() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    dim = x.shape[-1]
    attn, torch_attn = get_attn_models(dim, 2)
    output = attn(x, mask=mask)
    torch_output, _ = torch_attn(x, x, x, key_padding_mask=~mask)
    assert T.allclose(output, torch_output)


def test_cross_attention() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    kv = i["kv"]
    kv_mask = i["kv_mask"]
    dim = x.shape[-1]
    attn, torch_attn = get_attn_models(dim, 2)
    output = attn(x, kv=kv, mask=mask, kv_mask=kv_mask)
    torch_output, _ = torch_attn(x, kv, kv, key_padding_mask=~kv_mask)
    assert T.allclose(output, torch_output)


def test_transformer_encoder() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    ctxt = i["ctxt"]
    transformer = Transformer(
        dim=x.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=2,
        num_registers=2,
        layer_config={
            "ff_mult": 2,
            "num_heads": 2,
        },
    )
    out = transformer(**i)
    assert out.shape == (2, 7, 4)


def test_flash_transformer() -> None:
    if not T.cuda.is_available():
        pytest.skip("CUDA not available")
    i = get_transformer_inputs()
    del i["attn_bias"]
    del i["attn_mask"]
    transformer = Transformer(
        dim=i["x"].shape[-1],
        ctxt_dim=i["ctxt"].shape[-1],
        num_layers=2,
        num_registers=2,
        layer_config={
            "ff_mult": 2,
            "num_heads": 2,
        },
    )

    transformer = transformer.to("cuda")
    i = move_dev(i, "cuda")

    with T.autocast("cuda", enabled=True):
        out1 = transformer(**i)
        mask = transformer.get_combined_mask(i["mask"])
        out1 *= mask.unsqueeze(-1)
        transformer.do_packed = True
        out2 = transformer(**i)
        T.testing.assert_close(out1, out2)


def test_transformer_decoder() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    _, S, _ = x.shape
    ctxt = i["ctxt"]
    i["attn_mask"] = i["attn_mask"][:, :S, :S]
    i["attn_bias"] = i["attn_bias"][:, :S, :S]
    transformer = Transformer(
        dim=x.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=2,
        num_registers=2,
        layer_config={
            "ff_mult": 2,
            "num_heads": 2,
        },
        use_decoder=True,
    )
    out = transformer(**i)
    assert out.shape == (2, 7, 4)


def test_cae() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    ctxt = i["ctxt"]
    transformer = CrossAttentionEncoder(
        dim=x.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=2,
        num_registers=2,
        layer_config={
            "ff_mult": 2,
            "num_heads": 2,
        },
    )
    out = transformer(x, mask=mask, ctxt=ctxt)
    assert out.shape == (2, 7, 4)


def test_classattention_pooling() -> None:
    i = get_transformer_inputs()
    x = i["x"]
    mask = i["mask"]
    ctxt = i["ctxt"]
    cap = ClassAttentionPooling(
        dim=x.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=2,
        layer_config={
            "ff_mult": 2,
            "num_heads": 2,
        },
    )
    out = cap(x, mask=mask, ctxt=ctxt)
    assert out.shape == (2, 4)


def test_trasformervector_encoder() -> None:
    i = get_transformer_inputs()
    outp_dim = 2
    x = i["x"]
    mask = i["mask"]
    ctxt = i["ctxt"]
    transformer = TransformerVectorEncoder(
        inpt_dim=x.shape[-1],
        outp_dim=outp_dim,
        ctxt_dim=ctxt.shape[-1],
        encoder_config={
            "num_layers": 1,
            "layer_config": {
                "ff_mult": 2,
                "num_heads": 2,
            },
        },
        classattention_config={
            "num_layers": 1,
            "layer_config": {
                "ff_mult": 2,
                "num_heads": 2,
            },
        },
    )
    out = transformer(x, mask=mask, ctxt=ctxt)
    assert out.shape == (2, outp_dim)
