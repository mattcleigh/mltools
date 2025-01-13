import pytest
import torch as T
from torch import nn
from torch.nn.functional import layer_norm, scaled_dot_product_attention

from mltools.attention import merge_masks, my_scaled_dot_product_attention
from mltools.torch_utils import move_dev
from mltools.transformers import (
    Attention,
    ClassAttentionPooling,
    CrossAttentionEncoder,
    Residual,
    Transformer,
    pack,
    unpack,
)


def get_transformer_inputs(seed: int = 0) -> dict:
    batch_size = 2
    seq_len = 5
    dim = 4
    T.manual_seed(seed)
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


def test_residual() -> None:
    dim = 4
    x = T.randn(2, dim)
    fn = nn.Linear(dim, dim)
    layer = Residual(fn, dim=dim)
    out = layer(x)
    assert T.allclose(out, x)
    layer.gate.data += 1
    out = layer(x)
    assert T.allclose(out, x + fn(layer_norm(x, (dim,))))


def get_attn_models(dim, num_heads) -> tuple:
    attn = Attention(dim, num_heads=num_heads)
    torch_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    attn.attn_in.weight = torch_attn.in_proj_weight
    attn.attn_in.bias = torch_attn.in_proj_bias
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
    transformer = Transformer(
        dim=i["x"].shape[-1],
        ctxt_dim=i["ctxt"].shape[-1],
        num_layers=2,
        num_registers=2,
        layer_config={
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
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
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
        },
    )

    transformer = transformer.to("cuda")
    i = move_dev(i, "cuda")

    with T.autocast("cuda", enabled=True):
        out1 = transformer(**i)
        mask = transformer.get_combined_mask(i["mask"])
        out1 *= mask.unsqueeze(-1)
        transformer.pack_inputs = True
        out2 = transformer(**i)
        T.testing.assert_close(out1, out2, rtol=1e-3, atol=1e-3)


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
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
        },
        use_decoder=True,
    )
    out = transformer(**i)
    assert out.shape == (2, 7, 4)


def test_cae() -> None:
    i1 = get_transformer_inputs(0)
    i2 = get_transformer_inputs(1)
    x1 = i1["x"]
    x2 = i2["x"]
    x1_mask = i1["mask"]
    x2_mask = i2["mask"]
    ctxt = i1["ctxt"]
    transformer = CrossAttentionEncoder(
        dim=x1.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=2,
        dec_config={
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
        },
        enc_config={
            "ff_config": {"mult": 1},
            "attn_config": {"num_heads": 2},
        },
    )
    out = transformer(x1, x2, x1_mask, x2_mask, ctxt)
    assert len(out) == 2
    assert out[0].shape == (2, 5, 4)
    assert out[1].shape == (2, 5, 4)


def test_flash_cae() -> None:
    if not T.cuda.is_available():
        pytest.skip("CUDA not available")
    i1 = get_transformer_inputs(0)
    i2 = get_transformer_inputs(1)
    i1 = move_dev(i1, "cuda")
    i2 = move_dev(i2, "cuda")
    x1 = i1["x"]
    x2 = i2["x"]
    x1_mask = i1["mask"]
    x2_mask = i2["mask"]
    ctxt = i1["ctxt"]
    transformer = CrossAttentionEncoder(
        dim=x1.shape[-1],
        ctxt_dim=ctxt.shape[-1],
        num_layers=1,
        dec_config={
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
        },
        enc_config={
            "ff_config": {"mult": 1},
            "attn_config": {"num_heads": 2},
        },
    )
    transformer = transformer.to("cuda")

    with T.autocast("cuda", enabled=True):
        x1out1, x2out1 = transformer(x1, x2, x1_mask, x2_mask, ctxt)
        x1out1 *= x1_mask.unsqueeze(-1)
        x2out1 *= x2_mask.unsqueeze(-1)
        transformer.pack_inputs = True
        x1out2, x2out2 = transformer(x1, x2, x1_mask, x2_mask, ctxt)
        T.testing.assert_close(x1out1[0], x1out2[0], rtol=1e-3, atol=1e-3)
        T.testing.assert_close(x2out1[1], x2out2[1], rtol=1e-3, atol=1e-3)


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
            "ff_config": {"mult": 2},
            "attn_config": {"num_heads": 2},
        },
    )
    out = cap(x, mask=mask, ctxt=ctxt)
    assert out.shape == (2, 4)


if __name__ == "__main__":
    test_merge_masks()
    test_my_scaled_dot_product_attention()
    test_pack()
    test_unpack()
    test_residual()
    test_self_attention()
    test_cross_attention()
    test_transformer_encoder()
    test_flash_transformer()
    test_transformer_decoder()
    test_cae()
    test_flash_cae()
    test_classattention_pooling()
    print("All tests passed!")
