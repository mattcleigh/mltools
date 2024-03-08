import itertools
from functools import partial

import numpy as np
import torch as T
from flash_attn.utils.benchmark import benchmark_backward, benchmark_forward
from tqdm import tqdm

from mltools.transformers import Transformer

T.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
)
T.set_float32_matmul_precision("medium")
T.set_autocast_enabled(True)

# Define a simple transformer
model1 = Transformer(
    dim=256,
    num_layers=8,
    layer_config={
        "do_self_attn": True,
        "layerscale_init": None,
    },
)

model2 = Transformer(
    dim=256,
    num_layers=8,
    do_packed=True,
    layer_config={
        "do_self_attn": True,
        "layerscale_init": None,
    },
)


def full_nested(model, x, kv_mask) -> T.Tensor:
    xnt = T._nested_tensor_from_mask(x, kv_mask, mask_check=False)
    ynt = model(xnt).to_padded_tensor(0, x.size())
    # y = T.zeros_like(x)
    # y[mask] = ynt[T.all(ynt != 0, dim=-1)]
    return ynt


def full_standard(model, x, kv_mask) -> T.Tensor:
    y = model(x, kv_mask=kv_mask) * kv_mask.unsqueeze(-1)
    return y


# Sync the parameters
model2.load_state_dict(model1.state_dict())

##################

batch_size = 128
seq_len = 128
feature_dim = 256

##################

model1.to("cuda")
model2.to("cuda")

# Define the inputs
x = T.randn((batch_size, seq_len, feature_dim), device="cuda")
mask = T.ones((batch_size, seq_len), dtype=T.bool, device="cuda")

# Define the functions
fns = {
    "Padded": partial(full_standard, model1),
    "VarLen": partial(full_standard, model2),
    "Nested": partial(full_nested, model1),
}

# Print the differences
outputs = [fn(x, kv_mask=mask) for fn in fns.values()]
diffs = [(a - b).abs().max().item() for a, b, in itertools.combinations(outputs, 2)]
print(diffs)

# Warmup the GPU
for _ in range(10):
    outputs = [fn(x, kv_mask=mask) for fn in fns.values()]

fwd_times = {k: [] for k in fns.keys()}
bwd_times = {k: [] for k in fns.keys()}
padding_fractions = np.linspace(0, 0.99, 10)

# For a range of padding fractions
for pd in tqdm(padding_fractions):
    mask = T.arange(seq_len, device="cuda") > int(seq_len * pd)
    mask = mask.unsqueeze(0).expand(batch_size, -1)

    for k, fn in fns.items():
        tf = benchmark_forward(fn, x, kv_mask=mask, amp=True, verbose=False)[1].mean
        try:
            tb = benchmark_backward(fn, x, kv_mask=mask, amp=True, verbose=False)[
                1
            ].mean
        except RuntimeError:
            tb = np.nan
        fwd_times[k].append(tf * 1000)
        bwd_times[k].append(tb * 1000)

# Plot the results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for k in fns.keys():
    ax[0].plot(padding_fractions, fwd_times[k], label=k)
    ax[1].plot(padding_fractions, bwd_times[k], label=k)
ax[0].set_ylabel("Forward Time (ms)")
ax[1].set_ylabel("Backward Time (ms)")
ax[1].set_xlabel("Padding Fraction")
ax[0].legend()
plt.tight_layout()
plt.savefig("benchmark.png")
plt.close()
