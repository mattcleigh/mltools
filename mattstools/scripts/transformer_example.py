import torch as T

from mattstools.transformers import TransformerVectorEncoder
from mattstools.modules import DenseNetwork

data_size = (10, 100, 3)

example_inpt = T.rand(data_size)
padd_mask = None

transformer = TransformerVectorEncoder(
    model_dim = 64,
    outp_dim = 64,
    num_sa_blocks=3,
    num_ca_blocks=1,
    mha_kwargs={"num_heads": 4, "drp": 0.1},
    trans_ff_kwargs = {
        "num_blocks": 1,
        "hddn_dim": 64,
        "nrm": "layer",
        "drp": 0.0,
    },
)

embedder = DenseNetwork(
    inpt_dim=3,
    outp_dim=64,
    hddn_dim=512,
    num_blocks=1,
    nrm="layer"
)

data = embedder(example_inpt)
data = transformer.forward(data)
data = data / T.linalg.norm(data, dim=-1, keepdims=True)


