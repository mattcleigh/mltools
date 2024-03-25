import sys

sys.path.append("/home/users/l/leighm/DiffGAE/mltools")


from functools import partial

import torch as T
from torch import nn

from mltools.optimisers import Lion, Lookahead


def test_lookahead():
    x = T.randn(2, 10)
    model = nn.Linear(10, 5)
    optimizer = partial(T.optim.SGD, lr=0.1)
    lookahead_optimizer = Lookahead(model.parameters(), optimizer)

    # Perform a single optimization step
    loss = model(x).sum()
    lookahead_optimizer.zero_grad()
    loss.backward()
    lookahead_optimizer.step()

    # Assert that the parameters have been updated
    lookahead_optimizer.zero_grad()
    for param in model.parameters():
        assert param.grad is None  # Gradients should be cleared after the step
        assert T.all(param != 0)  # Parameters should have been updated


def test_lion():
    x = T.randn(2, 10)
    model = nn.Linear(10, 5)
    optimizer = Lion(model.parameters(), lr=0.1)

    # Perform a single optimization step
    loss = model(x).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Assert that the parameters have been updated
    optimizer.zero_grad()
    for param in model.parameters():
        assert param.grad is None  # Gradients should be cleared after the step
        assert T.all(param != 0)  # Parameters should have been updated
