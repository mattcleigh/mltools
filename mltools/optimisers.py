"""Custom optimisers for PyTorch."""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import torch as T
from torch.optim import Optimizer

from .torch_utils import ParameterNoWD

log = logging.getLogger(__name__)


class AdamWS(T.optim.AdamW):
    """AdamW optimizer where weight decay is only applied to matrices.

    Currently the reccomended method for transformers.
    See: https://github.com/karpathy/nanoGPT
    """

    def __init__(self, params: Iterable | dict, weight_decay: float = 1e-2, **kwargs):
        params = list(params)  # Ensures that checking wont delete elements
        if isinstance(params[0], tuple):  # Make it a list for generalisation
            params = [x for _, x in params]

        def exempt(p):
            return (p.dim() < 2) or isinstance(p, ParameterNoWD)

        nodecay_params = [p for p in params if exempt(p)]
        decay_params = [p for p in params if not exempt(p)]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"AdamWS: Applying weight decay {weight_decay} to {num_decay_params} "
            f"parameters, and 0.0 to {num_nodecay_params} parameters."
        )
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, **kwargs)


class Lookahead(Optimizer):
    """The lookahead optimizer.

    https://arxiv.org/abs/1907.08610
    """

    def __init__(
        self,
        params: Iterable | None = None,
        inner_optimizer: partial | Optimizer | None = None,
        k=10,
        alpha=0.5,
        **opt_kwargs,
    ) -> None:
        # Default optimiser is Adam
        if inner_optimizer is None:
            inner_optimizer = partial(T.optim.Adam, lr=0.001)

        # Otherwise use our fully initialised optimizer
        elif isinstance(inner_optimizer, Optimizer):
            self.optimizer = inner_optimizer

        # Otherwise initialise using our parameters
        else:
            self.optimizer = inner_optimizer(params, **opt_kwargs)

        # Other class features
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    @property
    def defaults(self):
        return self.optimizer.defaults

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update(self, group):
        """Update the parameter groups."""
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = T.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure=None):
        """Update all parameter groups with gradient descent."""
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        """Return the date dict for saving and realoading."""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, T.Tensor) else k): v for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        """Load a saved state dictionary."""
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        """Add to the parameter group, needed by pytorch."""
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class Lion(Optimizer):
    """The lion algorithm https://arxiv.org/pdf/2302.06675.pdf.

    Implementation derived from:
    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialise the lion optimizer.

        Args:
            params: iterable of parameters to optimize or dicts defining groups
            lr: Learning rate, should be ~5 lower than Adam (default: 1e-4)
            betas: coefficients used for computing running averages
            weight_decay: weight decay coefficient (default: 0)
        """
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @T.no_grad()
    def step(self, closure: Callable | None = None) -> Any:
        """Perform a single optimization step.

        Args
        ----
          closure:
            A closure that reevaluates the model and returns the loss
        """
        # Define the loss using the closure function
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        # Iterate through the parameter groups
        for group in self.param_groups:
            for p in group["params"]:
                # Skip if the gradients are empty
                if p.grad is None:
                    continue

                # Perform weight decay step initially
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Get the gradients
                grad = p.grad

                # State (default-dict for each group)
                # initialization at start of training
                state = self.state[p]
                if len(state) == 0:
                    state["mt"] = T.zeros_like(p)

                # Load the state's averages and betas
                mt = state["mt"]
                beta1, beta2 = group["betas"]

                # Perform the weight update
                ct = mt * beta1 + grad * (1 - beta1)
                p.add_(T.sign(ct), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                mt.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
