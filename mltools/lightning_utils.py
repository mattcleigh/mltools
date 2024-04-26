"""General utilities for lightning modules."""

import logging
import math

from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .torch_utils import get_sched

logger = logging.getLogger(__name__)


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        logger.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        logger.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        logger.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def linear_warmup(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
) -> LambdaLR:
    """Return a scheduler with a linear warmup."""
    return LambdaLR(optimizer, lambda x: min(1, x / max(1, warmup_steps)))


def linear_warmup_exp_decay(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    half_life: int = 1000,
    min_factor: float = 1e-3,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a sqrt decay."""

    def fn(x):
        if x < warmup_steps:
            return x / max(1, warmup_steps)
        decay = -math.log(2) / half_life
        return max(math.exp(decay * (x - warmup_steps)), min_factor)

    return LambdaLR(optimizer, fn)


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    model: LightningModule,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    min_factor: float = 1e-3,
    warmup_ratio: float | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Replace the total_steps with the model trainer's actual max_steps
    total_steps = get_max_steps(model) or total_steps

    # Replace the wamup_steps with the ratio
    if warmup_ratio is not None:
        warmup_steps = int(warmup_ratio * total_steps)

    # Define the actual scheduler function
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_factor, lr)

    # The lambda scheduler is the easiest way to define a custom scheduler
    return LambdaLR(optimizer, fn)


def standard_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler.

    In favour of deprecating this in the future, as it is overly verbose.
    """
    # Finish initialising the partialy created methods
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))

    # Use mltools to initialise the scheduler
    # as we can sync the cycle length with the number of steps per epoch
    sched = get_sched(
        model.hparams.sched_config.mltools,
        opt,
        steps_per_epoch=len(model.trainer.datamodule.train_dataloader()),
        max_epochs=model.trainer.max_epochs,
        max_steps=model.trainer.max_steps,
    )

    # Return the dict for the lightning trainer
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, **model.hparams.sched_config.lightning},
    }


def simple_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler."""
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = {
        "scheduler": model.hparams.scheduler(optimizer=opt, model=model),
        "interval": "step",
    }
    return [opt], [scheduler]
