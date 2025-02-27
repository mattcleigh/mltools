"""General utilities for lightning modules."""

import logging
import math
import re
from pathlib import Path
from typing import Any

import h5py
import torch as T
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .torch_utils import (
    get_activations,
    get_sched,
    get_submodules,
    gradient_norm,
    to_numpy,
)

log = logging.getLogger(__name__)


def save_predictions(
    model: LightningModule,
    datamodule: LightningModule,
    trainer: Trainer,
    file_path: str,
    ckpt_path: str | None = None,
) -> None:
    log.info("Running inference on test set")
    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Combining predictions across dataset")
    keys = list(outputs[0].keys())
    score_dict = {k: T.vstack([o[k] for o in outputs]) for k in keys}
    score_dict = to_numpy(score_dict)

    log.info("Saving outputs")
    output_dir = Path(file_path, "outputs")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "test_set.h5", mode="w") as file:
        for k in keys:
            file.create_dataset(k, data=score_dict[k])


class ActivationMonitor(Callback):
    """Callback to monitor the activations magnitudes at select layers in a model."""

    def __init__(
        self,
        logging_interval: int = 100,
        layer_types: list | None = None,
        layer_regex: list | None = None,
        param_regex: list | None = None,
    ) -> None:
        self.logging_interval = logging_interval
        self.layer_types = layer_types
        self.layer_regex = layer_regex
        self.param_regex = param_regex
        self.act_dict = {}

    def on_train_batch_start(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        """Add hooks to the model to monitor the layer activations."""
        if batch_idx % self.logging_interval != 0:
            return
        pl_module.hooks = get_activations(
            pl_module,
            self.act_dict,
            types=self.layer_types,
            regex=self.layer_regex,
        )

    def on_train_batch_end(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        """Remove the hooks after the batch and log the activations."""
        if batch_idx % self.logging_interval != 0:
            return
        for key, value in self.act_dict.items():
            pl_module.log(f"activations/{key}", value)
        self.act_dict = {}
        for hook in pl_module.hooks:
            hook.remove()
        for n, p in pl_module.named_parameters():
            if any(re.match(r, n) for r in self.param_regex):
                self.log(f"param/{n}", p.detach().abs().mean())


class LogGradNorm(Callback):
    """Logs the gradient norm."""

    def __init__(self, logging_interval: int = 1, depth: int = 0):
        self.logging_interval = logging_interval
        self.depth = depth

    def on_before_optimizer_step(
        self, _trainer: Trainer, pl_module: LightningModule, _optimizer: Optimizer
    ):
        if pl_module.global_step % self.logging_interval == 0:
            sub_modules = get_submodules(pl_module, self.depth)
            for subname, module in sub_modules:
                grad = gradient_norm(module)
                if grad > 0:
                    self.log("grad/" + subname, gradient_norm(module))


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        log.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        log.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        log.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def linear_warmup(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    init_factor: float = 1e-2,
) -> LambdaLR:
    """Return a scheduler with a linear warmup."""

    def fn(x: int) -> float:
        return min(1, init_factor + x * (1 - init_factor) / max(1, warmup_steps))

    return LambdaLR(optimizer, fn)


def linear_warmup_exp_decay(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    half_life: int = 1000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-1,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a sqrt decay."""

    def fn(x: int) -> float:
        if x < warmup_steps:
            return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
        decay = -math.log(2) / half_life
        return max(math.exp(decay * (x - warmup_steps)), final_factor)

    return LambdaLR(optimizer, fn)


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 5e-2,
    init_factor: float = 1e-5,
    model: LightningModule | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Attempt to get the max steps from the model trainer
    if total_steps == -1 and model is not None:
        total_steps = get_max_steps(model)

    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


def one_cycle(
    model: LightningModule,
    optimizer: Optimizer,
    total_steps: int = 1000,
    **kwargs,
) -> OneCycleLR:
    """Get the learning rate scheduler."""
    total_steps = get_max_steps(model) or total_steps
    return OneCycleLR(
        optimizer,
        **kwargs,
        total_steps=total_steps,
        max_lr=optimizer.param_groups[0]["lr"],
    )


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


def multi_optim_sched(model: LightningModule, modules: list) -> dict:
    """Configure multiple optimizers and learning rate sheduler."""
    opt_params = model.hparams.optimizer
    sched_params = model.hparams.scheduler
    assert len(modules) == len(opt_params) == len(sched_params)
    opts = []
    scheds = []
    for module, opt_param, sched_param in zip(
        modules, opt_params, sched_params, strict=False
    ):
        opt = opt_param(filter(lambda p: p.requires_grad, module.parameters()))
        sched = {
            "scheduler": sched_param(optimizer=opt, model=model),
            "interval": "step",
        }
        opts.append(opt)
        scheds.append(sched)
    return opts, scheds


class CatchupToLR(BaseFinetuning):
    """Finetuning callback that has the backbone linearly catch up to the model."""

    def __init__(self, unfreeze_at_step: int = 1000, catchup_steps: int = 1000) -> None:
        super().__init__()
        self.unfreeze_at_step = unfreeze_at_step
        self.catchup_steps = catchup_steps
        self.steps_done = 0
        self.backbone_lr = 0
        self.frozen = True

    def on_fit_start(self, trainer, pl_module) -> None:
        """Raises
        MisconfigurationException:
            If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, nn.Module):
            return super().on_fit_start(trainer, pl_module)
        raise ValueError("The LightningModule should have a `backbone` attribute")

    def freeze_before_training(self, pl_module) -> None:
        """Prevent the backbone from training initially.

        Called before `configure_optimizers` by the parent's setup() method.
        """
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module, step, optimizer) -> None:
        """Used to update the learning rate of the backbone."""
        # Still in the frozen stage - nothing to do
        if step < self.unfreeze_at_step or self.unfreeze_at_step == -1:
            return

        # Time to thaw, initial learning rate is negligable
        if self.frozen:
            log.info("Unfreezing the backbone!")
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                1e-8,  # Start with no learning rate
            )
            self.frozen = False

            # Add the group keys that are missing else many schedulers will fail
            original = optimizer.param_groups[0]
            for key in original:
                if key not in optimizer.param_groups[-1]:
                    optimizer.param_groups[-1][key] = original[key]

        # Linearly ramp up
        if self.steps_done < self.catchup_steps:
            model_lr = optimizer.param_groups[0]["lr"]
            delta = model_lr - self.backbone_lr
            steps_left = self.catchup_steps - self.steps_done
            increment = delta / steps_left
            self.backbone_lr = min(model_lr, self.backbone_lr + increment)
            optimizer.param_groups[-1]["lr"] = self.backbone_lr
            self.steps_done += 1

        # Fully caught up, ensure learning rates are always synced
        else:
            optimizer.param_groups[-1]["lr"] = optimizer.param_groups[0]["lr"]

    def on_train_batch_end(
        self, trainer, pl_module, _outputs, _batch, _batch_idx
    ) -> None:
        """Update the earning rate of the group after each batch pass.

        Same function as the partents old 'on_train_epoch_start' and also using global
        step instead of epoch.
        """
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            num_param_groups = len(optimizer.param_groups)
            self.finetune_function(pl_module, trainer.global_step, optimizer)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: ARG002
        """Called when the epoch begins.

        Overloaded from parent to prevent anything happening
        """
        return
