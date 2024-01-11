"""General utilities for lightning modules."""

from pytorch_lightning import LightningModule

from .torch_utils import get_sched


def standard_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler."""

    # Finish initialising the partialy created methods
    opt = model.hparams.optimizer(params=model.parameters())

    # Use mltools to initialise the scheduler
    # as we can sync the cycle length with the number of steps per epoch
    sched = get_sched(
        model.hparams.sched_config.mltools,
        opt,
        steps_per_epoch=len(model.trainer.datamodule.train_dataloader()),
        max_epochs=model.trainer.max_epochs,
    )

    # Return the dict for the lightning trainer
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, **model.hparams.sched_config.lightning},
    }
