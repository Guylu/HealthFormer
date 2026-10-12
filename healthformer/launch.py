"""Explicit training launch with optional tracking and local checkpoints."""
import logging

from lightning.fabric import Fabric
from lightning.fabric.plugins import MixedPrecision
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf
from .config import config as defaults, normalize_config, validate_methodology_config
from .trainer import Trainer
from .tracking import tracker


class ContinuousInputPrecision(MixedPrecision):
    """Keep raw measurements intact; autocast still controls backbone operations."""

    def convert_input(self, data):
        return data


def launch(settings):
    settings = OmegaConf.create({**defaults, **normalize_config(settings)})
    validate_methodology_config(settings)
    strategy = DDPStrategy(find_unused_parameters=settings.find_unused_parameters,
                           broadcast_buffers=False) if settings.devices > 1 else 'auto'
    precision_plugin = (ContinuousInputPrecision('bf16-mixed', device='cuda')
                        if settings.precision == 'bf16-mixed' else None)
    fabric = Fabric(accelerator=settings.accelerator, devices=settings.devices,
                    strategy=strategy, plugins=precision_plugin,
                    precision=None if precision_plugin else settings.precision)
    fabric.launch()
    fabric.seed_everything(settings.seed)
    try:
        tracker.start(settings)
        trainer = Trainer(settings, fabric)
        result = trainer.train()
    except BaseException:
        try:
            tracker.finish(exit_code=1)
        except Exception:
            logging.getLogger(__name__).exception('Tracking cleanup failed after a training error.')
        raise
    tracker.finish(exit_code=result or 0)
    return result or 0
