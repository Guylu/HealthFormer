"""Optional aggregate logging; never modifies the W&B module."""
import numbers
import torch

class Tracker:
    def __init__(self):
        self.run = None

    @staticmethod
    def is_rank_zero():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def start(self, settings):
        if not self.is_rank_zero() or not settings.allow_wandb:
            return
        if not settings.wandb_entity or not settings.wandb_project:
            raise ValueError('Online tracking requires wandb_entity and wandb_project.')
        import wandb
        # These capture controls are checked against the documented SDK pin.
        capture_settings = wandb.Settings(
            console='off', disable_code=True, disable_git=True,
            x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
            x_save_requirements=False,
        )
        self.run = wandb.init(
            entity=settings.wandb_entity, project=settings.wandb_project,
            mode='online', save_code=False, reinit='create_new', settings=capture_settings,
        )

    def log(self, metrics, **kwargs):
        if self.run is None or not self.is_rank_zero():
            return
        scalars = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().item()
            if isinstance(value, numbers.Number):
                scalars[key] = value
        if scalars:
            self.run.log(scalars, **kwargs)


    def finish(self, **kwargs):
        if self.run is not None and self.is_rank_zero():
            try:
                self.run.finish(**kwargs)
            finally:
                self.run = None

    # Participant plots and tables are intentionally not exported by aggregate logging.

tracker = Tracker()
