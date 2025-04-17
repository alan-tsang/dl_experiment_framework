import wandb

from .base_callback import BaseCallBack
from ..common.registry import registry


@registry.register_callback("wandb")
class WandbCallback(BaseCallBack):
    def __init__(self):
        super().__init__()

    def on_register(self):
        super().on_register()
        if registry.get("is_main_process") and registry.get("cfg.wandb.wandb_enable") and wandb.run is None:
            wandb.init(
                project=registry.get("cfg.wandb.wandb_project_name", 'default'),
                name=registry.get("cfg.run_name", 'default'),
                notes = registry.get("cfg.run_description", "default description"),
                mode="offline" if registry.get("cfg.wandb.wandb_offline", False) else "online",
                config = dict(registry.get("cfg")),
                dir = registry.get("cfg.wandb.wandb_dir", './'),
                save_code = True,
                tags = registry.get("cfg.wandb.wandb_tags")
            )

    # def after_running_epoch(self):
    #     if wandb.run is not None:
    #         for key, value in registry.get("metric").items():
    #             wandb.log({key: value})

    def after_running_batch(self):
        if wandb.run is not None:
            metrics = registry.get("metric")
            if metrics is not None:
                for key, value in metrics.items():
                    wandb.log({key: value})
