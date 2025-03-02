import wandb
from omegaconf import OmegaConf


def load_cfg(path, is_runtime = None, ds_cfg = None, from_cli = True):
    base_cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_cli() if from_cli else {}
    ds_cfg = OmegaConf.load(ds_cfg) if ds_cfg else {}

    runtime_cfg = {}
    if is_runtime:
        """
        for wandb sweep, this can experiment with different hyperparameters
        """

        from ..dist.init import init_distributed_mode, is_main_process
        from ..dist.utils import barrier
        from ..dist.cmc import broadcast_object_list

        init_distributed_mode()
        if is_main_process():
            wandb.init(
                project = base_cfg.training.project_name,
                name = base_cfg.run_name
            )
            runtime_cfg = wandb.config
        broadcast_object_list([runtime_cfg], src = 0)
        barrier()

    cfg = OmegaConf.merge(base_cfg, runtime_cfg, ds_cfg, cli_cfg)
    """
    if is_runtime, wandb will not init in the runner again, and as well, 
    the cfg should be updated to wandb.config
    """
    if is_runtime:
        wandb.config.update(cfg)
    return cfg
