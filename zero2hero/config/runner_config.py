from typing import Dict, List

from pydantic import BaseModel, Field


class LogConfig(BaseModel):
    log_level: str = "INFO"
    to_file: bool = True
    folder: str = "./logs"

class PtConfig(BaseModel):
    pt_save: bool = True
    pt_save_dir: str = "./checkpoints"
    pt_save_n_epochs: int = 1
    pt_best_monitor: str = "loss"
    pt_topk: int = 1
    monitor_greater_is_better: bool = False

class WandbConfig(BaseModel):
    wandb_enable: bool = False
    wandb_project_name: str = "default"
    wandb_offline: bool = True
    wandb_dir: str = "./"
    wandb_tags: List[str] = []

class TrainingConfig(BaseModel):
    epochs: int = 10
    print_model: bool = True
    fp16: bool = False
    train_monitor: Dict[str, bool] = {"loss": False}
    valid_monitor: Dict[str, bool] = {"loss": False}
    test_monitor: Dict[str, bool] = {"loss": False}

    valid_every_n_epochs: int = 1
    test_every_n_epochs: int = 1

    progress_every_n_epochs: int = 1
    progress_every_n_batches: int = 10
    ds_config: str | None = None

class SchedulerConfig(BaseModel):
    type: str = "LinearWarmupCosineLRScheduler"
    min_lr: float = 1e-5
    init_lr: float = 3e-4
    warmup_steps: int = 0
    warmup_start_lr: float = -1


class RunnerConfig(BaseModel):
    log: LogConfig = Field(default_factory=LogConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    pt: PtConfig = Field(default_factory=PtConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    run_name: str = "experiment_01"
    run_description: str = ""

