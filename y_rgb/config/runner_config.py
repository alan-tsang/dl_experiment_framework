"""RunnerConfig: for aspect parameters of the whole runner based on pydantic
not use anymore, because it will
"""
from typing import Dict, List

from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    epochs: int = 10
    is_sweep: bool = False
    ds_config: str | None = None
    gradient_accumulation: int = 1
    resume_from: str | None = None
    activation_checkpoint: List[str] | None = None
    grad_clip: float = 1.0
    fp16: bool = False

    print_model: bool = True
    train_monitor: Dict[str, bool] = {"loss": False}
    valid_monitor: Dict[str, bool] = {"loss": False}
    test_monitor: Dict[str, bool] = {"loss": False}

    valid_every_n_epochs: int = 1
    test_every_n_epochs: int = 1
    progress_every_n_epochs: int = 1
    progress_every_n_batches: int = 1

class SchedulerConfig(BaseModel):
    type: str = "LinearWarmupCosineLRScheduler"
    min_lr: float = 3e-5
    max_lr: float = 3e-4
    warmup_rate: float = 0.1
    warmup_start_lr: float = 1e-5

class OptimizerConfig(BaseModel):
    lr: float = 3e-4
    weight_decay: float = 0.01

class WandbConfig(BaseModel):
    wandb_enable: bool = False
    wandb_project_name: str = "default"
    wandb_offline: bool = False
    wandb_dir: str = "./"
    wandb_tags: List[str] = []

class PtConfig(BaseModel):
    pt_save: bool = True
    pt_save_dir: str = "./checkpoints"
    pt_best_monitor: Dict[str, bool] = {"loss": False}
    pt_topk: int = 3
    pt_save_n_epochs: int = 1
    # pt_save_n_batches: int = 1000

class LogConfig(BaseModel):
    to_file: bool = True
    folder: str = "./logs"
    log_level: str = "INFO"

class RunnerConfig(BaseModel):
    log: LogConfig = Field(default_factory=LogConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    pt: PtConfig = Field(default_factory=PtConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    run_name: str = "experiment_01"
    run_description: str = ""
