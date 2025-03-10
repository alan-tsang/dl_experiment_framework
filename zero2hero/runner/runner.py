"""

================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:38
@Data: 2025/02/02 22:00
@Data: 2025/02/28 01:00

================
"""

import argparse
import contextlib

import deepspeed
import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.cuda.amp import GradScaler

from .runner_base import RunnerBase
from ..callback import (CheckpointCallback, EpochSummaryCallBack, ProcessCallBack, WandbCallback)
from ..common.dl_util import get_batch_n, get_model_info
from ..common.logger import Logger
from ..common.registry import registry
from ..common.util import first_call_warning
from ..config.runner_config import RunnerConfig
from ..dist.init import is_main_process, main_process


class Runner(RunnerBase):
    def __init__(
            self,
            model: nn.Module,
            train_data_loader,
            valid_data_loader,
            test_data_loader,
            train_evaluator = None,
            valid_evaluator = None,
            test_evaluator = None,
            epochs = None,
            optimizer = None,
            callbacks = None,
            runner_config: RunnerConfig = None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.model = model

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.train_evaluator = train_evaluator
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator

        # NOTE: 启用deepspeed时，optimizer可以为空
        self.epochs = epochs
        self.optimizer = optimizer

        self.callbacks = callbacks if callbacks is not None else []
        self.runner_config = runner_config or RunnerConfig()

        self.scheduler = None

        self._assign_runtime_cfg()
        self._assign_runner_cfg(self.runner_config)
        self._assign_logger()
        self._apply_launch_strategy()
        self._setup_builtin_callbacks()


    def fit(self, *args, **kwargs):
        try:
            self.before_all()
            self.before_train()

            for epoch_i in range(self.epochs):
                registry.register("current_epoch", epoch_i)
                self.before_running_epoch()
                if registry.get("stop_training"):
                    break
                self.train()
                self.valid()
                self.test()
                self.after_running_epoch()
        except BaseException as e:
            self.logger.critical(f"训练时，发生异常：{e.__class__.__name__}")
            print(e)
            self.on_exception(e)
            raise
        finally:
            self.after_train()
            self.after_all()

    def train(self):
        current_epoch = registry.get("current_epoch")
        is_fp16 = registry.get("cfg.training.fp16")
        scaler = GradScaler() if is_fp16 else None

        for batch_i, data in enumerate(self.train_data_loader):
            registry.register("current_batch", batch_i)
            self.before_running_batch()
            self.train_data_loader.sampler.set_epoch(current_epoch)
            data = self._move_train_data_to_device(data)
            with self.maybe_autocast(is_fp16):
                model_output, _ = self.train_step(data)
            assert "loss" in model_output, "模型输出必须返回包含loss的Namespace"
            self.backward(scaler, model_output.loss)
            registry.register("metric.loss", model_output.loss)
            self.after_running_batch()


    def train_step(self, batch) -> tuple[any, argparse.Namespace]:
        """
        Must Returns:
            Loss in the model_output
        """
        if registry.get("current_step") is None:
            registry.register("current_step", 0)
        else:
            registry.register("current_step", registry.get("current_step") + 1)
        self.model.train()
        model_output = self.model(**batch)
        metric_val = None
        if self.train_evaluator:
            metric_val = self.train_evaluator.process(model_output, batch)

        return model_output, metric_val

    @staticmethod
    def maybe_autocast(enabled: bool = False):
        device_enable = registry.get("device") != torch.device("cpu")
        if enabled and device_enable:
            return torch.cuda.amp.autocast(dtype = torch.float16)
        else:
            return contextlib.nullcontext()


    def backward(self, scaler, loss):
        if registry.get("cfg.training.ds_config"):
            self.model.backward(loss)
            self.model.step()
        else:
            self.optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            current_epoch = registry.get("current_epoch")
            current_step = registry.get("current_step")

            self.scheduler.step(current_epoch, current_step) if self.scheduler \
                else None

    @torch.no_grad()
    def valid(self):
        """for dynamic cover, this func is Not merged with test"""
        if self.valid_data_loader is None:
            first_call_warning("valid", "未提供valid_data_loader，跳过valid")
            return

        if self.valid_evaluator is None:
            first_call_warning("valid_evaluator", "未提供valid_evaluator")

        current_epoch = registry.get("current_epoch")
        if current_epoch is None:
            should_valid = True
        elif (current_epoch + 1) % registry.get("cfg.training.valid_every_n_epochs") == 0:
            should_valid = True
        else:
            should_valid = False

        if not should_valid:
            return

        self.model.eval()
        for batch_i, data in enumerate(self.valid_data_loader):
            data = self._move_valid_data_to_device(data)
            model_output = self.valid_step(data)

        if self.valid_evaluator:
            eval_result = self.valid_evaluator.evaluate(self.valid_size)
            self.logger.info(f"Valid Summary: {eval_result}")
            self.wandb_log(eval_result)
            self.model.train()


    def valid_step(self, batch):
        model_output = self.model(**batch)
        self.valid_evaluator.process(model_output, batch)
        return model_output


    @torch.no_grad()
    def test(self):
        if self.test_data_loader is None:
            first_call_warning("test", "未提供test_data_loader，跳过test")
            return

        if self.test_evaluator is None:
            first_call_warning("test_evaluator", "未提供test_evaluator")

        current_epoch = registry.get("current_epoch")
        if current_epoch is None:
            should_test = True
        elif (current_epoch + 1) % registry.get("cfg.training.test_every_n_epochs") == 0:
            should_test = True
        else:
            should_test = False

        if not should_test:
            return

        self.model.eval()
        for batch_i, data in enumerate(self.test_data_loader):
            data = self._move_test_data_to_device(data)
            model_output = self.test_step(data)

        if self.test_evaluator:
            eval_result = self.test_evaluator.evaluate(self.test_size)
            self.logger.info(f"Test Summary: {eval_result}")
            self.wandb_log(eval_result)
            self.model.train()


    def test_step(self, batch):
        model_output = self.model(**batch)
        self.test_evaluator.process(model_output, batch)
        return model_output


    def before_all(self):
        super().before_all()
        self.logger.info(registry.get("device"))
        self.logger.info(registry.get("start_msg"))
        if registry.get("cfg.training.print_model"):
            # support distributed print, means print only once in the main process
            print(registry.get("model_info"))


    def after_all(self):
        super().after_all()
        self.logger.close_file_handler()
        if dist.is_initialized():
            dist.destroy_process_group()


    @main_process
    def wandb_log(self, log_dict):
        if registry.get("cfg.wandb.wandb_enable") and wandb.run is not None:
            wandb.log(log_dict)
            self.logger.info(f"wandb记录：{log_dict}")


    @staticmethod
    def _assign_runner_cfg(cfg: RunnerConfig):
        registry.register("cfg", cfg)

    def _assign_logger(self):
        # 有的话就用已创建的，忽略参数
        # 没有的话就会用这些参数创建
        self.logger = Logger.get_instance(
            "logger",
            **dict(registry.get("cfg.log")),
            # log_level = registry.get("cfg.log.log_level"),
            # to_file = registry.get("cfg.log.to_file"),
            # folder = registry.get("cfg.log.folder"),
            run_name = registry.get("cfg.run_name")
        )
    def _assign_runtime_cfg(self):
        if self.valid_data_loader:
            self.valid_size = len(self.valid_data_loader)
        if self.test_data_loader:
            self.test_size = len(self.test_data_loader)

    def _apply_launch_strategy(self):
        model_info = get_model_info(self.model)
        registry.register("model_info", model_info)

        # 单卡也使用分布式环境
        if torch.cuda.is_available():
            # if dist.is_available() and not dist.is_initialized():
            from ..dist.init import init_distributed_mode
            init_distributed_mode()
            device = torch.device(f'cuda:{dist.get_rank()}')
            self._prepare_dataloader()
        else:
            device = torch.device('cpu')
        registry.register("device", device)

        # 主进程信息注册
        registry.register("is_main_process", is_main_process())

        if device.type == "cpu":
            registry.register("start_msg", "使用CPU启动中...")
            self._set_scheduler()
            return

        # 初始化分布式模型和优化器
        self.model = self.model.to(device)
        ds_config = registry.get("cfg.training.ds_config")
        if ds_config:
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model = self.model,
                optimizer = self.optimizer if self.optimizer is not None else None,
                model_parameters = self.model.parameters(),
                config = ds_config,
                training_data = self.train_data_loader,
            )
            registry.register("start_msg", "使用deepspeed启动中...")
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids = [dist.get_rank()], output_device = dist.get_rank()
            )
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr = registry.get("lr"),
                )
            registry.register("start_msg", "使用torch.distributed启动中...")

        self._set_scheduler()

    def _set_scheduler(self):
        if not registry.get("cfg.training.ds_config"):
            scheduler_cls = registry.get_lr_scheduler_class(registry.get("cfg.scheduler.type"))
            if scheduler_cls:
                self.scheduler = scheduler_cls(
                    optimizer = self.optimizer,
                    max_epoch = self.epochs,
                    iters_per_epoch = int(get_batch_n(self.train_data_loader)),
                    **dict(registry.get("cfg.scheduler"))
                )
        else:
            self.logger.warning("启用了deepspeed，为自动配置warmup，"
                                "如果你想的话，请在配置文件中指定scheduler")

    def _prepare_dataloader(self):
        if dist.is_available() and dist.is_initialized():
            def wrap_dataloader(data_loader, shuffle = False):
                sampler = torch.utils.data.distributed.DistributedSampler(
                    data_loader.dataset,
                    num_replicas = dist.get_world_size(),
                    rank = dist.get_rank(),
                    shuffle = shuffle
                )
                return torch.utils.data.DataLoader(
                    data_loader.dataset,
                    batch_size = data_loader.batch_size,
                    sampler = sampler,
                    collate_fn = data_loader.collate_fn,
                    shuffle = False,
                    num_workers = data_loader.num_workers,
                    pin_memory = True,
                    drop_last = data_loader.drop_last
                )

            self.train_data_loader = wrap_dataloader(
                self.train_data_loader,
                shuffle = True
            )
            if self.valid_data_loader is not None:
                self.valid_data_loader = wrap_dataloader(self.valid_data_loader)
            if self.test_data_loader is not None:
                self.test_data_loader = wrap_dataloader(self.test_data_loader)


    def _setup_builtin_callbacks(self):
        epoch_summary_callback = EpochSummaryCallBack()
        progress_callback = ProcessCallBack(
            epochs = self.epochs,
            batchs = int(get_batch_n(self.train_data_loader)),
        )

        wandb_callback = WandbCallback() if registry.get("cfg.wandb.wandb_enable")\
                                        else None
        checkpoint_callback = CheckpointCallback(model = self.model) \
                                if registry.get("cfg.pt.pt_save") else None

        callbacks = [
            progress_callback,
            checkpoint_callback,
            wandb_callback,
            epoch_summary_callback,
        ]
        callbacks.extend(self.callbacks)
        self._register_callbacks(callbacks)

    def _move_train_data_to_device(self, data):
        device = registry.get("device")
        if isinstance(data, (list, tuple)):
            return [d.to(device) for d in data]
        elif isinstance(data, dict):
            return {k: v.to(device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            self.logger.warning(f"未知数据类型：{type(data)}")
            return data

    def _move_valid_data_to_device(self, data):
        return self._move_train_data_to_device(data)

    def _move_test_data_to_device(self, data):
        return self._move_train_data_to_device(data)
