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
import warnings
from typing import List

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
            train_data_loader,
            valid_data_loader,
            test_data_loader,
            model: nn.Module,
            epochs,
            optimizer = None,
            callbacks = None,
            runner_config: RunnerConfig = None,
            metric_func = None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.model = model
        # NOTE: 启用deepspeed时，optimizer可以为空
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.epochs = epochs

        self.callbacks = callbacks if callbacks is not None else []
        self.scheduler = None
        self.runner_config = runner_config or RunnerConfig()
        self.compute_metric = metric_func if metric_func else self.compute_metric

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
                output, _ = self.train_step(data)
            assert "loss" in output, ("train_step必须返回包含loss的Namespace, "
                                      "其默认行为是直接返回模型输出"
                                      "因此要么模型forward返回含loss的Namespace(推荐)"
                                      "要么改写train_step方法返回含loss的Namespace")
            self.backward(scaler, output.loss)
            self.after_running_batch()


    def train_step(self, batch) -> tuple[any, argparse.Namespace]:
        """
        Must Returns:
            Loss
        """
        if registry.get("current_step") is None:
            registry.register("current_step", 0)
        else:
            registry.register("current_step", registry.get("current_step") + 1)
        self.model.train()
        model_output = self.model(**batch)

        metric_value = self.compute_metric(model_output, batch, "train")
        metric_value = self.filter_metric(metric_value, "train")

        self.register_metrics_values(metric_value)

        return model_output, metric_value


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
        if self.valid_data_loader:
            self.model.eval()
            current_epoch = registry.get("current_epoch")
            metrics = []
            if (current_epoch + 1) % registry.get("cfg.training.valid_every_n_epochs") == 0:
                for batch_i, data in enumerate(self.valid_data_loader):
                    data = self._move_valid_data_to_device(data)
                    metric_filter = self.valid_step(data)
                    metrics.append(metric_filter)
            self.logger.info(f"VALID: {self.average_batch_metric(metrics)}")
            self.model.train()
        else:
            first_call_warning("valid", "未提供valid_data_loader，跳过valid")


    def valid_step(self, batch):
        model_output = self.model(**batch)
        metric_value = self.compute_metric(model_output, batch, "valid")
        metric_filter = self.filter_metric(metric_value, "valid")
        self.register_metrics_values(metric_filter)
        return metric_filter


    @torch.no_grad()
    def test(self, *args, **kwargs):
        if not self.test_data_loader:
            first_call_warning("test", "未提供test_data_loader，跳过test")
            return

        self.model.eval()
        current_epoch = registry.get("current_epoch")
        metrics = []
        if (current_epoch + 1) % registry.get("cfg.training.test_every_n_epochs") == 0:
            for batch_i, data in enumerate(self.test_data_loader):
                data = self._move_test_data_to_device(data)
                metric_filter = self.test_step(data)
                metrics.append(metric_filter)
            self.logger.info(f"TEST: {self.average_batch_metric(metrics)}")
            self.model.train()


    def test_step(self, batch):
        model_output = self.model(**batch)
        metric_value = self.compute_metric(model_output, batch, "test")
        metric_filter = self.filter_metric(metric_value, "test")
        self.register_metrics_values(metric_filter)
        return metric_filter


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


    def compute_metric(self, model_output, batch, mode: str) -> argparse.Namespace:
        """
        Note: this is a placeholder, you should implement it
            or make model_output contain the metric value you want to monitor
        """
        first_call_warning("metric",
                           "compute_metric方法未实现，metric计算直接返回模型输出")
        return model_output

    @staticmethod
    def filter_metric(metrics: argparse.Namespace, mode: str):
        should_monitor = list(registry.get(f"cfg.training.{mode}_monitor").keys())
        monitored = list(vars(metrics).keys())
        if not all([m in monitored for m in should_monitor]):
            first_call_warning(
                f"filter_metric_{mode}",
                f"{mode} 时期，compute_metric 返回的Namespace中缺少监控的keys;"
                f"应该包含：{should_monitor}，实际包含：{monitored}"
            )
        filtered_metrics = argparse.Namespace()
        for key, value in vars(metrics).items():
            if key in should_monitor:
                setattr(filtered_metrics, key, value)
        return filtered_metrics

    # 要求metric里的值支持sum
    @staticmethod
    def average_batch_metric(metrics: List[argparse.Namespace]) -> argparse.Namespace:
        """
        average metrics in different data loader' s batches
        """
        if not metrics:
            return argparse.Namespace()
        average_metric = argparse.Namespace()
        for key in vars(metrics[0]).keys():
            values = [getattr(metric, key) for metric in metrics]
            average = sum(values) / len(values)
            setattr(average_metric, key, average)
        return average_metric


    @staticmethod
    def register_metrics_values(metrics: argparse.Namespace):
        for key, value in vars(metrics).items():
            registry.register(f"metric.{key}", value)


    @main_process
    def wandb_log(self, log_dict):
        if registry.get("wandb_enable") and wandb.run is not None:
            wandb.log(log_dict)
            self.logger.info(f"wandb记录：{log_dict}")

    @main_process
    def just_log(self, msg):
        self.logger.just_print(msg)

    @staticmethod
    def maybe_autocast(enabled: bool = False):
        device_enable = registry.get("device") != torch.device("cpu")
        if enabled and device_enable:
            return torch.cuda.amp.autocast(dtype = torch.float16)
        else:
            return contextlib.nullcontext()

    @staticmethod
    def _assign_runner_cfg(cfg: RunnerConfig):
        registry.register("cfg", RunnerConfig.model_validate(cfg))

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

