"""
A runner for training, validation and testing.

maybe should be overridden in diff proj:
1. _move_train_data_to_device, _move_valid_data_to_device, _move_test_data_to_device
2. valid_step, test_step

don't recommend to override, unless you know what you are doing:
1. train_step
2. _apply_launch_strategy

try to use your own callbacks, then you can do something in different lifecycle of runner

"""
import contextlib
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import omegaconf
from torch.cuda.amp import GradScaler

from .runner_base import RunnerBase
from .. import BaseModel, Evaluator
from ..callback import (CheckpointCallback, EpochSummaryCallBack, ProcessCallBack, WandbCallback)
from ..callback.base_callback import BaseCallBack
from ..common.dl_util import get_batch_n, get_model_info
from ..common.logger import Logger
from ..common.registry import registry
from ..common.util import first_call_warning, better_dict_4_print

from ..dist.init import is_main_process, main_process
from ..scheduler import LinearWarmupCosineLRScheduler


class Runner(RunnerBase):
    """
    AI 模型训练核心执行器

    ██████╗ ██╗███╗   ██╗███╗   ██╗███████╗██████╗
    ██╔══██╗██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
    ██████╔╝██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
    ██╔══██╗██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
    ██║  ██║██║██║ ╚████║██║ ╚████║███████╗██║  ██║
    ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝

    典型用法：
    custom your yaml config file based on the cfg in example,
    >>> runner = Runner(
            model=your_model,
            train_data_loader=train_loader,
            ...
        )
    >>> runner.fit()

    扩展点：
    - 覆盖 _move_*_data_to_device 方法实现自定义设备迁移
    - 继承 TrainingSteps 类实现自定义训练逻辑
    - 通过 callback_manager 注册自定义回调
    """
    def __init__(
            self,
            model: BaseModel,
            train_data_loader: torch.utils.data.DataLoader,
            valid_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
            train_evaluator: Evaluator = None,
            valid_evaluator: Evaluator = None,
            test_evaluator: Evaluator = None,
            epochs: int = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            callbacks: List[BaseCallBack] = None,
            runner_config: Union[dict, omegaconf.DictConfig] = None,
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

        self.epochs = epochs
        # NOTE: 启用deepspeed时，optimizer可以为空
        self.optimizer = optimizer

        self.callbacks = callbacks if callbacks is not None else []
        self.runner_config = runner_config

        self.scheduler = None

        self._assign_runner_cfg(self.runner_config)
        self._assign_logger()
        self._assign_runtime_parameter()
        self._apply_launch_strategy()
        self._setup_builtin_callbacks()


    def fit(self, *args, **kwargs):
        start_epoch = 0
        start_batch = 0
        if self.resume_from:
            start_epoch, start_batch = self.checkpoint_callback.load_checkpoint(self.resume_from)
        if start_epoch >= self.epochs:
            self.logger.info("恢复点epoch>=设定的最大epoch，跳过")
            return
        try:
            self.before_all()
            self.before_train()

            for epoch_i in range(start_epoch, self.epochs):
                registry.register("current_epoch", epoch_i)
                self.before_running_epoch()
                if registry.get("stop_training"):
                    break
                self.train()
                self.valid()
                self.test()
                self.after_running_epoch()
        except BaseException as e:
            self.logger.critical(f"训练时，发生异常：{e.__class__.__name__},"
                                 f"异常信息：{e}")
            self.on_exception(e)
            # raise
        finally:
            self.after_train()
            self.after_all()


    def train(self):
        current_epoch = registry.get("current_epoch")
        is_fp16 = registry.get("cfg.training.fp16", False)
        scaler = GradScaler() if is_fp16 else None

        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        # ) as prof:
        for batch_i, data in enumerate(self.train_data_loader):
            registry.register("current_batch", batch_i)
            self.before_running_batch()
            self.train_data_loader.sampler.set_epoch(current_epoch)
            data = self._move_train_data_to_device(data)
            with self.maybe_autocast(is_fp16):
                model_output, _ = self.train_step(data)
                # self.model.module.visualize_architecture(model_output["logit"].mean(), '1')
                # break

            assert "loss" in model_output, "模型输出必须返回包含loss的字典"
            self.backward(scaler, model_output["loss"])
            registry.register("metric.loss", model_output["loss"].item())

            self.after_running_batch()


    def train_step(self, batch) -> tuple[dict, dict or None]:
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
            self.train_evaluator.process(model_output, batch)
            metric_val = self.train_evaluator.evaluate(len(model_output["loss"]))

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
            accumulation_steps = registry.get("cfg.training.gradient_accumulation", 1)
            max_norm = registry.get("cfg.training.grad_clip")

            # 缩放损失以平均梯度
            loss = loss / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
                # 混合精度训练中，梯度缩放器维护的梯度状态可能与实际梯度不同步。
                # 如果在多个反向传播调用后才执行unscale_()，可能导致梯度状态不一致。
                # scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            self.accumulation_count += 1
            if self.accumulation_count % accumulation_steps == 0:
                # 梯度裁剪
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=max_norm,
                        norm_type=2
                    )
                # 梯度缩放
                if scaler:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                current_step = registry.get("current_step")
                if self.scheduler:
                    self.scheduler.step(current_step)

                self.optimizer.zero_grad()

                self.accumulation_count = 0


    @torch.no_grad()
    def valid(self) -> dict or None:
        """for dynamic cover, this func is Not merged with test"""
        if self.valid_data_loader is None:
            first_call_warning("valid", "未提供valid_data_loader，跳过valid")
            return

        if self.valid_evaluator is None:
            first_call_warning("valid_evaluator", "未提供valid_evaluator")

        current_epoch = registry.get("current_epoch")
        if current_epoch is None:
            should_valid = True
        elif (current_epoch + 1) % registry.get("cfg.training.valid_every_n_epochs", 1) == 0:
            should_valid = True
        else:
            should_valid = False

        if not should_valid:
            return

        self.model.eval()
        for batch_i, data in enumerate(self.valid_data_loader):
            data = self._move_valid_data_to_device(data)
            model_output = self.valid_step(data)
        self.model.train()


        eval_result = None
        if self.valid_evaluator:
            eval_result = self.valid_evaluator.evaluate(self.valid_size)
            self.logger.info(f"Valid Summary: {eval_result}")
            self.wandb_log(eval_result)

        return eval_result


    def valid_step(self, batch) -> dict:
        model_output = self.model(**batch)
        self.valid_evaluator.process(model_output, batch) \
                            if self.valid_evaluator else None
        return model_output


    @torch.no_grad()
    def test(self) -> dict or None:
        if self.test_data_loader is None:
            first_call_warning("test", "未提供test_data_loader，跳过test")
            return

        if self.test_evaluator is None:
            first_call_warning("test_evaluator", "未提供test_evaluator")

        current_epoch = registry.get("current_epoch")
        if current_epoch is None:
            should_test = True
        elif (current_epoch + 1) % registry.get("cfg.training.test_every_n_epochs", 1) == 0:
            should_test = True
        else:
            should_test = False

        if not should_test:
            return

        self.model.eval()
        for batch_i, data in enumerate(self.test_data_loader):
            data = self._move_test_data_to_device(data)
            model_output = self.test_step(data)
        self.model.train()

        eval_result = None

        if self.test_evaluator:
            eval_result = self.test_evaluator.evaluate(self.test_size)
            self.logger.info(f"Test Summary: {eval_result}")
            self.wandb_log(eval_result)

        return eval_result


    def test_step(self, batch) -> dict:
        model_output = self.model.module.generate(**batch)
        self.test_evaluator.process(model_output, batch) \
                        if self.test_evaluator else None
        return model_output


    def before_all(self):
        super().before_all()
        self.logger.info(registry.get("device"))
        self.logger.info(registry.get("start_msg"))
        if registry.get("cfg.training.print_model", True):
            # support distributed print, means print only once in the main process
            print(registry.get("model_info"))



    def after_all(self):
        super().after_all()
        self.logger.close_file_handler()
        if dist.is_initialized():
            dist.destroy_process_group()


    @main_process
    def wandb_log(self, log_dict):
        if registry.get("cfg.wandb.wandb_enable", False):
            try:
                import wandb
            except ImportError:
                warnings.warn("未安装wandb，请使用pip install wandb安装.")
            else:
                if wandb.run is not None:
                    wandb.log(log_dict)
                    self.logger.info(f"wandb记录：{log_dict}")


    @staticmethod
    def _assign_runner_cfg(cfg: dict):
        registry.register("cfg", cfg)


    def _assign_logger(self):
        # 有的话就用已创建的，忽略参数
        # 没有的话就会用这些参数创建
        self.logger = Logger.get_instance(
            "logger",
            # **dict(registry.get("cfg.log")),
            log_level = registry.get("cfg.log.log_level", 'INFO'),
            to_file = registry.get("cfg.log.to_file", True),
            folder = registry.get("cfg.log.folder", './logs'),
            run_name = registry.get("cfg.run_name", 'default')
        )


    def _assign_runtime_parameter(self):
        if self.train_data_loader:
            self.train_size = len(self.train_data_loader.dataset)
        if self.valid_data_loader:
            self.valid_size = len(self.valid_data_loader.dataset)
        if self.test_data_loader:
            self.test_size = len(self.test_data_loader.dataset)

        self.accumulation_count = 0
        self.resume_from = registry.get("cfg.training.resume_from")
        if (modules := registry.get("cfg.training.activation_checkpoint")) is not None:
            self.logger.info(f"启用激活检查点: {modules}")
            from .activation_checkpointing import turn_on_activation_checkpointing
            turn_on_activation_checkpointing(self.model, modules)


    def _apply_launch_strategy(self):
        model_info = get_model_info(self.model)
        registry.register("model_info", model_info)
        print(better_dict_4_print(self.runner_config))

        # 单卡也使用分布式环境
        if torch.cuda.is_available():
            # if dist.is_available() and not dist.is_initialized():
            from ..dist.init import init_distributed_mode
            if not dist.is_initialized():
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
            try:
                import deepspeed
            except ImportError:
                raise ImportError("未安装deepspeed，无法使用deepspeed配置。"
                                  "请运行以下命令安装：\n"
                                  "pip install deepspeed")
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model = self.model,
                optimizer = self.optimizer if self.optimizer is not None else None,
                model_parameters = self.model.parameters(),
                config = ds_config,
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
                    lr = registry.get("cfg.optimizer.lr", 3e-4),
                    weight_decay= registry.get("cfg.optimizer.weight_decay", 0.01),
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

            from torch.utils.data import RandomSampler, SequentialSampler

            if isinstance(self.train_data_loader.sampler, RandomSampler):
                shuffle = True
            elif isinstance(self.train_data_loader.sampler, SequentialSampler):
                shuffle = False
            self.train_data_loader = wrap_dataloader(
                self.train_data_loader,
                shuffle = shuffle
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
        wandb_callback = None
        if registry.get("cfg.wandb.wandb_enable", False):
            try:
                import wandb
            except ImportError:
                raise ImportError("未安装wandb，无法使用wandb callback。"
                                  "请运行以下命令安装：\n"
                                  "pip install wandb")
            else:
                wandb_callback = WandbCallback()

        checkpoint_callback = CheckpointCallback(runner = self) \
                                if registry.get("cfg.pt.pt_save", True) \
                                   or self.resume_from else None
        self.checkpoint_callback = checkpoint_callback

        callbacks = [
            progress_callback,
            checkpoint_callback,
            wandb_callback,
            epoch_summary_callback,
        ]
        callbacks.extend(self.callbacks)
        self._register_callbacks(callbacks)


    def _move_data_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking = True)
        elif isinstance(data, (list, tuple)):
            return [self._move_data_to_device(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: self._move_data_to_device(v, device) for k, v in data.items()}
        else:
            warnings.warn(f"move data to device {device} failed, data type: {type(data)}; default return raw data!")
            return data

    def _move_train_data_to_device(self, data):
        """
        maybe should be overridden
        """
        return self._move_data_to_device(data, registry.get("device"))


    def _move_valid_data_to_device(self, data):
        """
        maybe should be overridden
        """
        return self._move_data_to_device(data, registry.get("device"))

    def _move_test_data_to_device(self, data):
        """
        maybe should be overridden
        """
        return self._move_data_to_device(data, registry.get("device"))
