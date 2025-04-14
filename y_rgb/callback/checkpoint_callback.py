"""
添加resume保存 - optimizer, scheduler, epoch, seed, step
现在是不支持deepspeed 三阶段的权重保存吗？
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import torch

from .base_callback import BaseCallBack
from ..common.logger import Logger
from ..common.registry import registry
from ..common.util import now


@registry.register_callback("checkpoint")
class CheckpointCallback(BaseCallBack):
    """
    保存训练过程中的模型检查点，支持多种保存策略。

    目录结构示例：
    - folder/
        - run_name-YYYY-mm-dd-HH_MM_SS/
            - initial.pt    # 初始模型
            - best.pt       # 最佳模型
            - epoch_{n}.pt  # 周期保存
            - epoch_{n}-batch_{m}.pt  # 批次保存
            - last.pt       # 最终模型
            - exception_*.pt  # 异常保存
            - epoch_{n}-[batch_{m}-]{metric}_{value}.pt  # TopK 模型

    Args:
        folder (str): 检查点保存根目录
        prefix (str): 运行名称前缀，默认为空
        every_n_epochs (int): 每隔多少周期保存
        every_n_batches (int): 每隔多少批次保存
        topk (int): 保留最佳K个模型（需配合monitor）
        monitor (str): 监控指标名称
        monitor_greater_is_better (bool): 指标方向（True表示越小越好）
    """

    def __init__(
        self,
        runner
    ):
        super().__init__()
        self.runner = runner

        self.topk = None
        self.monitor = None
        self.folder = None
        self.every_n_epochs = None
        self.every_n_batches = None
        self.monitor_greater_is_better = None
        self.topk_models = None
        self.save_enabled = None
        self.logger = None
        self.is_deepspeed = registry.get("cfg.training.ds_config") is not None

    def on_register(self):
        super().on_register()
        self.topk = registry.get("cfg.pt.pt_topk")

        self.folder = self._generate_run_folder(
            base_folder = registry.get("cfg.pt.pt_save_dir"),
            prefix = registry.get("cfg.run_name")
        )
        os.makedirs(self.folder, exist_ok=True)

        # 保存策略参数
        self.every_n_epochs = registry.get("cfg.pt.pt_save_n_epochs")
        self.every_n_batches = registry.get("cfg.pt.pt_save_n_batches")

        monitor = list(registry.get("cfg.pt.pt_best_monitor").items())[0]
        self.monitor = monitor[0]
        self._validate_init_params(self.topk, self.monitor)

        self.monitor_greater_is_better = monitor[1]

        # 状态跟踪
        self.topk_models: List[Dict] = []
        self.save_enabled = registry.get("is_main_process")

        self.logger = Logger.get_instance("logger")


    @staticmethod
    def _validate_init_params(topk: int, monitor: Optional[str]):
        if topk > 0 and not monitor:
            raise ValueError("必须指定监控指标(monitor)当topk>0时")


    @staticmethod
    def _generate_run_folder(base_folder: str, prefix: str) -> str:
        """生成带时间戳的运行目录"""
        timestamp = now()
        return os.path.join(
            base_folder, f"{prefix}_{timestamp}" if prefix else timestamp
        )


    def _get_model_state(self):
        return self.runner.model.module.state_dict() if hasattr(self.runner.model, "module") else self.runner.model.state_dict()

    def _get_optimizer_state(self):
        if registry.get("cfg.training.ds_config"):
            return None  # DeepSpeed自行管理
        return self.runner.optimizer.state_dict()


    def _get_scheduler_state(self):
        return self.runner.scheduler.state_dict() if self.runner.scheduler else None


    def save_checkpoint(self, name: str):
        """智能保存检查点"""
        if registry.get("cfg.training.ds_config"):
            self._save_deepspeed_checkpoint(name)
        else:
            self._save_standard_checkpoint(name)


    def _save_deepspeed_checkpoint(self, name: str):
        """DeepSpeed三阶段专用保存"""
        from deepspeed.utils import logger as ds_logger
        client_state = {
            "epoch": registry.get("current_epoch", 0) + 1,
            "batch": registry.get("current_batch", 0) + 1,
            "rng_state": torch.get_rng_state(),
            "cfg": dict(registry.get("cfg"))
        }
        self.runner.model.save_checkpoint(
            save_dir = self.folder,
            tag = name,
            client_state = client_state,
            save_latest = True
        )
        ds_logger.info(f"DeepSpeed检查点保存至: {self.folder}/name")


    def _save_standard_checkpoint(self, name: str):
        """保存标准PyTorch检查点（非DeepSpeed环境）"""
        # 确保仅在主进程执行保存操作
        if not registry.get("is_main_process", True):
            return

        name += '.pt'
        # 创建保存目录（如果不存在）
        os.makedirs(self.folder, exist_ok = True)

        # 获取完整训练状态
        checkpoint = {
            "model": self._get_model_state(),
            "optimizer": self._get_optimizer_state(),
            "scheduler": self._get_scheduler_state(),
            "epoch": registry.get("current_epoch", 0) + 1,
            "batch": registry.get("current_batch", 0) + 1,
            "rng_state": torch.get_rng_state(),
            "cfg": dict(registry.get("cfg")),
        }

        # 保存到文件
        save_path = os.path.join(self.folder, name)
        torch.save(checkpoint, save_path)
        self.logger.info(f"标准检查点保存至: {save_path}")
    
    # 各生命周期回调方法
    def before_train(self):
        """训练开始前保存初始模型"""
        self.save_checkpoint("initial")
        self.logger.info(f"初始模型保存成功: initial")

    
    def after_train(self):
        """训练结束后保存最终模型"""
        self.save_checkpoint("final")
        self.logger.info(f"最终模型保存成功: final")


    def after_running_epoch(self):
        current_epoch = registry.get("current_epoch")
        """周期结束处理"""
        if self.every_n_epochs and current_epoch % self.every_n_epochs == 0:
            self.save_checkpoint(f"epoch_{current_epoch+1}")
            if not self.is_deepspeed:
                self.save_checkpoint(f"latest")
            self.logger.info(f"周期模型保存成功: epoch_{current_epoch+1}")
            self._handle_topk_saving(current_epoch, None)

    def after_running_batch(self):
        current_epoch = registry.get("current_epoch")
        current_batch = registry.get("current_batch")
        """批次结束处理"""
        if self.every_n_batches and current_batch % self.every_n_batches == 0:
            self.save_checkpoint(f"epoch_{current_epoch+1}-batch_{current_epoch+1}")
            if not self.is_deepspeed:
                self.save_checkpoint(f"latest")
            self.logger.info(f"批次模型保存成功: epoch_{current_epoch}-batch_{current_batch}")
            self._handle_topk_saving(current_epoch, current_batch)


    def on_exception(self, exception: Exception):
        """异常处理"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exception_{type(exception).__name__}_{ts}"
        self.save_checkpoint(filename)
        self.logger.error(f"异常发生时保存模型: {filename}")


    def load_checkpoint(self, path: str):
        """智能加载检查点"""
        if registry.get("cfg.training.ds_config"):
            return self._load_deepspeed_checkpoint(path)
        else:
            return self._load_standard_checkpoint(path)


    def _load_deepspeed_checkpoint(self, path: str):
        tag = os.path.split('/')[-1]
        if tag == "latest":
            latest_file = os.path.join(path, "latest")
            with open(latest_file, 'r') as f:
                tag = f.read().strip()

        load_path, client_state = self.runner.model.load_checkpoint(
            load_dir = path,
            tag = tag,
            load_module_strict = True,
            load_optimizer_states = True,
            load_lr_scheduler_states = True
        )

        # 恢复附加状态
        if client_state:
            torch.set_rng_state(client_state["rng_state"])
            registry.register("cfg", client_state["cfg"])

        return client_state.get("epoch", 0), client_state.get("batch", 0)


    def _load_standard_checkpoint(self, path: str) -> tuple[int, int]:
        """加载标准PyTorch检查点（非DeepSpeed环境）"""
        checkpoint = torch.load(path, map_location = "cpu")

        # 1. 加载模型状态
        model = self.runner.model
        if hasattr(model, "module"):  # 分布式训练包装
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        # 2. 加载优化器状态
        if checkpoint["optimizer"] is not None:
            self.runner.optimizer.load_state_dict(checkpoint["optimizer"])

        # 3. 加载学习率调度器状态
        if checkpoint["scheduler"] is not None and self.runner.scheduler is not None:
            self.runner.scheduler.load_state_dict(checkpoint["scheduler"])

        # 4. 恢复随机状态
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])

        # 5. 恢复配置
        if "cfg" in checkpoint:
            registry.register("cfg", checkpoint["cfg"])

        # 6. 返回恢复的epoch和batch位置
        return checkpoint.get("epoch", 0), checkpoint.get("batch", 0)

    def _handle_topk_saving(self, epoch: int, batch: Optional[int]):
        """处理TopK模型保存逻辑"""
        if not (self.topk > 0 and self.monitor and self.save_enabled):
            return

        if (value := registry.get(f"metric.{self.monitor}")) is None:
            self.logger.warning(f"监控指标 '{self.monitor}' 不存在")
            return

        self._update_topk_models(value, epoch, batch)


    def _update_topk_models(
            self, current_value: float, epoch: int, batch: Optional[int]
    ):
        """维护TopK模型列表并更新最佳模型"""
        filename = self._generate_topk_filename(epoch, batch, current_value)
        self._add_to_topk(filename, current_value)
        self._prune_topk_list()
        self._update_best_model()

    def _generate_topk_filename(
            self, epoch: int, batch: Optional[int], value: float
    ) -> str:
        base = f"epoch_{epoch + 1}"
        if batch is not None:
            base += f"-batch_{batch + 1}"
        filename = f"{base}-{self.monitor}_{value:.4f}"
        return filename


    def _add_to_topk(self, filename: str, value: float):
        """尝试添加新模型到TopK列表"""
        if len(self.topk_models) < self.topk or self._is_better_than_worst(value):
            self.save_checkpoint(filename)
            self.logger.info(f"TopK保存模型: {filename}")
            self.topk_models.append(
                {"value": value, "path": os.path.join(self.folder, filename)}
            )
            self.topk_models.sort(key = lambda x: x["value"], reverse = self.monitor_greater_is_better)

    def _is_better_than_worst(self, value: float) -> bool:
        """判断当前值是否优于最差TopK值"""
        if not self.topk_models:
            return False
        worst = self.topk_models[-1]["value"]
        return value > worst if self.monitor_greater_is_better else value < worst


    def _prune_topk_list(self):
        while len(self.topk_models) > self.topk:
            removed = self.topk_models.pop()
            try:
                if os.path.isdir(removed["path"]):
                    shutil.rmtree(removed["path"])  # 删除目录
                else:
                    os.remove(removed["path"])  # 删除文件
                self.logger.info(f"TopK移除旧模型: {os.path.basename(removed['path'])}")
            except FileNotFoundError:
                self.logger.warning(f"文件未找到: {removed['path']}")


    def _update_best_model(self):
        if not self.topk_models:
            return
        best = self.topk_models[0]
        best_source = best["path"]
        if self.is_deepspeed:
            best_dest = os.path.join(self.folder, "best")
            if os.path.exists(best_dest):
                shutil.rmtree(best_dest)
            shutil.copytree(best_source, best_dest)
            self.logger.info(f"DeepSpeed最佳模型更新至: {best_dest}")
        else:
            best_source += '.pt'
            best_dest = os.path.join(self.folder, "best.pt")
            shutil.copyfile(best_source, best_dest)
            self.logger.info(f"标准最佳模型更新至: {best_dest}")
        self.logger.info(f"TopK更新最佳模型: {self.monitor}={best['value']:.4f}")


__all__ = ["CheckpointCallback"]
