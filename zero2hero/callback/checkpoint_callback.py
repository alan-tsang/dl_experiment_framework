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
            - initial.pth    # 初始模型
            - best.pth       # 最佳模型
            - epoch_{n}.pth  # 周期保存
            - epoch_{n}-batch_{m}.pth  # 批次保存
            - last.pth       # 最终模型
            - exception_*.pth  # 异常保存
            - epoch_{n}-[batch_{m}-]{metric}_{value}.pth  # TopK 模型

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
        model,
    ):
        super().__init__()
        self.model = model
        self.topk = None
        self.monitor = None
        self.folder = None
        self.every_n_epochs = None
        self.every_n_batches = None
        self.monitor_greater_is_better = None
        self.topk_models = None
        self.save_enabled = None
        self.logger = None

    def on_register(self):
        super().on_register()
        self.topk = registry.get("cfg.pt.pt_topk")
        self.monitor = registry.get("cfg.pt.pt_best_monitor")
        self._validate_init_params(self.topk, self.monitor)

        self.folder = self._generate_run_folder(
            base_folder = registry.get("cfg.pt.pt_save_dir"),
            prefix = registry.get("cfg.run_name")
        )
        os.makedirs(self.folder, exist_ok=True)

        # 保存策略参数
        self.every_n_epochs = registry.get("cfg.pt.pt_save_n_epochs")
        self.every_n_batches = registry.get("cfg.pt.pt_save_n_batches")

        self.monitor_greater_is_better = registry.get\
            (f"cfg.training.train_monitor.{self.monitor}")

        # 状态跟踪
        self.topk_models: List[Dict] = []
        self.save_enabled = registry.get("is_main_process")

        self.logger = Logger.get_instance("logger")

    @staticmethod
    def _validate_init_params(topk: int, monitor: Optional[str]):
        """验证初始化参数合法性"""
        if topk > 0 and not monitor:
            raise ValueError("必须指定监控指标(monitor)当topk>0时")

    @staticmethod
    def _generate_run_folder(base_folder: str, prefix: str) -> str:
        """生成带时间戳的运行目录"""
        timestamp = now()
        return os.path.join(
            base_folder, f"{prefix}_{timestamp}" if prefix else timestamp
        )


    def _get_state_dict(self):
        """获取模型状态字典（兼容分布式训练）"""
        return (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

    def save_model(self, filename: str):
        """保存模型到指定文件名"""
        if not self.save_enabled:
            return

        path = os.path.join(self.folder, filename)
        torch.save(self._get_state_dict(), path)

    # 各生命周期回调方法
    
    def before_train(self):
        """训练开始前保存初始模型"""
        self.save_model("initial.pth")
        self.logger.info(f"初始模型保存成功: initial.pth")

    
    def after_train(self):
        """训练结束后保存最终模型"""
        self.save_model("last.pth")
        self.logger.info(f"最终模型保存成功: last.pth")

    def after_running_epoch(self):
        current_epoch = registry.get("current_epoch")
        """周期结束处理"""
        if self.every_n_epochs and current_epoch % self.every_n_epochs == 0:
            self.save_model(f"epoch_{current_epoch+1}.pth")
            self.logger.info(f"周期模型保存成功: epoch_{current_epoch+1}.pth")
            self._handle_topk_saving(current_epoch, None)

    def after_running_batch(self):
        current_epoch = registry.get("current_epoch")
        current_batch = registry.get("current_batch")
        """批次结束处理"""
        if self.every_n_batches and current_batch % self.every_n_batches == 0:
            self.save_model(f"epoch_{current_epoch+1}-batch_{current_epoch+1}.pth")
            self.logger.info(f"批次模型保存成功: epoch_{current_epoch}-batch_{current_batch}.pth")
            self._handle_topk_saving(current_epoch, current_batch)

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
        """生成TopK模型文件名"""
        base = f"epoch_{epoch+1}"
        if batch is not None:
            base += f"-batch_{batch+1}"
        return f"{base}-{self.monitor}_{value:.4f}.pth"

    def _add_to_topk(self, filename: str, value: float):
        """尝试添加新模型到TopK列表"""
        if len(self.topk_models) < self.topk or self._is_better_than_worst(value):
            self.save_model(filename)
            self.logger.info(f"TopK保存模型: {filename}")
            self.topk_models.append(
                {"value": value, "path": os.path.join(self.folder, filename)}
            )
            self.topk_models.sort(key=lambda x: x["value"], reverse=self.monitor_greater_is_better)

    def _is_better_than_worst(self, value: float) -> bool:
        """判断当前值是否优于最差TopK值"""
        if not self.topk_models:
            return False
        worst = self.topk_models[-1]["value"]
        return value > worst if self.monitor_greater_is_better else value < worst

    def _prune_topk_list(self):
        """修剪超出TopK数量的旧模型"""
        while len(self.topk_models) > self.topk:
            removed = self.topk_models.pop()
            try:
                os.remove(removed["path"])
                self.logger.info(f"TopK移除旧模型: {os.path.basename(removed['path'])}")
            except FileNotFoundError:
                self.logger.warning(f"文件未找到: {removed['path']}")

    def _update_best_model(self):
        """更新最佳模型文件"""
        if self.topk_models:
            best = self.topk_models[0]
            shutil.copyfile(best["path"], os.path.join(self.folder, "best.pth"))
            self.logger.info(f"TopK更新最佳模型: {self.monitor}={best['value']:.4f}")

    def on_exception(self, exception: Exception):
        """异常处理"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exception_{type(exception).__name__}_{ts}.pth"
        self.save_model(filename)
        self.logger.error(f"异常发生时保存模型: {filename}")



__all__ = ["CheckpointCallback"]
