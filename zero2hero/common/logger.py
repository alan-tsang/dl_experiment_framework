import os
import logging
from typing import Optional

import torch

from ..dist import master_only
from ..common.mixin import ManagerMixin
from ..common.util import now


class CustomFormatter(logging.Formatter):
    """自定义 Formatter，支持动态控制换行符"""
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        fmt = fmt or '[%(asctime)s] | [%(levelname)s] | [%(filename)s:%(lineno)d %(funcName)s] %(message)s'
        super().__init__(fmt, datefmt)
        self.terminator = ""  # 禁用默认换行符

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return message + getattr(record, "end", "\n")


class Logger(ManagerMixin):
    _LOG_LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(
        self,
        name: str = "logger",
        level: str = "INFO",
        to_file: bool = False,
        folder: str = "./logs",
        run_name: str = "",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self._LOG_LEVEL_MAP[level.upper()])
        self.to_file = to_file
        self.folder = folder
        self.run_name = run_name
        self._file_handler: Optional[logging.FileHandler] = None

        # 配置处理器
        self._configure_handlers()

    def _configure_handlers(self) -> None:
        """配置控制台和文件处理器"""
        # 避免重复添加处理器
        if not self._logger.handlers:
            # 控制台处理器（始终添加，但仅在分布式 Rank 0 或非分布式环境生效）
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(CustomFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(console_handler)

        # 文件处理器（仅在 to_file=True 时添加）
        if self.to_file:
            self._add_file_handler()

    def _add_file_handler(self) -> None:
        """添加文件处理器"""
        # 分布式环境下仅在 Rank 0 进程处理
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        # 创建日志目录
        os.makedirs(self.folder, exist_ok=True)

        # 生成文件名
        prefix = f"{self.run_name}_" if self.run_name else ""
        filename = f"{prefix}{now()}.log"
        file_path = os.path.join(self.folder, filename)

        # 配置文件处理器
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(CustomFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        # file_handler.addFilter(self._distributed_filter)
        self._logger.addHandler(file_handler)
        self._file_handler = file_handler

        # 记录日志文件路径
        self._logger.info(f"日志管理器 {self.name} 已创建：{os.path.abspath(file_path)}")

    # @staticmethod
    # def _distributed_filter(record: logging.LogRecord) -> bool:
    #     """分布式环境过滤器：仅在 Rank 0 进程记录日志"""
    #     if torch.distributed.is_initialized():
    #         return torch.distributed.get_rank() == 0
    #     return True

    def close_file_handler(self) -> None:
        """关闭文件处理器"""
        if self._file_handler:
            self._logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def log(
        self,
        text: str,
        level: str = "INFO",
        end: str = "\n",
    ) -> None:
        log_level = self._LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
        if self._logger.isEnabledFor(log_level):
            self._logger.log(log_level, text, extra={"end": end}, stacklevel=3)

    # 简化日志方法（直接委托给 log_print）
    def debug(self, obj: any, end: str = "\n") -> None:
        self.log(str(obj), "DEBUG", end)

    def info(self, obj: any, end: str = "\n") -> None:
        self.log(str(obj), "INFO", end)

    def warning(self, obj: any, end: str = "\n") -> None:
        self.log(str(obj), "WARNING", end)

    def error(self, obj: any, end: str = "\n") -> None:
        self.log(str(obj), "ERROR", end)

    def critical(self, obj: any, end: str = "\n") -> None:
        self.log(str(obj), "CRITICAL", end)

    @master_only
    def just_print(self, obj: any, end: str = "\n", time_stamp = False, to_file: Optional[bool] = None) -> None:
        if time_stamp:
            obj = f"[{now()}] {obj}"
        print(obj, end=end, flush=True)
        if to_file and self._file_handler:
            self._file_handler.stream.write(f"{obj}{end}")

