import os
import sys
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

        self._configure_handlers()

    def _configure_handlers(self) -> None:
        if not self._logger.handlers:
            # logging 默认是std.err，与print没有保持一致
            # 因此会出现打印顺序不符合预期
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(CustomFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(console_handler)

        if self.to_file:
            self._add_file_handler()

    def _add_file_handler(self) -> None:
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
        self._logger.addHandler(file_handler)
        self._file_handler = file_handler

        # 记录日志文件路径
        self._logger.info(f"日志管理器 {self.name} 已创建：{os.path.abspath(file_path)}")


    def close_file_handler(self) -> None:
        if self._file_handler:
            self._logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def log(
        self,
        text: str,
        level: str = "INFO",
    ) -> None:
        log_level = self._LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
        if self._logger.isEnabledFor(log_level):
            self._logger.log(log_level, text, stacklevel=3)

    def debug(self, obj: any) -> None:
        self.log(str(obj), "DEBUG")

    def info(self, obj: any) -> None:
        self.log(str(obj), "INFO")

    def warning(self, obj: any) -> None:
        self.log(str(obj), "WARNING")

    def error(self, obj: any) -> None:
        self.log(str(obj), "ERROR")

    def critical(self, obj: any) -> None:
        self.log(str(obj), "CRITICAL")

    @master_only
    def just_print(self, obj: any, end: str = "\n", time_stamp = False, to_file: Optional[bool] = None) -> None:
        if time_stamp:
            obj = f"[{now()}] {obj}"
        print(obj, end=end, flush=True)
        if to_file and self._file_handler:
            self._file_handler.stream.write(f"{obj}{end}")
