import os
from datetime import datetime

import torch

from ..common.mixin import ManagerMixin
from ..common.util import now


class Logger(ManagerMixin):
    _LOG_LEVEL_MAP = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    _LOG_LEVEL = _LOG_LEVEL_MAP["INFO"]

    def __init__(self, name = "logger", level = "INFO", to_file = False, folder = './logs', run_name = '', **kwargs):
        super().__init__(name, **kwargs)
        assert level in Logger._LOG_LEVEL_MAP, \
            f"Invalid log level: {level}, valid levels: {Logger._LOG_LEVEL_MAP.keys()}"
        self.name = name
        self.set_log_level(level)
        self.to_file = to_file
        self.folder = folder if to_file else None
        self.run_name = run_name if to_file else None
        self._file_handler = None
        if self.to_file:
            self._post_init()


    def _post_init(self):
        if not self.to_file:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        prefix = self.run_name + '_' if self.run_name else ''
        self.run_name = prefix + now() + ".log"
        self.file_path = os.path.join(self.folder, self.run_name)
        os.makedirs(self.folder, exist_ok = True)
        self._file_handler = open(self.file_path, "a")
        self.info(f"日志管理器{self.name}已创建：{os.path.abspath(self.file_path)}")


    @classmethod
    def set_log_level(cls, level: str):
        """设置全局日志等级"""
        if level is None or level.strip() == "":
            raise ValueError("Log level cannot be None or empty.")
        level = level.upper()
        if level in cls._LOG_LEVEL_MAP:
            cls._GLOBAL_LOG_LEVEL = cls._LOG_LEVEL_MAP[level]
        else:
            raise ValueError(
                f"Invalid log level: {level}. Valid levels: {list(cls._LOG_LEVEL_MAP.keys())}"
            )

    def close_file_handler(self):
        if self._file_handler:
            self.info(f"日志文件已保存：{self.file_path}")
            self._file_handler.close()


    def log_print(self, text, level = "INFO", end = "\n", timestamp = True, to_file = None):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        assert level in Logger._LOG_LEVEL_MAP, \
            f"Invalid log level: {level}, valid levels: {Logger._LOG_LEVEL_MAP.keys()}"

        current_level = Logger._LOG_LEVEL_MAP.get(level.upper(), level)

        if current_level < Logger._GLOBAL_LOG_LEVEL:
            return

        if to_file is None:
            to_file = self.to_file

        if timestamp:
            timestamp_str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            prefix = f"[{timestamp_str}] [{level.upper()}]" if level else f"[{timestamp_str}]"
        else:
            prefix = f"[{level.upper()}]" if level else ""

        log_message = f"{prefix} {text}" if prefix else text

        print(log_message, end = end, flush = True)

        if to_file and self._file_handler:
            self._file_handler.write(f"{log_message}{end}")

        return

    # Note: for 分布式打印，但是已过期，通过dist.dist的重定向built_in_print实现
    def just_print(self, text: str, end = "\n", to_file = None):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        if to_file is None:
            to_file = self.to_file

        print(text, end = end, flush = True)

        if to_file and self._file_handler:
            self._file_handler.write(f"{text}{end}")


    def debug(self, text: str, end = "\n", timestamp = True):
        return self.log_print(text, level = "DEBUG", end = end, timestamp = timestamp)

    def info(self, text: str, end = "\n", timestamp = True):
        return self.log_print(text, level = "INFO", end = end, timestamp = timestamp)

    def warning(self, text: str, end = "\n", timestamp = True):
        return self.log_print(text, level = "WARNING", end = end, timestamp = timestamp)

    def error(self, text: str, end = "\n", timestamp = True):
        return self.log_print(text, level = "ERROR", end = end, timestamp = timestamp)

    def critical(self, text: str, end = "\n", timestamp = True):
        return self.log_print(text, level = "CRITICAL", end = end, timestamp = timestamp)
