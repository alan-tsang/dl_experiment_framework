from datetime import datetime

from .base_callback import BaseCallBack
from ..common.logger import Logger
from ..common.registry import registry


@registry.register_callback("epoch_summary")
class EpochSummaryCallBack(BaseCallBack):
    def __init__(self):
        super().__init__()
        self._epoch_start_time = None
        self._epoch_end_time = None
        self.logger = Logger.get_instance("logger")

    def before_running_epoch(self):
        self._epoch_start_time = datetime.now()

    def after_running_epoch(self):
        self._epoch_end_time = datetime.now()
        self._log_epoch_summary()

    def _log_epoch_summary(self):
        elapsed_time = self._epoch_end_time - self._epoch_start_time
        elapsed_seconds = elapsed_time.total_seconds()
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{seconds:.3f}"

        # Note: registry.get('metric') is a dictionary that contains the metric values
        info_items = (
            ", ".join([f"{k}: {v}" for k, v in registry.get('metric').items()])
            if registry.get('metric')
            else "无"
        )

        self.logger.info(
            f"Epoch {registry.get('current_epoch') + 1} 用时: {time_str}s 总结：{info_items}\n-----------"
        )
