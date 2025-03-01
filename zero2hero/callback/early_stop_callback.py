from .base_callback import BaseCallBack
from ..common.registry import registry

@registry.register_callback("early_stop")
class EarlyStopCallBack(BaseCallBack):
    """
    Early stopping callback to stop training when a monitored metric stops improving.

    Parameters:
        monitor (str): Metric to monitor (e.g., "val_loss").
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): Whether to print messages when training is stopped early.
        mode (str): One of {"min", "max"}. In "min" mode, training will stop
                   when the monitored metric stops decreasing; in "max" mode,
                   when the metric stops increasing.
    """

    def __init__(
        self,
        monitor: str = "loss",
        delta: float = 0.0001,
        patience: int = 3,
        verbose: bool = True,
        reverse: bool = False,
    ):
        super(EarlyStopCallBack, self).__init__()
        self.monitor = monitor
        self.delta = delta
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse

        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None


    def _is_better(self, current: float, best: float) -> bool:
        if self.reverse:
            return current < best - self.delta
        else:
            return current > best + self.delta


    def after_running_epoch(self):
        """
        Check the monitored metric after each epoch and decide if training should stop.
        """
        current_value = self.trainer.info.get(self.monitor, None)

        if current_value is None:
            self.logger.warning(f"Warning: monitored metric '{self.monitor}' not found in trainer.info.")
            return

        if self.best_value is None:
            self.best_value = current_value
            return

        if self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = registry.get("current_epoch")
                self.trainer.info["should_stop"] = True
                if self.verbose:
                    self.logger.info(
                        f"触发提前停止训练. Best {self.monitor}: {self.best_value:.4f} at epoch {self.stopped_epoch - self.patience}."
                    )


__all__ = [
    'EarlyStopCallBack'
]