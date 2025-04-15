"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:29
================
"""
from ..common.logger import Logger


class BaseCallBack:
    """
    CallBack的基类，基于训练的生命周期设计。
    """

    def __init__(self):
        self.trainer = None
        self._info = {}

    def on_register(self):
        """
        Called when the callback is registered.
        """
        self.logger = Logger.get_instance("logger")
        self.logger.info(f"Callback {self.__class__.__name__} registered.")

    def on_exception(self, *args, **kwargs):
        """
        Called when an exception is raised.
        """
        pass

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_running_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_running_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_running_batch(self):
        """
        Called before each iteration.
        """
        pass

    def after_running_batch(self):
        """
        Called after each iteration.
        """
        pass

    def before_all(self):
        """
        Called before the training starts.
        """
        pass

    def after_all(self):
        """
        Called after the training ends.
        """
        pass
