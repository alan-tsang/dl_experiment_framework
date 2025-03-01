"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:35
================
"""
from ..callback.base_callback import BaseCallBack


class RunnerBase:

    def __init__(self):
        self.callbacks = []

    def fit(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def valid(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def _register_callbacks(self, callbacks):
        self.callbacks = []
        for callback in callbacks:
            if callback is None:
                continue
            assert isinstance(callback, BaseCallBack),\
            "Callbacks must be subclass of BaseCallBack."
            callback.on_register()
            self.callbacks.append(callback)

    def _unregister_callbacks(self):
        self.callbacks = []
        return True

    def on_exception(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_exception(*args, **kwargs)

    def before_all(self):
        for callback in self.callbacks:
            callback.before_all()

    def after_all(self):
        for callback in self.callbacks:
            callback.after_all()

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def before_running_batch(self):
        for callback in self.callbacks:
            callback.before_running_batch()

    def after_running_batch(self):
        for callback in self.callbacks:
            callback.after_running_batch()

    def before_running_epoch(self):
        for callback in self.callbacks:
            callback.before_running_epoch()

    def after_running_epoch(self):
        for callback in self.callbacks:
            callback.after_running_epoch()