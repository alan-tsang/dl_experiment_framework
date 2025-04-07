"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math

from .common.registry import registry


@registry.register_lr_scheduler("LinearWarmupStepLRScheduler")
class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


@registry.register_lr_scheduler("LinearWarmupCosineLRScheduler")
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        max_lr,
        warmup_rate=0.05,
        warmup_start_lr=2e-5,
        **kwargs
    ):
        self.optimizer = optimizer
        self.iters_per_epoch = iters_per_epoch
        self.max_steps = max_epoch * iters_per_epoch

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_rate = warmup_rate
        self.warmup_steps = int(self.warmup_rate * self.max_steps)
        self.warmup_start_lr = min(warmup_start_lr, min_lr)

    def step(self, cur_step):
        if cur_step < self.warmup_steps:
            lr = warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                warmup_start_lr=self.warmup_start_lr,
                max_lr=self.max_lr,
            )
            # print(lr)
        else:
            lr = cosine_lr_schedule(
                step=cur_step - self.warmup_steps,
                optimizer=self.optimizer,
                max_steps=self.max_steps - self.warmup_steps,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
            )
            # print(lr)


def cosine_lr_schedule(optimizer, step, max_steps, max_lr, min_lr):
    """Decay the learning rate"""
    lr = (max_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * step / max_steps)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def warmup_lr_schedule(optimizer, step, max_step, warmup_start_lr, max_lr):
    """Warmup the learning rate"""
    lr = warmup_start_lr + (max_lr - warmup_start_lr) * step / max(max_step, 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
