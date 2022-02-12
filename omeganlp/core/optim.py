#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AdamW, get_linear_schedule_with_warmup
from omeganlp.core.io import instantiate_interface


def default_pretrained_optimizer_and_scheduler(
    model, num_training_steps, num_warmup_steps=0.01, weight_decay=0.01, lr=1e-5
):
    model_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model_parameters if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model_parameters if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    if num_warmup_steps < 1:
        num_warmup_steps = int(num_training_steps * num_warmup_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler


class OptimComponents:
    def __init__(
        self, optim_fn, model, optimizer_config=None, lr_scheduler_config=None, **kwargs
    ):
        # TODO: make optimizer_config do
        if isinstance(optim_fn, str):
            optim_fn = instantiate_interface(optim_fn)
        optimizer, lr_scheduler = optim_fn(model, **kwargs)
        config = {"optimizer": optimizer}

        if lr_scheduler is not None:
            if lr_scheduler_config is not None:
                lr_scheduler_config.update({"scheduler": lr_scheduler})
            else:
                lr_scheduler_config = lr_scheduler
            config.update({"lr_scheduler": lr_scheduler_config})

        self._config = config

    @property
    def config(self):
        return self._config
