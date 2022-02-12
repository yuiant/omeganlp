#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning.callbacks import Callback


class ExceptionCallback(Callback):
    def on_exception(trainer, litmodel, exception):
        print("saving checkpoints by Exception")
        # TODO:save checkpoint
        raise exception
