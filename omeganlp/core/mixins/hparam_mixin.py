#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning.core.mixins import (
    HyperparametersMixin as PL_HyperparametersMixin,
)


class HyperparametersMixin(PL_HyperparametersMixin):
    def __init__(self):
        super().__init__()

    def update_hparams(self):
        for k in self.hparams.keys():
            if hasattr(self, k):
                v = getattr(self, k)
                self.hparams.update({k: v})

    def set_hparams_attributes(self):
        for k, v in self.hparams.items():
            setattr(self, k, v)
