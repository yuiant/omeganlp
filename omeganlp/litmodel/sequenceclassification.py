#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchmetrics import Accuracy, MetricCollection

from omeganlp.litmodel import LitModel


class SequenceClassificationModel(LitModel):
    @property
    def val_metrics(self):
        return MetricCollection({"val/acc": Accuracy(top_k=1)})

    @property
    def test_metrics(self):
        return MetricCollection({"test/acc": Accuracy(top_k=1)})
