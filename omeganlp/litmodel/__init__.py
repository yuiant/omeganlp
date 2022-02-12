#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, model, criteria=None, optim_config=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criteria = criteria

        if optim_config is not None:
            optim_config = dict(optim_config)
        self.optim_config = optim_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criteria(logits, y)
        return loss

    def training_step_end(self, batch_parts):
        loss = batch_parts
        self.log_dict({"training/loss": loss})
        # TODO gradient hist

    @property
    def val_metrics(self):
        raise NotImplementedError

    @property
    def test_metrics(self):
        raise NotImplementedError

    def training_epoch_end(self, training_step_outputs):
        return super().training_epoch_end(training_step_outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self.criteria(outputs, y)
        return {"loss": loss, "outputs": outputs, "y": y}

    def validation_step_end(self, batch_parts):
        loss = batch_parts["loss"]
        y = batch_parts["y"]
        outputs = batch_parts["outputs"]
        metrics = {}
        metrics = self.val_metrics(outputs, y)
        metrics.update({"val/loss": loss})
        self.log_dict(metrics)

    def validation_epoch_end(self, validation_step_outputs):
        return super().validation_epoch_end(validation_step_outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        loss = batch_parts["loss"]
        y = batch_parts["y"]
        outputs = batch_parts["outputs"]
        metrics = {}
        metrics = self.test_metrics(outputs, y)
        metrics.update({"test/loss": loss})
        self.log_dict(metrics)

    def test_epoch_end(self, test_step_outputs):
        return super().test_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        if self.optim_config is None:
            raise NotImplementedError

        else:
            return self.optim_config
