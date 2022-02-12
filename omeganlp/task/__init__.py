#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import hydra
from omegaconf import DictConfig

from omeganlp.core.io import find_class


class Task:
    def __init__(self, trainer, litmodel, datamodule, processor=None):
        self.trainer = trainer
        self.litmodel = litmodel
        self.datamodule = datamodule

        if datamodule is not None:
            self.processor = datamodule.processor
        else:
            self.processor = processor

        self.root_dir = self.trainer.default_root_dir
        self.model = litmodel.model

    def train(self):
        return self.trainer.fit(self.litmodel, self.datamodule)

    def validate(self):
        return self.trainer.validate(self.litmodel, self.datamodule)

    def test(self):
        return self.trainer.test(self.litmodel, self.datamodule)

    def infer(self, examples):
        pass

    def save_inferer(
        self, inferer_dir: str = "./inferer", to_onnx: bool = False, compress=False
    ):
        inferer_dir = os.path.join(self.root_dir, inferer_dir)
        os.makedirs(inferer_dir, exist_ok=True)

        model_save_dir = os.path.join(inferer_dir, "model")
        os.makedirs(model_save_dir, exist_ok=True)

        processor_save_dir = os.path.join(self.root_dir, "processor")
        os.makedirs(processor_save_dir, exist_ok=True)

        self.processor.save(processor_save_dir)
        # TODO: save best model
        if to_onnx:
            # TODO:try to trans to onnx model
            pass
        # TODO: save litmodel
        # TODO: save task

    def load_inferer(self, inferer_dir: str = "./inferer"):
        pass

    def load_best_model(self):
        pass

    @staticmethod
    def update_model(datamodule):
        return {}

    @staticmethod
    def load(config: DictConfig):
        task_cls = find_class(config.task._target_)

        datamodule = hydra.utils.instantiate(config.datamodule)
        datamodule.prepare_data()

        num_training_steps = config.trainer.max_epochs * datamodule.num_step("train")

        add_model_params = task_cls.update_model(datamodule)

        model = hydra.utils.instantiate(config.litmodel.model, **add_model_params)

        optim = hydra.utils.instantiate(
            config.optim, model=model, num_training_steps=num_training_steps
        )

        litmodel = hydra.utils.instantiate(
            config.litmodel, model=model, optim_config=optim.config
        )

        # cbs = [config.callbacks.model_checkpoint, config.callbacks.early_stopping,
        #        config.callbacks.exception]
        # TODO: exception save checkpoint
        cbs = [config.callbacks.model_checkpoint, config.callbacks.early_stopping]
        callbacks = [hydra.utils.instantiate(cb) for cb in cbs]

        trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)
        task = hydra.utils.instantiate(config.task, trainer, litmodel, datamodule)

        return task


class SequenceClassificationTask(Task):
    @staticmethod
    def update_model(datamodule):
        processor = datamodule.processor
        model_params = {"num_labels": processor.num_labels}

        if hasattr(processor, "vocabulary"):
            model_params.update({"vocab_size": len(processor.vocabulary)})

        return model_params
