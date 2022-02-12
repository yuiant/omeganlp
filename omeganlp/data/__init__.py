#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule

from omeganlp.core.io import check_path
from omeganlp.data.processor import BaseDataProcessor


class DataType:
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class DataModule(LightningDataModule):
    def __init__(
        self,
        processor: BaseDataProcessor,
        data_root: str,
        save_processed_data: bool = False,
        batch_size=None,
        train_batch_size=64,
        val_batch_size=64,
        test_batch_size=64,
        **kwargs
    ):
        super().__init__()

        if batch_size is not None:
            train_batch_size = batch_size
            val_batch_size = batch_size
            test_batch_size = batch_size

        # self.save_hyperparameters()

        self.processor = processor
        self.data_root = data_root

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.trainset = []
        self.valset = []
        self.testset = []

        self.save_processed_data = save_processed_data
        self.dataloader_kwargs = kwargs

    def get_dataset(self, dataset_path: Optional[str] = None, mode=DataType.TRAIN):
        """ Agent of processor.get_dataset """

        if dataset_path is None and self.data_root is not None:
            dataset_path = check_path(self.data_root, mode, ignore_exception=True)

        if isinstance(dataset_path, str) and os.path.exists(dataset_path):
            if dataset_path.endswith("pt"):
                dataset = self.processor.load_processed(dataset_path)
            else:
                is_train = True
                dataset = self.processor.get_dataset(dataset_path, is_train=is_train)

        return dataset

    def dataset2dataloader(self, dataset, batch_size):
        """ Agent of processor.dataset2dataloader """
        return self.processor.dataset2dataloader(
            dataset, batch_size, **self.dataloader_kwargs
        )

    def num_step(self, mode=DataType.TRAIN):
        def _num_step(dataset, batch_size):
            return int(np.ceil(len(dataset) / batch_size))

        if mode == DataType.TRAIN:
            return _num_step(self.trainset, self.train_batch_size)

        elif mode == DataType.DEV:
            return _num_step(self.valset, self.val_batch_size)

        elif mode == DataType.TEST:
            return _num_step(self.testset, self.test_batch_size)

    def prepare_data(self):
        # TODO: may move to setup method
        if check_path(self.data_root, DataType.TRAIN):
            self.trainset = self.get_dataset(mode=DataType.TRAIN)

        if check_path(self.data_root, DataType.DEV):
            self.valset = self.get_dataset(mode=DataType.DEV)

        if check_path(self.data_root, DataType.TEST):
            self.testset = self.get_dataset(mode=DataType.TEST)

        if self.save_processed_data:
            data_dir = os.path.join(self.data_root, ".processed")
            os.makedirs(data_dir, exist_ok=True)

            self.processor.save_processed(
                self.trainset, os.path.join(data_dir, "train.pt")
            )
            self.processor.save_processed(self.valset, os.path.join(data_dir, "dev.pt"))
            self.processor.save_processed(
                self.testset, os.path.join(data_dir, "test.pt")
            )

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return self.dataset2dataloader(self.trainset, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return self.dataset2dataloader(self.valset, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return self.dataset2dataloader(self.testset, batch_size=self.test_batch_size)
