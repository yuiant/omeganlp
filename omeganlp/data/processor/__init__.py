#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from abc import ABCMeta, abstractmethod
from typing import Any

import torch


class BaseDataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def processing(self, example: Any, is_train: bool = False):
        """
        example --> features
        Process an example(a single sample) to your specified encoded features

        Keyword Arguments:
        example:Any   -- your data example
        is_train:bool -- (default False) whether the example contains 'y'
        """

    @abstractmethod
    def read(self, data_dir: str, **kwargs):
        """
        Keyword Arguments:
        data_dir:str -- raw data dir
        """

    @abstractmethod
    def get_dataset(self, data_dir: str, is_train: bool = False, **kwargs):
        """
        Keyword Arguments:
        data_dir:str -- raw data dir
        """

    @abstractmethod
    def save(self, **kwargs):
        """
        save the metadata for your processor
        or pickle the processor
        """

    @classmethod
    @abstractmethod
    def load(cls, **kwargs):
        """
        call load classmethod to load saved metadata and initialize processor
        or unpickle the processor
        """

    def data2dataset(self, datas, is_train: bool = False, **kwargs):
        raise NotImplementedError

    @staticmethod
    def dataset2dataloader(dataset, batch_size, **kwargs):
        raise NotImplementedError

    def update_params(self, **kwargs):
        raise NotImplementedError

    def decode(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_processed(data_dir):
        return torch.load(data_dir)

    @staticmethod
    def save_processed(data, data_dir):
        dirname = os.path.dirname(data_dir)
        os.makedirs(dirname, exist_ok=True)
        torch.save(data, data_dir)
