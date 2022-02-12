#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader, default_collate
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

from omeganlp.core.io import (
    dump_config,
    dump_json,
    dump_vocab_txt,
    get_class_fullname,
    instantiate_interface,
    read_config,
    read_json,
)
from omeganlp.core.mixins import HyperparametersMixin
from omeganlp.core.vocabulary import Vocabulary

from . import BaseDataProcessor


@dataclass
class Example:
    text_a: str
    text_b: str
    label: Union[str, int, List[Union[str, int]]]
    guid: str = ""


class SequenceClassificationDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]


class BaseSequenceClassificationProcessor(BaseDataProcessor, HyperparametersMixin):
    def __init__(
        self,
        labels: Union[List[Union[int, str]], str],
        tokenizer_dir: str,
        max_length: int,
        key_text_a: str = "text_a",
        key_text_b: str = "text_b",
        key_label="label",
        key_guid: str = "_id",
        **kwargs
    ):
        super().__init__()

        if isinstance(labels, str):
            assert os.path.exists(labels)
            labels = list(read_json(labels))
        assert len(set(labels)) > 1 and len(labels) == len(set(labels))

        self.save_hyperparameters()

        self.max_length = max_length
        self.labels = labels
        self.tokenizer_dir = tokenizer_dir
        self.key_text_a = key_text_a
        self.key_text_b = key_text_b
        self.key_label = key_label
        self.key_guid = key_guid

        self._index2label = {i: label for i, label in enumerate(self.labels)}
        self._label2index = {label: i for i, label in enumerate(self.labels)}

        self._keymap = {
            "text_a": self.key_text_a,
            "text_b": self.key_text_b,
            "label": self.key_label,
            "guid": self.key_guid,
        }

        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        return None

    @property
    def num_labels(self):
        return len(self.labels)

    def label2index(self, label: Union[int, str]) -> Optional[int]:
        return self._label2index.get(label)

    def index2label(self, index: int) -> Optional[Union[int, str]]:
        return self._index2label.get(index)

    def encode_label(self, label: Union[int, str]) -> Optional[int]:
        return self.label2index(label)

    def decode(self, index: int) -> Union[int, str]:
        return self.index2label(index)

    def read(self, data_dir: str, **kwargs):
        return list(read_json(data_dir))


class TextProcessor(BaseSequenceClassificationProcessor):
    def __init__(
        self,
        labels: List[Union[int, str]],
        tokenizer_dir: str = "jieba.lcut",
        max_length: int = 128,
        key_text_a: str = "text_a",
        key_text_b: str = "text_b",
        key_label: str = "label",
        key_guid: str = "_id",
        concat: str = " ",
        vocab: Optional[Union[str, List[str]]] = None,
        vocab_corpus: Optional[Union[str, Iterable]] = None,
        load_dir: Optional[str] = None,
        **vocab_kwargs
    ):
        """
        Keyword Arguments:
        labels:List[Union[int,str]]        --
        vocab :str                     --
        tokenize:Callable[[str],List[str]] -- oov_subword:bool                   -- (default True)
        padding_idx:int                    -- (default 0)
        unk_idx:int                        -- (default 1)
        **kwargs                           --
        """

        super(TextProcessor, self).__init__(
            labels,
            tokenizer_dir,
            max_length,
            key_text_a,
            key_text_b,
            key_label,
            key_guid,
            concat=concat,
            **vocab_kwargs
        )

        self.concat = concat

        self.vocabulary = None
        if vocab is not None:
            self.vocabulary = Vocabulary(vocab, **vocab_kwargs)
        elif vocab_corpus is not None:
            self.vocabulary = self.build_vocabulary_from_corpus(
                vocab_corpus, **vocab_kwargs
            )

            print("built")

    def get_tokenizer(self):
        return instantiate_interface(self.tokenizer_dir)

    def build_vocabulary_from_corpus(self, corpus: Union[str, Iterable], **kwargs):
        if isinstance(corpus, str) and os.path.exists(corpus):
            corpus = list(read_json(corpus))

        texts = []
        for c in corpus:
            example = None
            if isinstance(c, dict):
                example_dict = {k: c.get(v) for k, v in self._keymap.items()}
                example = Example(**example_dict)
            elif isinstance(c, Example):
                example = c

            assert example is not None

            text_a = example.text_a if example.text_a is not None else ""
            text_b = example.text_b if example.text_b is not None else ""
            texts += [text_a, text_b]

        return Vocabulary.build_from_corpus(texts, self.tokenizer, **kwargs)

    def processing(
        self, example: Union[Example, Dict], is_train: bool = True, max_length=None
    ):
        if isinstance(example, dict):
            example_dict = {k: example.get(v) for k, v in self._keymap.items()}
            example = Example(**example_dict)

        max_length = self.max_length if max_length is None else max_length
        text_a = example.text_a if example.text_a is not None else ""
        text_b = example.text_b if example.text_b is not None else ""
        text = self.concat.join([text_a, text_b]).strip()
        token = self.tokenizer(text)
        encoded = self.vocabulary(token, max_length)

        if is_train:
            return encoded, self.encode_label(example.label)
        else:
            return encoded

    def get_dataset(self, data_dir, is_train: bool = True, **kwargs):
        data = list(self.read(data_dir, **kwargs))
        return self.data2dataset(data, is_train, **kwargs)

    def data2dataset(self, datas, is_train: bool = True, **kwargs):

        if self.vocabulary is None:
            # use default params for vocabulary
            self.vocabulary = self.build_vocabulary_from_corpus(datas)

        processed = []
        for data in tqdm(datas):
            processed += [self.processing(data, is_train=is_train, **kwargs)]

        if is_train:
            processed = sorted(processed, key=lambda x: len(x[0]))
            x, y = [*zip(*processed)]
            return SequenceClassificationDataset(x, y)
        else:
            return SequenceClassificationDataset(x=processed)

    @staticmethod
    def dataset2dataloader(dataset, batch_size, **kwargs):
        is_train = dataset.y is not None

        def collate_fn(batch):
            padding_batch = []
            for b in batch:
                if is_train:
                    x, y = b
                    padding_batch += [
                        (
                            torch.tensor(x),
                            torch.tensor(y),
                        )
                    ]
                else:
                    x = b
                    padding_batch += [torch.tensor(x)]

            if is_train:
                return default_collate(padding_batch)
            else:
                return default_collate(padding_batch), None

        kwargs.update(collate_fn=collate_fn)

        return DataLoader(dataset, batch_size, **kwargs)

    def save(self, save_dir):
        self.update_hparams()
        self.vocabulary.update_hparams()

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # save vocab
        vocab = self.vocabulary.hparams.pop("vocab")
        vocab_save_path = os.path.join(save_dir, "vocab.txt")
        dump_vocab_txt(vocab, vocab_save_path)
        self.vocabulary.hparams.update({"vocab": vocab_save_path})
        self.hparams.update(**self.vocabulary.hparams)

        # save labels
        labels_save_path = os.path.join(save_dir, "labels.json")
        dump_json(self.labels, labels_save_path)
        self.hparams.update({"labels": labels_save_path})

        # compatible with hydra
        self.hparams.update({"_target_": get_class_fullname(self)})

        # save processor
        processor_save_path = os.path.join(save_dir, "processor.yaml")
        dump_config(self.hparams, processor_save_path)

    @classmethod
    def load(cls, load_dir=None, **kwargs):
        """
        load from a specific dir and use kwargs to overwrite
        update vocab & labels if files exist in load_dir
        kwargs:the first priority
        """
        config = dict()
        if load_dir is not None and os.path.exists(
            os.path.join(load_dir, "processor.yaml")
        ):
            processor_save_path = os.path.join(load_dir, "processor.yaml")
            config = read_config(processor_save_path)

            # compatible with hydra
            config.pop("_target_", None)

            labels_path = config["labels"]
            labels_path = os.path.join(load_dir, os.path.basename(labels_path))
            config["labels"] = labels_path

            vocab_path = config["vocab"]
            vocab_path = os.path.join(load_dir, os.path.basename(vocab_path))
            config["vocab"] = vocab_path

            assert os.path.exists(config["labels"])
            assert os.path.exists(config["vocab"])

        config.update(**kwargs)
        return cls(**config)


class PretrainedModelProcessor(BaseSequenceClassificationProcessor):
    def __init__(
        self,
        tokenizer_dir: str,
        labels: List[Union[str, int]],
        max_length: int = 100,
        key_text_a: str = "text_a",
        key_text_b: str = "text_b",
        key_label: str = "label",
        key_guid: str = "_id",
        **tokenizer_kwargs
    ):
        """
        Keyword Arguments:
        tokenizer_dir:str             --
        labels:List[Union[str, int]]  --
        max_length:int                -- (default 100)
        key_text_a:str                -- (default 'text_a')
        key_text_b:str                -- (default 'text_b')
        key_label:str                 -- (default 'label')
        key_guid:str                  -- (default '_id')
        """

        self._tokenizer_kwargs = tokenizer_kwargs
        super(PretrainedModelProcessor, self).__init__(
            labels,
            tokenizer_dir,
            max_length,
            key_text_a,
            key_text_b,
            key_label,
            key_guid,
        )

    def get_tokenizer(self):
        return self._tokenizer_init(self.tokenizer_dir, **self._tokenizer_kwargs)

    def get_dataset(self, data_dir: str, is_train: bool = True, **kwargs):
        datas = list(self.read(data_dir, **kwargs))
        return self.data2dataset(datas, is_train, **kwargs)

    def data2dataset(
        self, datas: List[Union[Example, Dict]], is_train: bool = True, **kwargs
    ):
        # TODO: return dict-like result,use new api
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        y = None
        for data in tqdm(datas):
            processed = self.processing(data, is_train)
            input_ids += [processed["input_ids"]]
            attention_mask += [processed["attention_mask"]]
            token_type_ids += [processed["token_type_ids"]]
            labels += [processed.get("label")]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        x = TensorDataset(input_ids, attention_mask, token_type_ids)
        if is_train:
            y = torch.tensor(labels)
        return SequenceClassificationDataset(x, y)

    @staticmethod
    def dataset2dataloader(dataset, batch_size, **kwargs):
        is_train = dataset.y is not None

        def collate_fn(batch):
            if is_train:
                x, y = default_collate(batch)
                y = y.type(torch.long)
                return x, y
            else:
                x = default_collate(batch)
                return x, None

        kwargs.update(collate_fn=collate_fn)

        return DataLoader(dataset, batch_size, **kwargs)

    def processing(self, example: Union[Example, Dict], is_train: bool = True):
        if isinstance(example, dict):
            text_a = example.get(self.key_text_a)
            text_b = example.get(self.key_text_b)
            guid = example.get(self.key_guid)
            label = example.get(self.key_label)
            example = Example(text_a, text_b, label, guid)
        encoded = self.tokenizer(
            example.text_a,
            example.text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        if is_train:
            label = self.encode_label(example.label)
            encoded.update(label=label)
        return encoded

    def save(self, save_dir: str):
        self.update_hparams()

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # save tokenizer
        # XXX: Toakenizer.from_pretrained(...) config.json not found
        self.tokenizer.save_pretrained(save_dir)
        self.hparams.update(tokenizer_dir=save_dir)

        # save labels
        labels_save_path = os.path.abspath(os.path.join(save_dir, "labels.json"))
        dump_json(self.labels, labels_save_path)
        self.hparams.update(labels=labels_save_path)

        # save processor
        processor_save_path = os.path.join(save_dir, "processor.yaml")
        dump_config(self.hparams, processor_save_path)

    @classmethod
    def load(cls, load_dir=None, **kwargs):
        config = dict()
        if load_dir is not None and os.path.exists(
            os.path.join(load_dir, "processor.yaml")
        ):
            processor_save_path = os.path.join(load_dir, "processor.yaml")
            config = read_config(processor_save_path)

            assert os.path.exists(config["labels"])

        config.update(**kwargs)
        return cls(**config)

    @staticmethod
    def _tokenizer_init(tokenizer_dir, backup_tokenizer=BertTokenizer, **kwargs):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
        except Exception as e:
            return backup_tokenizer.from_pretrained(tokenizer_dir, **kwargs)
