#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List

from omeganlp.core.mixins import HyperparametersMixin
from omeganlp.core.io import read_vocab_txt
import os


class Vocabulary(HyperparametersMixin):
    def __init__(
        self,
        vocab,
        padding_idx=0,
        unk_idx=1,
        pad="[PAD]",
        unk="[UNK]",
        oov_subword=True,
    ):
        super().__init__()
        if isinstance(vocab, str) and os.path.exists(vocab):
            vocab = read_vocab_txt(vocab)
        if pad in vocab and vocab.index(pad) != padding_idx:
            vocab.pop(vocab.index(pad))
            vocab.insert(padding_idx, pad)
        if unk in vocab and vocab.index(unk) != unk_idx:
            vocab.pop(vocab.index(unk))
            vocab.insert(unk_idx, unk)

        self.save_hyperparameters()

        self.set_hparams_attributes()

        self._stoi = {word: i for i, word in enumerate(self.vocab)}
        self._itos = {i: word for i, word in enumerate(self.vocab)}

    def stoi(self, word):
        return self._stoi.get(word, self.unk_idx)

    def itos(self, index):
        return self._itos.get(index, self.unk)

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def build_from_corpus(cls, corpus: Iterable[str], tokenizer, **kwargs):
        vocab = []
        for c in corpus:
            vocab += tokenizer(c)

        vocab = list(set(vocab))
        return cls(vocab=vocab, **kwargs)

    def encode(self, text: str):
        idx = self.stoi(text)
        if self.oov_subword and idx == self.unk_idx and len(text) > 1:
            text = list(text)
            idx = [self.stoi(t) for t in text]

        else:
            idx = [idx]

        return idx

    def __call__(self, sequence: List[str], max_length) -> List[int]:
        encoded = []
        for s in sequence:
            encoded += self.encode(s)
        seq_len = len(encoded)
        if seq_len <= max_length:
            encoded += [self.padding_idx] * (max_length - seq_len)
        else:
            encoded = encoded[:max_length]
        return encoded
