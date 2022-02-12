#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        num_labels,
        vocab_size,
        embedding_dim=100,
        embedding=None,
        freeze_embedding=False,
        kernel_sizes=[2, 3, 4, 5],
        filter_num=2,
        dropout=0.5,
        **kwargs
    ):
        super(TextCNN, self).__init__()

        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding, **kwargs
            )
        else:
            # embedding initiailzation is Gaussians by default
            self.embedding = nn.Embedding(vocab_size, embedding_dim, **kwargs)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filter_num,
                    kernel_size=(k, embedding_dim),
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=len(kernel_sizes) * filter_num, out_features=num_labels
        )

    def embedding_encoding(self, x):
        return self.embedding(x)

    def embedding_modeling(self, embedding):
        x = embedding.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def forward(self, x):
        embedding = self.embedding_encoding(x)
        return self.embedding_modeling(embedding)
