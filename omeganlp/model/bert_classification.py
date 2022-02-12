#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class BertSequenceClassification(nn.Module):
    def __init__(self, bert_model_dir: str, num_labels: int, **kwargs):
        """
        Keyword Arguments:
        bert_model_dir:str --
        num_labels:int     -- (default 2)
        **kwargs           --
        """
        super(BertSequenceClassification, self).__init__()
        kwargs.update(num_labels=num_labels)
        config = AutoConfig.from_pretrained(bert_model_dir, **kwargs)

        # TODO: use BertModel.from_pretrained instead
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            bert_model_dir, config=config
        )

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        output = self.bert(input_ids, attention_mask, token_type_ids)
        logits = output[0]
        return logits
