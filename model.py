# coding=utf8

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def forward(self, x, *wargs, **kwargs):
        raise NotImplemented


class BertClassifier(BaseModel):
    def __init__(self,conf):
        super(BertClassifier, self).__init__()
        self.conf = conf
        self.bert_path = getattr(conf,'bert_path',None)
        pretrain_name = 'bert-base-cased'
        if self.bert_path:
            pretrain_name = self.bert_path
        print(f'Bert Model from: {pretrain_name}')
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size,conf.class_num)

    def forward(self,input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask = attention_mask),
        activated_cls = outputs[1]
        activated_cls = self.drop(activated_cls)
        logits = self.classifier(activated_cls)
        return logits