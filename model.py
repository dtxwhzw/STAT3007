# coding=utf8

import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def forward(self, x, *wargs, **kwargs):
        raise NotImplemented