# coding=utf8

import torch
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.to_list()
        if isinstance(item, slice):
            begin, end, step = item.indices(len(self))
            return [self.get_example(i) for i in range(begin,end,step)]
        if isinstance(item, list):
            return [self.get_example(i) for i in item]
        else:
            return self.get_example(item)

    def get_example(self,item):
        raise NotImplemented