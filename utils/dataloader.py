# coding=utf8

from torch.utils.data import dataloader
import numpy as np
from utils.data_prepare import prepare
from dataset import SentimentDataset


def get_data_loader(f_path,batch_size,label_map,tokenizer,max_len,num_workers):
    np.random.seed(14)
    texts,labels = prepare(f_path,label_map)
    ds = SentimentDataset(tokenizer,texts,labels,max_len)
    return dataloader.DataLoader(ds,batch_size=batch_size,num_workers=num_workers,shuffle=True)


