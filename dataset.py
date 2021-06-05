# coding=utf8

import torch
from torch.utils.data.dataset import Dataset
from gensim import models


class BaseDataset(Dataset) :
    def __init__(self) :
        super(BaseDataset, self).__init__()

    def __len__(self) :
        raise NotImplemented

    def __getitem__(self, item) :
        if torch.is_tensor(item) :
            item = item.to_list()
        if isinstance(item, slice) :
            begin, end, step = item.indices(len(self))
            return [self.get_example(i) for i in range(begin, end, step)]
        if isinstance(item, list) :
            return [self.get_example(i) for i in item]
        else :
            return self.get_example(item)

    def get_example(self, item) :
        raise NotImplemented


class SentimentDataset(BaseDataset):
    def __init__(self,tokenizer,texts,labels,max_len):
        super(SentimentDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label,dtype=torch.long)
        }
#
#
# class W2vDataset(BaseDataset):
#     def __init__(self,w2v,tokenizer,texts,labels,max_len):
#         super(W2vDataset, self).__init__()
#         self.w2v = models.KeyedVectors.load_word2vec_format(w2v)
#         self.tokenzier = tokenizer
#         self.texts = texts
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, item):
#         text = self.texts[item]
#         label = self.labels[item]
#
#         text = self.tokenzier.tokenize(text)
#
#         vec = torch.tensor([self.w2v[i] for i in text], dtype = torch.long)


class GPT2Dataset(BaseDataset):
    def __init__(self, tokenizer, texts, labels, max_len):
        super(GPT2Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer.pad_token = '[UNK]'

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt")

        return {
            'text':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label,dtype=torch.long)
        }