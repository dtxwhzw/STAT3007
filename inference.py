# coding=utf8

import os
import torch
import json
from torch.utils.data import dataloader
from torch.nn import Softmax
from model import BertClassifier
from dataset import SentimentDataset
from transformers import BertTokenizer
from utils.conf_utils import Config


class Predictor(object):
    def __init__(self, args):
        pretrain_name = 'bert-base-cased'
        if args.model_info.bert_path:
            pretrain_name = args.model_info.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        print(f"Tokenizer from:{pretrain_name}")
        train_conf = args.train_info
        model_conf = args.model_info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_num = model_conf.class_num
        self.model = BertClassifier(model_conf)
        self.model.load_state_dict(torch.load(train_conf.model_path, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.lr = train_conf.lr
        self.max_len = train_conf.max_seq_len
        self.conf = args
        self.label_map = json.load(open(args.label_map_path))
        self.id2label = dict([(i, label_str) for label_str, i in self.label_map.items()])
        self.softmax = Softmax(dim=1)

    def predict(self, sens):
        d_loader = self.sen_2_dl(sens)
        y_pred = list()
        with torch.no_grad():
            for batch in d_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(input_ids, attention_mask)
                logits = torch.sigmoid(logits)
                y_pred.append(logits)
        y_pred = torch.cat(y_pred)
        y_pred = y_pred.cpu().numpy()
        res = list()
        for y in y_pred:
            res.append(self._score_2_dict(y))
        return res

    def _score_2_dict(self, single_pred):
        res = dict()
        for i, score in enumerate(single_pred):
            label_str = self.id2label[i]
            res[label_str] = float(score)
        return res

    def sen_2_dl(self, sens):
        texts = [i.strip() for i in sens]
        labels = [999]# this is a invalid parameter but dataloader needs the this
        ds = SentimentDataset(self.tokenizer, texts, labels, self.max_len)
        _loader = dataloader.DataLoader(ds, batch_size=self.conf.train_info.batch_size, shuffle=False)
        return _loader


def parse_args(conf):
    if isinstance(conf, str):
        config = Config.from_json_file(conf)
    else:
        config = Config.from_dict(conf)
    return config


if __name__ == '__main__':
    import sys
    from time import time
    f_conf = sys.argv[1]
    t_args = parse_args(f_conf)
    t_predictor = Predictor(t_args)
    t0 = time()
    # input_str = sys.argv[2]
    input_str = ['Always on the search for the optimal GTD program. Was hooked several years to omnifocus and things and stumbled upon todoist by chance. Well, for me the best presently, better file and email integration, top design and flexibility']
    res = t_predictor.predict(input_str)
    print(res)
    print(time()-t0)
