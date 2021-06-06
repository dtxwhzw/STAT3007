# coding=utf8

import os
import torch
import pickle
import numpy as np
import json
from torch.utils.data import dataloader
from model import BertClassifier, GPT2Classifier
from dataset import SentimentDataset,GPT2Dataset
from transformers import BertTokenizer, GPT2Tokenizer
from utils.data_prepare import prepare
from utils import metrics
from utils.conf_utils import Config


class Evaluator(object):
    def __init__(self, args):
        pretrain_name = 'bert-base-cased'
        if args.model_info.bert_path:
            pretrain_name = args.model_info.bert_path
        print(f"Tokenizer from:{pretrain_name}")
        train_conf = args.train_info
        model_conf = args.model_info
        self.model_type = model_conf.model
        if self.model_type == 'bert_seq':
            self.model = BertClassifier(model_conf)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
            self.ds = SentimentDataset
        if self.model_type == 'GPT2':
            self.model = GPT2Classifier(model_conf)
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrain_name)
            self.ds = GPT2Dataset
        self.model.load_state_dict(torch.load(train_conf.model_path))
        self.device = train_conf.device
        self.class_num = model_conf.class_num
        self.model.to(self.device)
        self.lr = train_conf.lr
        self.max_len = train_conf.max_seq_len
        self.conf = args
        self.label_map = json.load(open(args.label_map_path))
        self.id2label = dict([(i, label_str) for label_str, i in self.label_map.items()])

    def run(self, batch_size=64):
        test_path = self.conf.train_info.test_path
        test_loader = self.get_data_loader(test_path,batch_size)
        acc, recall, f1_score, cm, report,res = self.evaluate(test_loader)
        print(f"Accuracy score of the model is {acc}")
        print(f"Recall score of the model is {recall}")
        print(f"F1 score of the model is {f1_score}")
        print(f"Confusion matrix of the model is {cm}")
        print(report)
        dir_ = os.path.dirname(test_path)
        dir_ = os.path.dirname(dir_)
        dir_ = os.path.split(dir_)[0]
        new_path = os.path.join(dir_,'logs','bad_case.json')
        f = open(new_path,'w')
        for i in res:
            print(json.dumps(i,ensure_ascii=False),file=f)

    def evaluate(self, _loader):
        self.model.eval()
        y_true = list()
        y_pred = list()
        res = []
        with torch.no_grad():
            for batch in _loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                y = batch['labels']
                y = torch.squeeze(y, 1)
                y = y.to(self.device)
                logits = self.model(input_ids, attention_mask)
                y_true.append(y)
                y_pred.append(logits)
                pred_labels = torch.argmax(logits,dim=1)
                preds = pred_labels.cpu().numpy()
                true = batch['labels'].squeeze().numpy()
                if len(true) < 1:
                    continue
                for i, c_y in enumerate(true):
                    if c_y != preds[i]:
                        tmp_dict = {
                            'true_label':self.id2label[c_y],
                            'pred_label':self.id2label[preds[i]],
                            'text': batch['text'][i]
                        }
                        res.append(tmp_dict)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
        cm = metrics.cal_cm(y_true, y_pred)
        acc_score = metrics.cal_accuracy(y_true, y_pred)
        recall = metrics.cal_recall(y_true, y_pred)
        f1_score = metrics.cal_f1(y_true, y_pred)
        label_range = [i for i in range(len(self.label_map))]
        target_name = [x[0] for x in sorted(self.label_map.items(), key=lambda x: x[1])]
        report = metrics.get_classification_report(y_true, y_pred, label_range, target_name)
        return acc_score, recall, f1_score, cm, report, res

    def get_data_loader(self,f_path,batch_size):
        np.random.seed(14)
        texts, labels = prepare(f_path,self.label_map)
        ds = self.ds(self.tokenizer, texts, labels, self.max_len)
        return dataloader.DataLoader(ds, batch_size=batch_size, num_workers=self.conf.num_workers, shuffle=True)


def parse_args(conf):
    if isinstance(conf, str):
        config = Config.from_json_file(conf)
    else:
        config = Config.from_dict(conf)
    return config


def main(args):
    evaluator = Evaluator(args)
    train_conf = args.train_info
    batch_size = train_conf.batch_size
    evaluator.run(batch_size=batch_size)


if __name__ == '__main__':
    import sys
    f_conf = sys.argv[1]
    t_args = parse_args(f_conf)
    main(t_args)