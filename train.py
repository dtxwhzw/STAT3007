# coding=utf8

import os
import torch
import pickle
import numpy as np
import json
from torch.utils.data import dataloader
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import BertClassifier
from dataset import SentimentDataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from utils.data_prepare import prepare
from utils import metrics
from utils.conf_utils import Config


class Train(object):
    def __init__(self,args):
        pretrain_name = 'bert-base-cased'
        if args.model_info.bert_path:
            pretrain_name = args.model_info.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        print(f"Tokenizer from:{pretrain_name}")
        train_conf = args.train_info
        model_conf = args.model_info
        self.model = BertClassifier(model_conf)
        self.device = train_conf.device
        self.class_num = model_conf.class_num
        self.model.to(self.device)
        self.lr = train_conf.lr
        self.tensorboard_log_path = getattr(train_conf,"log_path",None)
        if self.tensorboard_log_path:
            self.writer = SummaryWriter(train_conf.log_path)
        self.max_len = train_conf.max_seq_len
        self.conf = args
        self.label_map = json.load(open(args.label_map_path))
        self.train_history_path = getattr(train_conf,"train_history_path",None)

    def train(self, train_path, valid_path, epochs, model_path, lr, batch_size):
        train_loader = self.get_data_loader(train_path,batch_size=batch_size)
        valid_loader = self.get_data_loader(valid_path,batch_size=batch_size)
        optimizer = AdamW(self.model.parameters(),
                          lr=lr,
                          eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        criterion = F.cross_entropy
        best_f1 = 0.
        train_history = dict()
        for epoch in tqdm(range(epochs)):
            train_history.setdefault(epoch,{})
            self.train_epoch(train_loader,optimizer,scheduler,criterion,epoch)
            acc, f1_score, cm, report = self.evaluate(valid_loader)
            if f1_score > best_f1:
                best_f1 = f1_score
                torch.save(self.model.state_dict(),model_path)
            print('\n\n')
            print(f"Following is validation metrics at epoch {epoch}")
            print(f"Accuracy of the model is {acc}")
            print(f"F1 score is : {f1_score}, and best_f1 is {best_f1}")
            print(f'Confusion matrix is: {cm}')
            print(report)
            train_history[epoch]['train_result'] = {
                'acc': acc,
                'f1' : f1_score,
                'best_f1' : best_f1,
                'confusion matrix' : cm,
                'report' : report
            }

            acc, f1_score, cm, report = self.evaluate(train_loader)
            if self.tensorboard_log_path:
                self.writer.add_scalar('Train/F1',f1_score,epochs)
            print('\n\n')
            print(f"Following is train metrics at epoch {epoch}")
            print(f"Accuracy of the model is {acc}")
            print(f"F1 score is : {f1_score}")
            print(f'Confusion matrix is: {cm}')
            print(report)
            train_history[epoch]['eval_result'] = {
                'accuracy' : acc,
                'f1' : f1_score,
                'confusion matrix' : cm,
                'report' : report
            }
            if self.train_history_path:
                with open(f"{self.train_history_path}",'wb') as f:
                    pickle.dump(train_history, f)

        if self.tensorboard_log_path:
            self.writer.flush()
            self.writer.close()

    def train_epoch(self,train_loader,optimizer,scheduler,criterion,epoch):
        self.model.train()
        loss_arr = []
        for step, batch in tqdm(enumerate(train_loader),desc=f'iterating in epoch: {epoch}'):
            self.model.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            y = batch['labels']
            y = torch.squeeze(y,1)
            y = y.to(self.device)
            logits = self.model(input_ids,attention_mask)
            loss = criterion(logits,y)
            loss_arr.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_average = torch.mean(torch.tensor(loss_arr))
        if self.tensorboard_log_path:
            self.writer.add_scalar("Train/Loss", loss_average, epoch)

    def evaluate(self,_loader):
        self.model.eval()
        y_true = []
        y_pred = []
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
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        cm = metrics.cal_cm(y_true, y_pred)
        acc_score = metrics.cal_accuracy(y_true, y_pred)
        f1_score = metrics.cal_f1(y_true, y_pred)
        label_range = [i for i in range(len(self.label_map))]
        target_name = [x[0] for x in sorted(self.label_map.items(), key=lambda x: x[1])]
        report = metrics.get_classification_report(y_true, y_pred, label_range, target_name)
        return acc_score, f1_score, cm, report

    def get_data_loader(self,f_path,batch_size):
        np.random.seed(14)
        texts, labels = prepare(f_path,self.label_map)
        ds = SentimentDataset(self.tokenizer, texts, labels, self.max_len)
        return dataloader.DataLoader(ds, batch_size=batch_size, num_workers=self.conf.num_workers, shuffle=True)


def parse_args(conf):
    if isinstance(conf, str):
        config = Config.from_json_file(conf)
    else:
        config = Config.from_dict(conf)
    return config


def main(args):
    trainer = Train(args)
    train_conf = args.train_info
    train_path = train_conf.train_path
    valid_path = train_conf.valid_path
    epochs = train_conf.epochs
    lr = train_conf.lr
    model_path = train_conf.model_path
    batch_size = train_conf.batch_size
    trainer.train(train_path, valid_path, epochs, model_path, lr, batch_size)


if __name__ == '__main__':
    import sys
    f_conf = sys.argv[1]
    t_args = parse_args(f_conf)
    main(t_args)