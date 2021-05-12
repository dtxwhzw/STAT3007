# coding=utf8

import os
import torch
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
from utils.dataloader import get_data_loader
from utils.data_prepare import prepare


class Train(object):
    def __init__(self,args):
        pretrain_name = 'bert-base-cased'
        if args.bert_path:
            pretrain_name = args.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        print(f"Tokenizer from:{pretrain_name}")
        train_conf = args.train_conf
        model_conf = args.model_conf
        self.model = BertClassifier(model_conf)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_num = model_conf.class_num
        self.model.to(self.device)
        self.lr = train_conf.lr
        self.tensorboard_log_path = getattr(train_conf,'log_path',None)
        if self.tensorboard_log_path:
            self.writer = SummaryWriter(train_conf.log_path)
        self.max_len = train_conf.max_len
        self.conf = args
        self.label_map = json.load(open(args.label_map_path))

    def train(self,train_path,vaild_path,epochs,model_path,lr,batch_size):
        train_loader = self.get_data_loader(train_path,batch_size=batch_size)
        valid_loader = self.get_data_loader(vaild_path,batch_size=batch_size)
        optimizer = AdamW(self.model.parameters(),
                          lr=lr,
                          eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        criterion = F.cross_entropy()
        best_f1 = 0.
        for epoch in tqdm(range(epochs)):
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

            acc,f1_score,cm,report = self.evaluate(train_loader)
            if self.tensorboard_log_path:
                self.writer.add_scalar('Train/F1',f1_score,epochs)
            print('\n\n')
            print(f"Following is validation metrics at epoch {epoch}")
            print(f"Accuracy of the model is {acc}")
            print(f"F1 score is : {f1_score}, and best_f1 is {best_f1}")
            print(f'Confusion matrix is: {cm}')
            print(report)
            #TODO add train history and train history path

        if self.tensorboard_log_path:
            self.writer.flush()
            self.writer.close()

    def train_epoch(self,train_loader,optimizer,scheduler,criterion,epoch):
        pass

    def evaluate(self,_loader):
        pass

    def get_data_loader(self,f_path,batch_size):
        np.random.seed(14)
        texts, labels = prepare(f_path,self.label_map)
        ds = SentimentDataset(self.tokenizer, texts, labels, self.max_len)
        return dataloader.DataLoader(ds, batch_size=batch_size, num_workers=self.conf.num_workers, shuffle=True)