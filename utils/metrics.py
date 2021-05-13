# coding=utf8

import torch
import numpy as np
from sklearn.metrics import *


def cal_accuracy(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred,dim=1)
    pred_labels = pred_labels.cpu().numpy()
    acc = accuracy_score(y_true, pred_labels)
    return acc


def cal_recall(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred,dim=1)
    pred_labels = pred_labels.cpu().numpy()
    recall = recall_score(y_true, pred_labels, average="macro")
    return recall


def cal_f1(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred,dim=1)
    pred_labels = pred_labels.cpu().numpy()
    f1 = f1_score(y_true, pred_labels, average="macro")
    return f1


def cal_cm(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    pred_labels = torch.argmax(y_pred,dim=1)
    pred_labels = pred_labels.cpu().numpy()
    cm = confusion_matrix(y_true, pred_labels)
    return cm


def get_classification_report(y_true, y_pred, labels, target_names, digits = 4):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    y_true = np.squeeze(y_true)
    pred_labels = torch.argmax(y_pred, dim=1)
    pred_labels = pred_labels.cpu().numpy()
    report = classification_report(y_true, pred_labels, labels=labels, target_names=target_names, digits=digits)
    return report