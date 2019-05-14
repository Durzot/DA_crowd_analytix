# -*- coding: utf-8 -*-
"""
Usefuls functions for models
Python 3 virtual environment 3.7_pytorch_sk or base conda

@date: 5th May 2019
@author: Yoann Pradat
"""

import numpy as np
import pandas as pd
import itertools
import time
import torch
from sklearn.base import BaseEstimator, TransformerMixin

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
def f1_macro(y_pred, y_true):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    precisions = {}
    recalls = {}
    f1s = {}
    for cl in np.unique(y_true):
        if sum(y_pred==cl)==0:
            precision = 0
        else:
            precision = sum((y_pred==cl) & (y_true==cl))/sum(y_pred==cl)
        if sum(y_true==cl)==0:
            print("Warning! Ill-defined f1-score as no label of the class is in y_true")
            recall = 0
        else:
            recall = sum((y_pred==cl) & (y_true==cl))/sum(y_true==cl)
        
        if recall+precision == 0:
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall + precision)
        precisions[cl] = precision
        recalls[cl] = recall
        f1s[cl] = f1
    return precisions, recalls, f1s

class AccuracyValueMeter(object):
    """Computes and stores the predictions per category"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.sum = {i: [0 for _ in range(self.n_classes)] for i in range(self.n_classes)}
        self.pred = []        
        self.label = []
        self.precisions = {}
        self.recalls = {}
        self.f1s = {}
        self.f1_macro = 0
        self.count = 0

    def update(self, pred, label, size=1):
        for p, l in zip(pred, label):
            self.sum[p][l] += 1
        self.pred += list(pred)
        self.label += list(label)
        self.count += size
        self.precisions, self.recalls, self.f1s = f1_macro(self.pred, self.label)
        self.f1_macro = np.mean(list(self.f1s.values()))

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X), axis=1).reshape(-1, 1)
