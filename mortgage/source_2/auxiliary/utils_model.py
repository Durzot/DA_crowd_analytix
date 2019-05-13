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


class CustomGridSearch(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, param_grid, scoring, cv, verbose):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        
    def fit(self, X, y):
        self.cv_results_= {}
        self.best_params_ = {}
        splits = []
        for idx_train, idx_test in self.cv.split(X, y):
            splits.append((idx_train, idx_test))
        
        n_splits = self.cv.get_n_splits()
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        params_combinations = list(itertools.product(*(param_values)))
        n_combinations = len(params_combinations)
        
        for i in range(n_splits):
            self.cv_results_['split%d_train_score'%i] = np.zeros(n_combinations)
            self.cv_results_['split%d_test_score'%i] = np.zeros(n_combinations)
        self.cv_results_['mean_train_score'] = np.zeros(n_combinations)
        self.cv_results_['std_train_score'] = np.zeros(n_combinations)
        self.cv_results_['mean_test_score'] = np.zeros(n_combinations)
        self.cv_results_['std_test_score'] = np.zeros(n_combinations)
        self.cv_results_['params'] = []
        
        st_time = time.time()
        best_mean_test_score = 0
        self.best_params_ = {}
        self.best_estimator_ = {}
        
        if type(self.estimator) == 'sklearn.pipeline.Pipeline':
            raise ValueError("GridSearch for pipeline not implemented yet")
        else:
            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())
            print(params_combinations)
            for j, params in enumerate(params_combinations):
                d_params = {k:v for (k,v) in zip(param_names, params)}
                str_d_params = '; '.join(["%s: %s" % (str(k), str(v)) for k,v in d_params.items()])
                if self.verbose > 0:
                    print("[fit %d/%d] %.3g s| params %s"%(j+1, n_combinations, time.time()-st_time, str_d_params))
                
                self.estimator = self.estimator.set_params(**d_params)
                scores_train = []
                scores_test = []
                for i, (idx_train, idx_test) in enumerate(splits):
                    self.estimator = self.estimator.fit(X.loc[idx_train], y.loc[idx_train])
                    score_train = self.scoring(self.estimator, X.loc[idx_train], y.loc[idx_train])
                    score_test = self.scoring(self.estimator, X.loc[idx_test], y.loc[idx_test])
                    scores_train.append(score_train)
                    scores_test.append(score_test)
                    self.cv_results_['split%d_train_score'%i][j] = score_train
                    self.cv_results_['split%d_test_score'%i][j] = score_test
                    if self.verbose > 1:
                        print("[cv %d/%d] | train score %.3g ; test score % .3g" % (i+1, n_splits, score_train, score_test))
                 
                if np.mean(scores_test) > best_mean_test_score:
                    best_mean_test_score = np.mean(scores_test)
                    self.best_params_ = d_params
                    self.best_estimator_ = clone(self.estimator)
                self.cv_results_['mean_train_score'][j] = np.mean(scores_train)
                self.cv_results_['std_train_score'][j] = np.std(scores_train)
                self.cv_results_['mean_test_score'][j] = np.mean(scores_test)
                self.cv_results_['std_test_score'][j] = np.std(scores_test)
                self.cv_results_['params'].append(d_params)
        return self

