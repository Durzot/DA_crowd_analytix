# -*- coding: utf-8 -*-
"""
Classes to load data in dataframes or in a pytorch dataset
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../source/")
from auxiliary.utils_data import *
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
import torch.utils.data as data

class MortgageData(object):
    def __init__(self, train_file="./data/CAX_MortgageModeling_Train.csv", test_file="./data/CAX_MortgageModeling_Test.csv", 
                 encoder="Hot", n_splits=5, random_state=0, other_lim=0.005):
        self.n_splits = n_splits
        self.random_state = random_state
        self.other_lim = other_lim
        # Load data
        data_train = pd.read_csv(train_file) 
        data_test = pd.read_csv(test_file, usecols=lambda x: x not in ["RESULT"]) 
        
        # Lower case column names
        data_train.columns = list(map(str.lower, data_train.columns))
        data_test.columns = list(map(str.lower, data_test.columns))
        
        # Categorical features
        cols_cat = ['amortization', 'mortgage purpose', 'payment frequency', 'property type', 'term', 'age range',
                   'gender', 'income type', 'naics code']
        data_train[cols_cat] = data_train[cols_cat].astype('object')
        data_test[cols_cat] = data_test[cols_cat].astype('object')

        # Compartimentalize features
        self.X_train = data_train.iloc[:, :-1]  
        self.y_train = data_train.iloc[:, -1].map({"NOT FUNDED": 0, "FUNDED":1})
        self.X_test = data_test
        cols_ignore = ['unique_id', 'mortgage number', 'fsa']
        cols_cat = [x for x in self.X_train if self.X_train[x].dtype==object and x not in cols_ignore]
        cols_num = [x for x in self.X_train.columns if x not in cols_cat and x not in cols_ignore]

        # Make sure all categories of every categorical feature is here
        X_all = pd.concat((self.X_train, self.X_test), axis=0)
        categories = [X_all[x].unique() for x in cols_cat]
        del X_all

        if encoder=="Lab":
            self.tpipe = Pipeline([('ignore', Ignore(cols_ignore=cols_ignore)),
                                   ('preprocesser', Preprocesser(cols_num=cols_num)),
                                   ('scaler', Scaler()),
                                   ('encoder', LabEncoder(categories=categories,cols_cat=cols_cat,other_lim=other_lim))])
        elif encoder=="Hot":
            self.tpipe = Pipeline([('ignore', Ignore(cols_ignore=cols_ignore)),
                                   ('preprocesser', Preprocesser(cols_num=cols_num)),
                                   ('scaler', Scaler()),
                                   ('encoder', HotEncoder(categories=categories,cols_onehot=cols_cat,other_lim=other_lim))])
        else:
            raise ValueError("Please choose between 'Hot' and 'Lab' for encoding of categorical variables.")
        
        # Transform the data
        self.X_ttrain = self.tpipe.fit_transform(self.X_train, self.y_train)
        self.X_ttest = self.tpipe.transform(self.X_test)
        self.n_input = self.X_ttrain.shape[1]
            
        # Resample for balancing data. Used only if prescribed
        categorical_features = [x for x in self.X_ttrain.columns if any([s in x for s in cols_cat])]
        categorical_features = [self.X_ttrain.columns.get_loc(x) for x in categorical_features]
        self.sm = SMOTENC(categorical_features=categorical_features, 
                          sampling_strategy='auto',
                          random_state=random_state,
                          k_neighbors=5,
                          n_jobs=1)

        self.strat_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.splits = []
        for idx_train, idx_test in self.strat_cv.split(self.X_ttrain, self.y_train):
            self.splits.append((idx_train, idx_test))

    def resample(self, index=None):
        if index is None:
            self.X_ttrain, self.y_train = self.sm.fit_resample(self.X_ttrain, self.y_train)
        else:
            # Split before resampling
            idx_train, idx_test = self.splits[index]
            self.X_tttrain, self.y_ttrain = self.X_ttrain.iloc[idx_train], self.y_train.iloc[idx_train]
            self.X_tttrain, self.y_ttrain = self.sm.fit_resample(self.X_tttrain, self.y_ttrain)
            self.X_tttrain = pd.DataFrame(self.X_tttrain, columns=self.X_ttrain.columns)
            self.y_ttrain = pd.Series(self.y_ttrain)
            self.X_tttest, self.y_ttest = self.X_ttrain.iloc[idx_test], self.y_train.iloc[idx_test]
        self.resampled = True
        return self

    def get_train(self, index=None):
        if index is None:
            return self.X_ttrain, self.y_train
        else:
            if self.resampled:
                return self.X_tttrain, self.X_tttest, self.y_ttrain, self.y_ttest
            else:
                idx_train, idx_test = self.splits[index]
                self.X_tttrain, self.y_ttrain = self.X_ttrain.iloc[idx_train], self.y_train.iloc[idx_train]
                self.X_tttest, self.y_ttest = self.X_ttrain.iloc[idx_test], self.y_train.iloc[idx_test]
                return self.X_tttrain, self.X_tttest, self.y_ttrain, self.y_ttest

    def get_test(self):
        return self.X_ttest

class Mortgage(data.Dataset):
    def __init__(self, mortgage_data, train=True, index_split=0):
        self.mortgage_data = mortgage_data
        self.train = train
        self.index_split = index_split
   
        X_tttrain, X_tttest, y_ttrain, y_ttest = self.mortgage_data.get_train(index_split)

        if self.train:
            self.X = X_tttrain
            self.y = y_ttrain
        else:
            self.X = X_tttest
            self.y = y_ttest

    def __getitem__(self, index):
        return self.X.iloc[index].values, self.y.iloc[index]

    def __len__(self):
        return self.X.shape[0]

class MortgageTest(data.Dataset):
    def __init__(self, mortgage_data):
        self.mortgage_data = mortgage_data        
        self.X = mortgage_data.get_test()

    def __getitem__(self, index):
        return self.X.iloc[index].values

    def __len__(self):
        return self.X.shape[0]
