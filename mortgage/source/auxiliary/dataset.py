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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import torch.utils.data as data

class MortgageData(object):
    def __init__(self, train_file="./data/CAX_MortgageModeling_Train.csv", test_file="./data/CAX_MortgageModeling_Test.csv"):
        # Load data
        data_train = pd.read_csv(train_file) 
        data_test = pd.read_csv(test_file, usecols=lambda x: x not in ["RESULT"]) 
        
        # Lower case column names
        data_train.columns = list(map(str.lower, data_train.columns))
        data_test.columns = list(map(str.lower, data_test.columns))
        
        # Categorical features
        cols_cat = ['amortization', 'rate', 'mortgage purpose', 'payment frequency', 'property type', 'term', 'age range',
                   'gender', 'income type', 'naics code']
        data_train[cols_cat] = data_train[cols_cat].astype('object')
        data_test[cols_cat] = data_test[cols_cat].astype('object')

        # Compartimentalize features
        X_train = data_train.iloc[:, :-1]        
        X_test = data_test
        cols_ignore = ['unique_id', 'mortgage number', 'fsa']
        cols_onehot = [x for x in X_train if X_train[x].dtype==object and x not in cols_ignore]
        cols_num = [x for x in X_train.columns if x not in cols_onehot and x not in cols_ignore]

        # Make sure all categories of every categorical feature is here
        X_all = pd.concat((X_train, X_test), axis=0)
        categories = [X_all[x].unique() for x in cols_onehot]
        del X_all

        # Transform the data
        tpipe = Pipeline([('ignore', Ignore(cols_ignore=cols_ignore)),
                          ('preprocesser', Preprocesser(cols_num=cols_num)),
                          ('scaler', Scaler()),
                          ('encoder', Encoder(categories=categories, cols_onehot=cols_cat))])
        
        self.y_train = data_train.iloc[:, -1].map({"NOT FUNDED": 0, "FUNDED":1})
        self.X_ttrain = tpipe.fit_transform(X_train, self.y_train)
        self.X_ttest = tpipe.transform(X_test)
        self.n_input = self.X_ttrain.shape[1]

    def get_train(self):
        return self.X_ttrain, self.y_train

    def get_test(self):
        return self.X_ttest

class Mortgage(data.Dataset):
    def __init__(self, mortgage_data, train=True, test_size=0.2, random_state=0):
        self.mortgage_data = mortgage_data
        self.test_size = test_size
        self.train = train
        self.random_state = random_state

        # Get transformed data       
        X_ttrain, y_train = self.mortgage_data.get_train()

        # Split between train and x-val
        X_tttrain, X_tttval, y_ttrain, y_ttval = train_test_split(X_ttrain, y_train, 
                                                                  stratify=y_train,
                                                                  test_size=self.test_size, 
                                                                  random_state=self.random_state) 
        if self.train:
            self.X = X_tttrain
            self.y = y_ttrain
        else:
            self.X = X_tttval
            self.y = y_ttval

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
