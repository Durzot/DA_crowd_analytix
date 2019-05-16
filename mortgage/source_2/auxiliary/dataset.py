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

sys.path.append("../source_2/")
from auxiliary.utils_data import *
from sklearn.model_selection import StratifiedKFold, train_test_split
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
                   'gender', 'income type', 'naics code', 'fsa']
        data_train[cols_cat] = data_train[cols_cat].astype('object')
        data_test[cols_cat] = data_test[cols_cat].astype('object')

        # Keep only district information
        data_train.loc[:,'fsa'] = data_train.fsa.apply(lambda x: x[0])
        data_test.loc[:,'fsa'] = data_train.fsa.apply(lambda x: x[0])

        # Remap label names
        data_train.iloc[:, -1] = data_train.iloc[:, -1].map({"NOT FUNDED": 0, "FUNDED":1})

        # Separate train in train and xval with xval used only for the purporse
        # of evaluating models. No training whatsoever will be done on it
        self.X_train, self.X_xval, self.y_train, self.y_xval = train_test_split(data_train.iloc[:, :-1],
                                                                                data_train.iloc[:, -1],
                                                                                stratify=data_train.iloc[:, -1],
                                                                                test_size=0.15,
                                                                                random_state=random_state)
        self.X_test = data_test

        # Compartimentalize features
        cols_ignore = ['unique_id', 'mortgage number']
        cols_cat = [x for x in self.X_train if self.X_train[x].dtype==object and x not in cols_ignore]
        cols_num = [x for x in self.X_train.columns if x not in cols_cat and x not in cols_ignore]

        # Make sure all categories of every categorical feature is here
        X_all = pd.concat((self.X_train, self.X_xval, self.X_test), axis=0)
        categories = [X_all[x].unique() for x in cols_cat]

        # Pipeline for transforming the data
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
        self.X_txval = self.tpipe.transform(self.X_xval)
        self.X_ttest = self.tpipe.transform(self.X_test)
        self.n_input = self.X_ttrain.shape[1]
            
        # Positions of categorical columns
        self.columns = self.X_ttrain.columns
        self.categorical_features = [self.columns.get_loc(x)  for x in self.columns if any([s in x for s in cols_cat])]
        
        # Fix splits for cross-validation of parameters once and for all
        self.strat_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.splits = []
        for index_train, index_test in self.strat_cv.split(self.X_ttrain, self.y_train):
            self.splits.append((index_train, index_test))

    def split(self, resample=False, index=None):
        self.sm = SMOTENC(categorical_features=self.categorical_features,
                          sampling_strategy='auto',
                          random_state=self.random_state,
                          k_neighbors=5,
                          n_jobs=1)

        if index is None:
            if resample:
                # If no index is provided transform the whole training set
                self.X_ttrain, self.y_train = self.sm.fit_resample(self.X_ttrain, self.y_train)
        else:
            # If index is provided transform the corresponding train split (on X_ttrain)
            # Be aware that the test split (on X_ttest) is not transformed for accurate
            # evaluation of performance on unbalanced dataset. As a consequence splitting 
            # has to occur before rebalancing

            index_train, index_test = self.splits[index]
            self.X_tttrain, self.y_ttrain = self.X_ttrain.iloc[index_train], self.y_train.iloc[index_train]
            self.X_tttest, self.y_ttest = self.X_ttrain.iloc[index_test], self.y_train.iloc[index_test]

            if resample:
                # Transform train split
                self.X_tttrain, self.y_ttrain = self.sm.fit_resample(self.X_tttrain, self.y_ttrain)

                # Match back to pandas data types
                self.X_tttrain = pd.DataFrame(self.X_tttrain, columns=self.columns)
                self.y_ttrain = pd.Series(self.y_ttrain)
            
        self.resampled = resample
        return self

    def get_train(self, index=None):
        if index is None:
            # If no index is provided provide the whole training set
            return self.X_ttrain, self.y_train
        else:
            # If index is provided provide train and test splits
            return self.X_tttrain, self.X_tttest, self.y_ttrain, self.y_ttest

    def get_xval(self):
        return self.X_txval, self.y_xval

    def get_test(self):
        return self.X_ttest

###################################
# Classes for create Pytorch loaders
###################################

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

class MortgageXval(data.Dataset):
    def __init__(self, mortgage_data):
        self.mortgage_data = mortgage_data        
        self.X, self.y = mortgage_data.get_xval()

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
