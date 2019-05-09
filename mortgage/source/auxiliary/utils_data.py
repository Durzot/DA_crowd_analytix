# -*- coding: utf-8 -*-
"""
Usefuls functions for processing data
Python 3 virtual environment 3.7_pytorch_sk or base conda

@date: 5th May 2019
@author: Yoann Pradat
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import clone

class Ignore(BaseEstimator, TransformerMixin):
    def __init__(self, cols_ignore):
        self.cols_ignore=cols_ignore
    def fit(self, X, y):
        return self
    def transform(self, X):
        Xc = pd.DataFrame.copy(X)
        for x in self.cols_ignore:
            del Xc[x]
        return Xc
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
class Preprocesser(BaseEstimator, TransformerMixin):
    def __init__(self, cols_num, eps=1e-1):
        self.cols_num=cols_num
        self.esp = eps
    def fit(self, X, y):
        self.tds_upper = X.tds.quantile(0.99)
        self.gds_upper = X.gds.quantile(0.99)
        return self
    def transform(self, X):
        Xc = pd.DataFrame.copy(X)
        # Indicator of extreme values
        Xc.loc[:, 'tds_le_0'] = np.where(Xc.tds <= 0, 1, 0)
        Xc.loc[:, 'ltv_gt_80'] = np.where(Xc.ltv > 80, 1, 0)
        # Clip too extreme values of tds and tmgds
        Xc.loc[:, 'tds'] = Xc.tds.clip(lower=0, upper=self.tds_upper)
        Xc.loc[:, 'gds'] = Xc.gds.clip(lower=0, upper=self.gds_upper)
        # Log transform
        eps = 1e-1
        for x in self.cols_num:
            Xc.loc[:, x] = np.log10(Xc[x] + eps)
        # As gds and tds are very correlated (0.99) we will only keep tds and add feature tds - gds
        Xc.loc[:, 'tmgds'] = Xc.loc[:, 'tds'] - Xc.loc[:, 'gds']
        del Xc['gds']
        return Xc  

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, center=True, norm_ord=2):
        self.center=center
        self.norm_ord=norm_ord
    def fit(self, X, y):
        n, _ = X.shape
        self.centers = {}
        self.norms = {}
        for x in X:
            if X[x].dtype=='object' or set(X[x].unique()).issubset(set([0,1])):
                pass
            elif self.center:
                center = np.mean(X[x])
                norm = np.linalg.norm(X[x]-center)
                self.centers[x] = center
                self.norms[x] = norm/np.sqrt(n)
            else:
                norm = np.linalg.norm(X[x]-center, ord=norm_ord)
                self.norms[x] = norm/np.sqrt(n)
        return self
    def transform(self, X):
        Xc = pd.DataFrame.copy(X)
        for x in Xc:
            if Xc[x].dtype=='object' or set(X[x].unique()).issubset(set([0,1])):
                pass
            elif self.center:
                Xc.loc[:, x] = (Xc[x]-self.centers[x])/self.norms[x]
            else:
                Xc.loc[:, x] = Xc[x]/self.norms[x]
        return Xc
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class HotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_onehot, categories, other_lim=0.005):
        self.cols_onehot = cols_onehot
        self.categories = categories
        self.other_lim = other_lim
    def fit(self, X, y):
        n, _ = X.shape
        self.category_other = {}
        for x, category in zip(self.cols_onehot, self.categories):
            category_other = []
            x_counts = X[x].value_counts()
            for cat in category:
                if cat not in x_counts.index:
                    category_other.append(cat)
                elif x_counts[cat]/n < self.other_lim:
                    category_other.append(cat)
            self.category_other[x] = category_other
        Xc = pd.DataFrame.copy(X)
        for i, x in enumerate(self.cols_onehot):
            Xc.loc[Xc[x].isin(self.category_other[x]), x] = 'other'
        self.categories = [Xc[x].unique() for x in self.cols_onehot]
        self.onehotenc = OneHotEncoder(categories=self.categories)
        self.onehotenc.fit(Xc[self.cols_onehot])
        return self
    def transform(self, X):
        Xc = pd.DataFrame.copy(X)
        for x in self.cols_onehot:
            Xc.loc[Xc[x].isin(self.category_other[x]), x] = 'other'
        Xdummy = self.onehotenc.transform(Xc[self.cols_onehot])
        Xdummy = Xdummy.toarray()
        cols_Xdummy = ['']*Xdummy.shape[1]
        offset = 0
        for x, category in zip(self.cols_onehot, self.categories):
            for i, cat in enumerate(category):
                cols_Xdummy[offset + i] = '%s_%s' % (x, str(cat))
            offset += len(category)
            del Xc[x]
        Xdummy = pd.DataFrame(Xdummy, columns=cols_Xdummy).astype(int)
        Xdummy.index = index=Xc.index
        for x, category in zip(self.cols_onehot, self.categories):
            if 'other' in category:
                del Xdummy['%s_%s' % (x, 'other')]
            else:
                del Xdummy['%s_%s' % (x, str(category[0]))]
        Xc = pd.concat((Xc, Xdummy), axis=1)
        return Xc
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
class LabEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_cat, categories, other_lim=0.005):
        self.cols_cat = cols_cat
        self.categories = categories
        self.other_lim = other_lim
    def fit(self, X, y):
        n, _ = X.shape
        self.encoders = {}
        self.category_other = {}
        for x, category in zip(self.cols_cat, self.categories):
            category_other = []
            x_counts = X[x].value_counts()
            for cat in category:
                if cat not in x_counts.index:
                    category_other.append(cat)
                elif x_counts[cat]/n < self.other_lim:
                    category_other.append(cat)
            self.category_other[x] = category_other
            
        Xc = pd.DataFrame.copy(X)
        self.categories = []
        for i, x in enumerate(self.cols_cat):
            Xc.loc[Xc[x].isin(self.category_other[x]), x] = 'other'
            Xc.loc[:, x] = Xc.loc[:, x].astype(str)
            self.categories.append(Xc[x].unique())
            encoder = LabelEncoder()
            encoder.fit(Xc[x])
            self.encoders[x] = encoder
        return self
    def transform(self, X):
        Xc = pd.DataFrame.copy(X)
        for x in self.cols_cat:
            Xc.loc[Xc[x].isin(self.category_other[x]), x] = 'other'
            Xc.loc[:, x] = Xc.loc[:, x].astype(str)
            Xc.loc[:, x] = self.encoders[x].transform(Xc[x])
        return Xc
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
