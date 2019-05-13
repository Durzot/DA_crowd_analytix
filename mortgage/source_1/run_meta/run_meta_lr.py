# -*- coding: utf-8 -*-
"""
Functions to train meta models on the predictions of base learners
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import numpy as np
import pandas as pd

import os
import sys

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb

# ================================ PARAMETERS ================================ # 
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--model_type', type=str, default='RandomForest_1',  help='type of model')
parser.add_argument('--max_depth', type=int, default=15, help='max_depth')
parser.add_argument('--max_features', type=str, default='sqrt', help='max_features')
parser.add_argument('--random_state', type=int, default=0, help='random state for the model')
parser.add_argument('--n_jobs', type=int, default=2, help='number of jobs gridsearch')
parser.add_argument('--verbose', type=int, default=2, help='verbose gridsearch')
opt = parser.parse_args()
st_time = time.time()

# ====================== LOAD BASE LEARNER PREDICTIONS ====================== # 
mortgage_data = MortgageData(encoder="Lab")

# To get row ids
X_train = mortgage_data.X_train
X_test = mortgage_data.X_test

df_btrain = pd.DataFrame({"Index test": X_train.index.values})
df_btest = pd.DataFrame({"Unique_ID": X_test.unique_id.values})

# Splits for cross-val of meta learner
bstrat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.random_state)
