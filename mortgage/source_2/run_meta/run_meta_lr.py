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
import argparse
import time

sys.path.append("./source_2")
from auxiliary.utils_model import *
from auxiliary.dataset import *

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

# ====================== LOAD TRAIN, XVAL, TEST IDS ====================== # 
mortgage_data = MortgageData(encoder="Lab")

# To get row ids
X_train = mortgage_data.X_train
X_xval = mortgage_data.X_xval
X_test = mortgage_data.X_test

df_btrain = pd.DataFrame({"Index test": X_train.reset_index().index.values})
df_bxval= pd.DataFrame({"Index xval": X_xval.index.values})
df_btest = pd.DataFrame({"Unique_ID": X_test.unique_id.values})

# Splits for cross-val of meta learner
bstrat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.random_state)

# ====================== MLPNETS PREDICTIONS ====================== # 
df_btrain_mlp = pd.DataFrame({"Index test": X_train.reset_index().index.values})
df_bxval_mlp = pd.DataFrame({"Index xval": X_xval.index.values})
df_btest_mlp = pd.DataFrame({"Unique_ID": X_test.unique_id.values})

model_type = "MLP_2"
path_pred  = "./predictions/%s" % model_type
model_names = ["MLPNet3", "MLPNet3Drop", "MLPNet4", "MLPNet4Drop"]
dropout_rates = [0.2, 0.4]
index_splits = [1, 2, 3, 4, 5]

for model_name in model_names:
    if "Drop" in model_name:
        for p in dropout_rates:
            model_name_p = model_name + "_" + str(p) 
            col_name = "%s_class_1" % model_name_p
            df_btrain.loc[:, col_name] = -1
            df_btest.loc[:, col_name] = -1
            
            list_pred_proba_split_xval = list()
            list_pred_proba_split_test = list()

            # On splits
            for index_split in index_splits:
                
                # On train data
                df_pred_split = pd.read_csv(os.path.join(path_pred, "%s_oof_split_%d.csv" % (model_name_p, index_split)))
                pred_proba_split = softmax(df_pred_split[["Output 0", "Output 1"]].values)
                df_pred_split.loc[:, col_name] = pred_proba_split[:, 1]
                df_btrain_mlp.loc[df_btrain_mlp['Index test'].isin(df_pred_split['Index test']), col_name] = df_pred_split[col_name].values

                # On xval data
                df_pred_split_xval = pd.read_csv(os.path.join(path_pred, "%s_xval_split_%d.csv" % (model_name_p, index_split)))
                pred_proba_split_xval = softmax(df_pred_split_xval[["Output 0", "Output 1"]].values)
                list_pred_proba_split_xval.append(pred_proba_split_xval[:, 1])

                # on test data
                df_pred_split_test = pd.read_csv(os.path.join(path_pred, "%s_test_split_%d.csv" % (model_name_p, index_split)))
                pred_proba_split_test = softmax(df_pred_split_test[["Output 0", "Output 1"]].values)
                list_pred_proba_split_test.append(pred_proba_split_test[:, 1])

            # On xval data
            pred_proba_xval = np.array(list_pred_proba_split_xval).mean(axis=0)
            df_bxval_mlp.loc[df_bxval_mlp['Index xval'].isin(df_pred_split_xval['Index xval']), col_name] = pred_proba_xval

            # On test data
            pred_proba_test = np.array(list_pred_proba_split_test).mean(axis=0)
            df_btest_mlp.loc[df_btest_mlp.Unique_ID.isin(df_pred_split_test.Unique_ID), col_name] = pred_proba_test

    else:
        col_name = "%s_class_1" % model_name
        df_btrain.loc[:, col_name] = -1
        df_btest.loc[:, col_name] = -1

        list_pred_proba_split_xval = list()
        list_pred_proba_split_test = list()

        # On splits
        for index_split in index_splits:
            # On train data
            df_pred_split = pd.read_csv(os.path.join(path_pred, "%s_oof_split_%d.csv" % (model_name, index_split)))
            pred_proba_split = softmax(df_pred_split[["Output 0", "Output 1"]].values)
            df_pred_split.loc[:, col_name] = pred_proba_split[:, 1]
            df_btrain_mlp.loc[df_btrain_mlp['Index test'].isin(df_pred_split['Index test']), col_name] = df_pred_split[col_name].values

            # On xval data
            df_pred_split_xval = pd.read_csv(os.path.join(path_pred, "%s_xval_split_%d.csv" % (model_name, index_split)))
            pred_proba_split_xval = softmax(df_pred_split_xval[["Output 0", "Output 1"]].values)
            list_pred_proba_split_xval.append(pred_proba_split_xval[:, 1])

            # on test data
            df_pred_split_test = pd.read_csv(os.path.join(path_pred, "%s_test_split_%d.csv" % (model_name, index_split)))
            pred_proba_split_test = softmax(df_pred_split_test[["Output 0", "Output 1"]].values)
            list_pred_proba_split_test.append(pred_proba_split_test[:, 1])


        # On xval data
        pred_proba_xval = np.array(list_pred_proba_split_xval).mean(axis=0)
        df_bxval_mlp.loc[df_bxval_mlp['Index xval'].isin(df_pred_split_xval['Index xval']), col_name] = pred_proba_xval

        # On test data
        pred_proba_test = np.array(list_pred_proba_split_test).mean(axis=0)
        df_btest_mlp.loc[df_btest_mlp.Unique_ID.isin(df_pred_split_test.Unique_ID), col_name] = pred_proba_test
