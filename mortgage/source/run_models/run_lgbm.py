# -*- coding: utf-8 -*-
"""
Functions to train models on the train set
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import warnings
import time
import itertools
import time

sys.path.append("./source")
from auxiliary.utils_data import *
from auxiliary.utils_model import *
from auxiliary.dataset import *

from sklearn.externals import joblib
from sklearn.metrics import f1_score
import lightgbm as lgb

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--model_type', type=str, default='LGBM_1',  help='type of model')
parser.add_argument('--boosting_type', type=str, default='gbdt', help='boosting type')
parser.add_argument('--num_leaves', type=int, default=30, help='num_leaves')
parser.add_argument('--random_state', type=int, default=0, help='random state for the model')
parser.add_argument('--verbose', type=int, default=2, help='verbose gridsearch')
opt = parser.parse_args()
st_time = time.time()

# ======================== FUNCTIONS FOR GRIDSEARCH ======================== #

def f1_macro_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.where(y_pred > 0.7, 1, 0)
    return 'f1_macro', f1_score(y_true, y_pred, average='macro'), True

class GBMGridSearch(BaseEstimator, TransformerMixin):
    def __init__(self, cols_cat, param_grid, scoring, cv, verbose, path_model, model_name):
        self.cols_cat = cols_cat
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.path_model = path_model
        self.model_name = model_name
        
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
        self.best_score_ = 0
        self.best_params_ = {}
        self.best_estimator_ = {}
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        for j, params in enumerate(params_combinations):
            d_params = {k:v for (k,v) in zip(param_names, params)}
            str_d_params = '; '.join(["%s: %s" % (str(k), str(v)) for k,v in d_params.items()])
            if self.verbose > 0:
                print("[fit %d/%d] %.3g s| params %s"%(j+1, n_combinations, time.time()-st_time, str_d_params))

            scores_train = []
            scores_test = []
            for i, (idx_train, idx_test) in enumerate(splits):
                lgb_train = lgb.Dataset(X.loc[idx_train], y.loc[idx_train], feature_name='auto', categorical_feature=self.cols_cat, free_raw_data=False)
                lgb_test = lgb.Dataset(X.loc[idx_test], y.loc[idx_test], feature_name='auto', categorical_feature=self.cols_cat, free_raw_data=False)

                gbm = lgb.train(d_params,
                                lgb_train,
                                feval=f1_macro_eval,
                                valid_sets=[lgb_test],               
                                valid_names=['test'],
                                verbose_eval=False)

                _, score_train, _ = self.scoring(gbm.predict(X.loc[idx_train]), lgb_train)
                _, score_test, _ = self.scoring(gbm.predict(X.loc[idx_test]), lgb_test)
                scores_train.append(score_train)
                scores_test.append(score_test)
                self.cv_results_['split%d_train_score'%i][j] = score_train
                self.cv_results_['split%d_test_score'%i][j] = score_test
                if self.verbose > 1:
                    print("[cv %d/%d] | train score %.3g ; test score % .3g" % (i+1, n_splits, score_train, score_test))

            if np.mean(scores_test) > self.best_score_:
                self.best_score_ = np.mean(scores_test)
                self.best_params_ = d_params
                self.best_estimator_ = gbm
                joblib.dump(gbm, os.path.join(self.path_model, '%s_grid.best_estimator.pkl' % self.model_name))
                
            self.cv_results_['mean_train_score'][j] = np.mean(scores_train)
            self.cv_results_['std_train_score'][j] = np.std(scores_train)
            self.cv_results_['mean_test_score'][j] = np.mean(scores_test)
            self.cv_results_['std_test_score'][j] = np.std(scores_test)
            self.cv_results_['params'].append(d_params)
        return self



# ========================== TRAINING AND TEST DATA ========================== #
mortgage_data = MortgageData(encoder="Lab")
X_ttrain, y_train = mortgage_data.get_train()
X_ttest = mortgage_data.get_test()

categorical_features = mortgage_data.categorical_features
cols_cat = list(X_ttrain.iloc[:, categorical_features].columns)
strat_cv = mortgage_data.strat_cv

# ========================== THE MODEL ========================== #
param_grid = {'boosting_type': [opt.boosting_type],
              'objective': ['binary'],
              'metric': ['binary_logloss'],
              'num_leaves': [opt.num_leaves],
              'learning_rate': [0.05],
              'feature_fraction': [0.7],
              'bagging_fraction': [0.9],
              'bagging_freq': [5],
              'num_boost_round': [500],
              'early_stopping_rounds': [25],
              'verbose': [0]}

# ====================== DEFINE STUFF FOR LOGS ====================== #
path_pred  = "./predictions/%s" % opt.model_type
path_log  = "./log/%s" % opt.model_type
path_model = "./trained_models/%s" % opt.model_type

if not os.path.exists(path_pred):
    os.mkdir(path_pred)
if not os.path.exists(path_log):
    os.mkdir(path_log)
if not os.path.exists(path_model):
    os.mkdir(path_model)
    
model_name = "lgb_%s_%s" % (opt.boosting_type, opt.num_leaves)
file_log = os.path.join(path_log, '%s.txt' % (model_name))

# ========================== GRIDSEARCH ========================== #
grid = GBMGridSearch(cols_cat=cols_cat, 
                     param_grid=param_grid, 
                     scoring=f1_macro_eval, 
                     cv=strat_cv, 
                     verbose=2,
                     path_model=path_model,
                     model_name=model_name)

grid = grid.fit(X_ttrain, y_train)
best_estimator = grid.best_estimator_

# Save the result of the gridsearch
joblib.dump(grid, os.path.join(path_model, '%s_grid.pkl' % model_name))

# ===================== FILL LOG AND SAVE =========================== # 
print("="*80)
print("Best params %s\n" % grid.best_params_)
print("Best score %.5g" % grid.best_score_)
print("="*80)

with open(file_log, 'a') as log:
    log.write(str(opt) + '\n\n')
    log.write("="*80)
    log.write('\n')
    log.write("Best params %s\n" % grid.best_params_)
    log.write("Best score %.5g\n" % grid.best_score_)
    log.write("="*80)
    log.write('\n')

    for i, (idx_train, idx_test) in enumerate(mortgage_data.splits):
        print("Predicting on split [%d/%d] ..." % (i+1, mortgage_data.n_splits))

        # Fit the estimator with best parameters on the split
        best_estimator = best_estimator.refit(X_ttrain.iloc[idx_train], y_train.iloc[idx_train])

        # Evaluate the score and save cr
        y_pred_train = np.where(best_estimator.predict(X_ttrain.iloc[idx_train]) > 0.7, 1, 0)
        y_pred_test = np.where(best_estimator.predict(X_ttrain.iloc[idx_test]) > 0.7, 1, 0)

        log.write("\n\nSplit [%d/%d]\n" % (i+1, mortgage_data.n_splits))
        recalls, precisions, f1s = f1_macro(y_pred_train, y_train.iloc[idx_train])
        for k in recalls.keys():
            print("Train class %d | precision %.4g; recall %.4g; f1 %.4g" % (k, recalls[k], precisions[k], f1s[k]))
            log.write("Train class %d | precision %.4g; recall %.4g; f1 %.4g\n" % (k, recalls[k], precisions[k], f1s[k]))
        recalls, precisions, f1s = f1_macro(y_pred_test, y_train.iloc[idx_test])
        for k in recalls.keys():
            print("Test class %d | precision %.4g; recall %.4g; f1 %.4g" % (k, recalls[k], precisions[k], f1s[k]))
            log.write("Test class %d | precision %.4g; recall %.4g; f1 %.4g\n" % (k, recalls[k], precisions[k], f1s[k]))
        
        # Save the estimator on the split
        joblib.dump(best_estimator, os.path.join(path_model, '%s_split_%d.pkl' % (model_name, i+1)))

        # Save prediction on out-of-fold split
        y_pred_test_proba = best_estimator.predict(X_ttrain.iloc[idx_test])
        df_pred_test = pd.DataFrame(np.concatenate((idx_test.reshape(-1,1), y_pred_test.reshape(-1,1),
                                                    y_pred_test_proba.reshape(-1, 1)), axis=1))
        df_pred_test.columns = ["Index test", "Prediction", "Proba class 1"]
        df_pred_test.to_csv(os.path.join(path_pred, "%s_split_%d.csv" % (model_name, i+1)))
        print("Finished!\n")

# ===================== REFIT AND PREDICT ON TEST =========================== #
print("Predicting on the test set...")

# Refit on the complete train set
best_estimator = best_estimator.refit(X_ttrain, y_train)
X_test = mortgage_data.X_test

# Save prediction on test 
y_pred_test = np.where(best_estimator.predict(X_ttest) > 0.7, 1 ,0)
y_pred_test_proba = best_estimator.predict(X_ttest)
df_pred_test = pd.DataFrame(np.concatenate((X_test.unique_id.values.reshape(-1,1), 
                                            y_pred_test.reshape(-1,1), y_pred_test_proba.reshape(-1,1)), axis=1))
df_pred_test.columns = ["Unique_ID", "Prediction", "Proba class 1"]
df_pred_test.to_csv(os.path.join(path_pred, "%s_test.csv" % (model_name)))

y_pred_test = np.where(y_pred_test==0, "NOT FUNDED", "FUNDED")
submission_test = pd.DataFrame({'Unique_ID': X_test['unique_id'], 'Result_Predicted': y_pred_test})
submission_file = pd.read_csv("./data/CAX_MortgageModeling_SubmissionFormat.csv")
submission_file = submission_file[['Unique_ID']].merge(submission_test, how='left', on=['Unique_ID'])
submission_file.to_csv(os.path.join(path_pred, "%s_submission.csv" % model_name), index=False)

with open(file_log, 'a') as log:
    log.write('\n\n')
    log.write("="*80)
    log.write("\nRunning time script %.6g sec\n" % (time.time()-st_time))
    log.write("="*80)

print("Finished!\n")
print("Running time script %.5g sec" % (time.time()-st_time))


