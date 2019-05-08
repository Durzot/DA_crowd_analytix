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

sys.path.append("./source")
from auxiliary.utils_data import *
from auxiliary.utils_model import *
from auxiliary.dataset import *

from sklearn.externals import joblib
import lightgbm as lgb

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--model_type', type=str, default='LGBM_1',  help='type of model')
parser.add_argument('--criterion', type=str, default='gini', help='criterion')
parser.add_argument('--max_depth', type=int, default=15, help='max_depth')
parser.add_argument('--max_features', type=str, default='sqrt', help='max_features')
parser.add_argument('--random_state', type=int, default=0, help='random state for the model')
parser.add_argument('--n_jobs', type=int, default=2, help='number of jobs gridsearch')
parser.add_argument('--verbose', type=int, default=2, help='verbose gridsearch')
opt = parser.parse_args()
st_time = time.time()

# ========================== TRAINING AND TEST DATA ========================== #
mortgage_data = MortgageData(encoder="Lab")
X_ttrain, y_train = mortgage_data.get_train()
X_ttest = mortgage_data.get_test()
# ========================== THE MODEL ========================== #
param_grid = {'boosting_type': ['gbdt'],
              'objective': ['binary'],
              'metric': ['binary_logloss'],
              'num_leaves': [19, 25, 31, 40],
              'learning_rate': [0.05],
              'feature_fraction': [0.7, 0.8, 0.9],
              'bagging_fraction': [0.7, 0.8, 0.9],
              'bagging_freq': [5],
              'num_boost_round': [200, 400. 600],
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
    
model_name = "rf_%s_%s_%s" % (opt.criterion, opt.max_depth, opt.max_features)
file_log = os.path.join(path_log, '%s.txt' % (model_name))

# ========================== GRIDSEARCH ========================== #
grid = GridSearchCV(estimator=estimator,
                    param_grid=param_grid,
                    scoring='f1_macro', 
                    cv=mortgage_data.strat_cv,
                    return_train_score=True,
                    verbose=opt.verbose,
                    n_jobs=opt.n_jobs)

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
        best_estimator.fit(X_ttrain.iloc[idx_train], y_train.iloc[idx_train])

        # Evaluate the score and save cr
        y_pred_train = best_estimator.predict(X_ttrain.iloc[idx_train])
        y_pred_test = best_estimator.predict(X_ttrain.iloc[idx_test])

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
        y_pred_test_proba = best_estimator.predict_proba(X_ttrain.iloc[idx_test])
        df_pred_test = pd.DataFrame(np.concatenate((idx_test.reshape(-1,1), y_pred_test.reshape(-1,1), y_pred_test_proba), axis=1))
        df_pred_test.columns = ["Index test", "Prediction", "Proba class 0", "Proba class 1"]
        df_pred_test.to_csv(os.path.join(path_pred, "%s_split_%d.csv" % (model_name, i+1)))
        print("Finished!\n")

# ===================== REFIT AND PREDICT ON TEST =========================== #
print("Predicting on the test set...")

# Refit on the complete train set
best_estimator.fit(X_ttrain, y_train)
X_test = mortgage_data.X_test

# Save prediction on test 
y_pred_test = best_estimator.predict(X_ttest)
y_pred_test_proba = best_estimator.predict_proba(X_ttest)
df_pred_test = pd.DataFrame(np.concatenate((X_test.unique_id.values.reshape(-1,1), 
                                            y_pred_test.reshape(-1,1), y_pred_test_proba), axis=1))
df_pred_test.columns = ["Unique_ID", "Prediction", "Proba class 0", "Proba class 1"]
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


