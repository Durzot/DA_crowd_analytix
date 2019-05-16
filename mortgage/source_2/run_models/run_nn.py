# -*- coding: utf-8 -*-
"""
Run the model on the test splits and also on the test data
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import time
import datetime

sys.path.append("./source_2")
from auxiliary.utils_data import *
from auxiliary.utils_model import *
from auxiliary.dataset import *
from models.models_nn import *

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--other_lim', type=float, default=0.005, help='threshold for categories gathering')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs to train for on split 0')
parser.add_argument('--n_epoch_other', type=int, default=10, help='number of epochs to retrain for on other splits')
parser.add_argument('--dropout_rate', type=float, default=None, help='dropout rate if applicable')
parser.add_argument('--model_type', type=str, default='MLP_3',  help='type of model')
parser.add_argument('--model_name', type=str, default='MLPNet3',  help='name of the model for log')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='adam',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=0.05,  help='learning rate')
parser.add_argument('--lr_other', type=float, default=0.01,  help='learning rate for other splits')
parser.add_argument('--lr_decay_fact', type=float, default=0.95,  help='decay factor in learning rate')
parser.add_argument('--momentum', type=float, default=0,  help='momentum (only SGD)')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
parser.add_argument('--random_state', type=int, default=0, help='random state for split of data')
opt = parser.parse_args()

# ========================== NETWORK AND OPTIMIZER ========================== #
mortgage_data = MortgageData(other_lim=opt.other_lim, encoder="Hot")
n_input = mortgage_data.n_input
n_splits = mortgage_data.n_splits

if opt.dropout_rate is not None:
    network = eval("%s(n_input=%d, n_classes=opt.n_classes, p=opt.dropout_rate)" % (opt.model_name, n_input))   
    opt.model_name = opt.model_name + "_" + str(opt.dropout_rate)
else:
    network = eval("%s(n_input=%d, n_classes=opt.n_classes)" % (opt.model_name, n_input))

if opt.cuda:
    network.cuda()

def get_optimizer(optimizer, lr, momentum=None):
    if opt.optimizer == "adam":
        return torch.optim.Adam(network.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        return torch.optim.SGD(network.parameters(), lr=opt.lr)
    else:
        raise ValueError("Please choose between 'adam' and 'sgd' for the --optimizer")

optimizer = get_optimizer(opt.optimizer, opt.lr, opt.momentum)

if opt.criterion == "cross_entropy":
    criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError("Please choose 'cross_entropy' for --criterion")

# ======================== DEFINE STUFF FOR LOGS ======================== #
path_model = './trained_models/%s' % opt.model_type
path_log = "./log/%s" % opt.model_type
path_pred  = "./predictions/%s" % opt.model_type

if not os.path.exists(path_model):
    os.mkdir(path_model)
if not os.path.exists(path_log):
    os.mkdir(path_log)
if not os.path.exists(path_pred):
    os.mkdir(path_pred)

# Log file for recording score on each split
log_file_split = os.path.join(path_log, '%s_split.txt' % (opt.model_name))
if not os.path.exists(log_file_split):
    with open(log_file_split, 'a') as log:
        log.write(str(opt) + '\n\n')
        log.write(str(network) + '\n')

# ========================== TRAINING AND TEST ========================== #

# Looping on splits for training models. A separate model is trained on each
# split so as to avoid be able to train meta models on the each of the out-of-fold
# splits. It is absolutely vital to avoid leakage of training in validation sets

for index_split in range(n_splits):
    model_name = "%s_%s" % (opt.model_name, index_split+1)
    if index_split == 0:
        # Initialize the weights
        network.apply(init_weight)
        network = network.float()
    else:
        model = os.path.join(path_model, '%s_1.pth' % opt.model_name)
        if opt.cuda:
            network.load_state_dict(torch.load(model))  
            network.cuda()
        else:
            network.load_state_dict(torch.load(model, map_location='cpu'))
        print("Weights from %s loaded" % model)

    # =================== LOAD THE DATA  =================== #
    mortgage_data = MortgageData(other_lim=opt.other_lim, encoder="Hot")
    mortgage_data = mortgage_data.split(resample=True, index=index_split)
    print("Class imbalance corrected on train index")

    # For ID in predictions
    X_xval = mortgage_data.X_xval
    X_test = mortgage_data.X_test
    
    # Training set in X_train, resampled
    dataset_train = Mortgage(mortgage_data=mortgage_data, train=True, index_split=index_split)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    
    # Test set in X_train
    dataset_test = Mortgage(mortgage_data=mortgage_data, train=False, index_split=index_split)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

    # Test set X_xval
    dataset_xval = MortgageXval(mortgage_data=mortgage_data)
    loader_xval = torch.utils.data.DataLoader(dataset_xval, batch_size=opt.batch_size, shuffle=False)

    # Test set X_test. Used for predictions
    dataset_xtest = MortgageTest(mortgage_data=mortgage_data)
    loader_xtest = torch.utils.data.DataLoader(dataset_xtest, batch_size=opt.batch_size, shuffle=False)

    print('training set size %d' % len(dataset_train))
    print('test set size %d' % len(dataset_test))
    print('xtest set size %d' % len(dataset_xtest))
    print('xval set size %d' % len(dataset_xval))

    n_batch_train = np.int(np.ceil(len(dataset_train)/opt.batch_size))
    n_batch_test = np.int(np.ceil(len(dataset_test)/opt.batch_size))

    # ====================== DEFINE STUFF FOR LOGS ====================== #
    log_file = os.path.join(path_log, '%s.txt' % model_name)
    if not os.path.exists(log_file):
        with open(log_file, 'a') as log:
            log.write(str(opt) + '\n\n')
            log.write(str(network) + '\n')
            log.write("train labels %s\n" % np.bincount(dataset_train.y))
            log.write("test labels %s\n\n" % np.bincount(dataset_test.y))
    
    log_train_file = "./log/%s/%s_train.csv" % (opt.model_type, model_name)
    log_test_file = "./log/%s/%s_test.csv" % (opt.model_type, model_name)
    
    if not os.path.exists(log_train_file): 
        df_logs_train = pd.DataFrame(columns=['model', 'random_state', 'date', 'n_epoch', 'lr', 'crit', 'optim', 'epoch', 
                                              'loss', 'f1_macro', 'pred_0_0', 'pred_0_1', 'precision_0', 'recall_0', 'f1_0',
                                              'pred_1_0', 'pred_1_1', 'precision_1', 'recall_1', 'f1_1'])
    else:
        df_logs_train = pd.read_csv(log_train_file, header='infer')
    
    if not os.path.exists(log_test_file):
        df_logs_test = pd.DataFrame(columns=['model', 'random_state', 'date', 'n_epoch', 'lr', 'crit', 'optim', 'epoch',
                                              'loss', 'f1_macro', 'pred_0_0', 'pred_0_1', 'precision_0', 'recall_0', 'f1_0',
                                              'pred_1_0', 'pred_1_1', 'precision_1', 'recall_1', 'f1_1'])
    else:
        df_logs_test = pd.read_csv(log_test_file, header='infer')
    
    value_meter_train = AccuracyValueMeter(opt.n_classes)
    value_meter_test = AccuracyValueMeter(opt.n_classes) 

    # ====================== LEARNING LOOP ====================== #
    if index_split == 0:
        lr = opt.lr
        n_epoch = opt.n_epoch
    else:
        lr = opt.lr_other
        n_epoch = opt.n_epoch_other

    for epoch in range(n_epoch):
        # TRAINING
        st_time = time.time()
        network.train()
        value_meter_train.reset()
        loss_train = 0
    
        # LEARNING RATE SCHEDULE
        if epoch > 0:
            lr *= opt.lr_decay_fact
            optimizer = get_optimizer(opt.optimizer, lr, opt.momentum)
    
        for batch, (data, label) in enumerate(loader_train):
            data = data.float()
            if opt.cuda:
                data, label = data.cuda(), label.cuda()
        
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, label)
            loss_train += loss
            loss.backward()
            optimizer.step()
        
            pred = output.cpu().data.numpy().argmax(axis=1)
            label = label.cpu().data.numpy()
            value_meter_train.update(pred, label, opt.batch_size)
        
            print('[train epoch %d/%d ; batch: %d/%d] train loss: %.3g' % (epoch+1, n_epoch, batch+1, n_batch_train, loss.item()))
    
        loss_train = float(loss_train.cpu())/n_batch_train
        dt = time.time()-st_time
        s_time = "%d min %d sec" % (dt//60, dt%60)
        
        print('='*40)
        print('[train epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, n_epoch, loss_train,
                                                                             value_meter_train.f1_macro, s_time))
        for i in range(opt.n_classes):
            print('cat %d: %s' % (i, value_meter_train.sum[i]))
        print('='*40)
        
        with open(log_file, 'a') as log:
            log.write('[train epoch %d/%d] | loss %.5g | f1_macro %.3g | time %s' % (epoch+1, n_epoch, loss_train,
                                                                                     value_meter_train.f1_macro, s_time) +  '\n')
            for i in range(opt.n_classes):
                log.write('cat %d: %s' % (i, value_meter_train.sum[i]) + '\n')
    
        row_train = {'model': model_name, 
                     'random_state': opt.random_state,
                     'date': get_time(),
                     'n_epoch': n_epoch,
                     'lr': opt.lr,
                     'crit': opt.criterion,
                     'optim': opt.optimizer,
                     'epoch': epoch+1,
                     'loss': np.round(loss_train, 4),
                     'f1_macro': np.round(value_meter_train.f1_macro, 4),
                     'precision_0': np.round(value_meter_train.precisions[0], 4),
                     'recall_0': np.round(value_meter_train.recalls[0], 4),
                     'f1_0': np.round(value_meter_train.f1s[0], 4),
                     'precision_1': np.round(value_meter_train.precisions[1], 4),
                     'recall_1': np.round(value_meter_train.recalls[1], 4),
                     'f1_1': np.round(value_meter_train.f1s[1], 4)}
        
        for i in range(opt.n_classes):
            for j in range(opt.n_classes):
                row_train["pred_%d_%d" % (i, j)] = value_meter_train.sum[i][j]
        
        df_logs_train = df_logs_train.append(row_train, ignore_index=True)
        df_logs_train.to_csv(log_train_file, header=True, index=False)
    
        # TEST
        st_time = time.time()
        network.eval()
        value_meter_test.reset()
        loss_test = 0
        
        with torch.no_grad():
            for batch, (data, label) in enumerate(loader_test):
                data = data.float()
                if opt.cuda:
                    data, label = data.cuda(), label.cuda()
                
                output = network(data)
                loss = criterion(output, label)
                loss_test += loss
        
                pred = output.cpu().data.numpy().argmax(axis=1)
                label = label.cpu().data.numpy()
                value_meter_test.update(pred, label, opt.batch_size)
        
        loss_test = float(loss_test.cpu())/n_batch_test
        dt = time.time()-st_time
        s_time = "%d min %d sec" % (dt//60, dt%60)
        
        print('='*40)
        print('[test epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, n_epoch, loss_test,
                                                                            value_meter_test.f1_macro, s_time))
        for i in range(opt.n_classes):
            print('cat %d: %s' % (i, value_meter_test.sum[i]))
        print('='*40)
        
        with open(log_file, 'a') as log:
            log.write('[test epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, n_epoch, loss_test,
                                                                                    value_meter_test.f1_macro, s_time) + '\n')
            for i in range(opt.n_classes):
                log.write('cat %d: %s' % (i, value_meter_test.sum[i]) + '\n')
        
        row_test = {'model': model_name, 
                    'random_state': opt.random_state,
                    'date': get_time(),
                    'n_epoch': n_epoch,
                    'lr': opt.lr,
                    'crit': opt.criterion,
                    'optim': opt.optimizer,
                    'epoch': epoch+1,
                    'loss': np.round(loss_test,4), 
                    'f1_macro': np.round(value_meter_test.f1_macro,4),
                    'precision_0': np.round(value_meter_test.precisions[0],4),
                    'recall_0': np.round(value_meter_test.recalls[0],4),
                    'f1_0': np.round(value_meter_test.f1s[0],4),
                    'precision_1': np.round(value_meter_test.precisions[1],4),
                    'recall_1': np.round(value_meter_test.recalls[1],4),
                    'f1_1': np.round(value_meter_test.f1s[1],4)}
        
        for i in range(opt.n_classes):
            for j in range(opt.n_classes):
                row_test["pred_%d_%d" % (i, j)] = value_meter_test.sum[i][j]
        
        df_logs_test = df_logs_test.append(row_test, ignore_index=True)
        df_logs_test.to_csv(log_test_file, header=True, index=False)
        
        print("Saving net")
        torch.save(network.state_dict(), os.path.join(path_model, '%s.pth' % model_name))

    # In here we make record predictions of the trained model on three different datasets
    # 1. On the index_test of X_ttrain i.e the out-of-fold split
    # 2. On the xval dataset whose purpose is only to evaluate the model
    # 3. On the test dataset used to make predictions

    # ====================== TESTING LOOP OOF ====================== #
    network.eval()
    with torch.no_grad():
        list_output = []
        list_pred = []
        list_label = []
        for data, label in loader_test:
            data = data.float()
            if opt.cuda:
                data = data.cuda()
            output = network(data).cpu().data.numpy()
            pred = output.argmax(axis=1)
    
            list_output.append(output)
            list_pred.append(pred)
            list_label.append(label)
    
    output = np.concatenate(list_output, axis=0)
    pred = np.concatenate(list_pred, axis=0)
    label = np.concatenate(list_label, axis=0)
    
    # Evaluate the score and save cr
    with open(log_file_split, 'a') as log:
        log.write("\n")
        log.write("="*40)
        log.write("\n")
        log.write("OUT-OF_FOLD\n")
        log.write("\n\nSplit [%d/%d]\n" % (index_split+1, n_splits))
        
        print("="*40)
        print("OUT-OF_FOLD")
        print("\nSplit [%d/%d]" % (index_split+1, n_splits))

        recalls, precisions, f1s = f1_macro(pred, label)
        for k in recalls.keys():
            log.write("Test class %d | precision %.4g; recall %.4g; f1 %.4g\n" % (k, recalls[k], precisions[k], f1s[k]))
            print("Test class %d | precision %.4g; recall %.4g; f1 %.4g" % (k, recalls[k], precisions[k], f1s[k]))
    
    index_train, index_test = mortgage_data.splits[index_split]
    df_pred_oof = pd.DataFrame(np.concatenate((index_test.reshape(-1,1), pred.reshape(-1,1), output), axis=1))
    df_pred_oof.columns = ["Index test", "Prediction", "Output 0", "Output 1"]
    df_pred_oof.to_csv(os.path.join(path_pred, "%s_oof.csv" % model_name))
    print("Finished test on oof!\n")

    # ====================== TESTING LOOP XVAL ====================== #
    network.eval()
    with torch.no_grad():
        list_output = []
        list_pred = []
        list_label = []
        for data, label in loader_xval:
            data = data.float()
            if opt.cuda:
                data = data.cuda()
            output = network(data).cpu().data.numpy()
            pred = output.argmax(axis=1)
    
            list_output.append(output)
            list_pred.append(pred)
            list_label.append(label)
    
    output = np.concatenate(list_output, axis=0)
    pred = np.concatenate(list_pred, axis=0)
    label = np.concatenate(list_label, axis=0)

    with open(log_file_split, 'a') as log:
        log.write("\n")
        log.write("="*40)
        log.write("\n")
        log.write("XVAL\n")
        log.write("\n\nSplit [%d/%d]\n" % (index_split+1, n_splits))
        
        print("="*40)
        print("XVAL")
        print("\nSplit [%d/%d]" % (index_split+1, n_splits))

        recalls, precisions, f1s = f1_macro(pred, label)
        for k in recalls.keys():
            log.write("Test class %d | precision %.4g; recall %.4g; f1 %.4g\n" % (k, recalls[k], precisions[k], f1s[k]))
            print("Test class %d | precision %.4g; recall %.4g; f1 %.4g" % (k, recalls[k], precisions[k], f1s[k]))
    
    index_xval = X_xval.index.values
    df_pred_xval = pd.DataFrame(np.concatenate((index_xval.reshape(-1,1), pred.reshape(-1,1), output), axis=1))
    df_pred_xval.columns = ["Index xval", "Prediction", "Output 0", "Output 1"]
    df_pred_xval.to_csv(os.path.join(path_pred, "%s_xval.csv" % model_name))
    print("Finished test on xval!\n")

    # ====================== TESTING LOOP XTEST ====================== #
    network.eval()
    with torch.no_grad():
        list_output = []
        list_pred = []
        for data in loader_xtest:
            data = data.float()
            if opt.cuda:
                data = data.cuda()
            output = network(data).cpu().data.numpy()
            pred = output.argmax(axis=1)
        
            list_output.append(output)
            list_pred.append(pred)

    output = np.concatenate(list_output, axis=0)
    pred = np.concatenate(list_pred, axis=0).reshape(-1, 1)
    unique_id = X_test.unique_id.values.reshape(-1,1)

    df_pred_xtest = pd.DataFrame(np.concatenate((unique_id, pred, output), axis=1))
    df_pred_xtest.columns = ["Unique_ID", "Prediction", "Output 0", "Output 1"]
    df_pred_xtest.to_csv(os.path.join(path_pred, "%s_test.csv" % model_name))
    print("Finished test on xtest!\n")


