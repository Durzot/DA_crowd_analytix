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
import time
import datetime

sys.path.append("./source")
from auxiliary.utils_data import *
from auxiliary.utils_model import *
from auxiliary.dataset import *
from models.models_nn import *

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--st_epoch', type=int, default=0, help='if continuing training, epoch from which to continue')
parser.add_argument('--model_type', type=str, default='MLP',  help='type of model')
parser.add_argument('--model_name', type=str, default='MLPNet2',  help='name of the model for log')
parser.add_argument('--model', type=str, default=None,  help='optional reload model path')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='sgd',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=1e-2,  help='learning rate')
parser.add_argument('--lr_decay_fact', type=float, default=2,  help='decay factor in learning rate')
parser.add_argument('--lr_decay_freq', type=int, default=10,  help='decay frequency (in epochs) in learning rate')
parser.add_argument('--momentum', type=float, default=0,  help='momentum (only SGD)')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
parser.add_argument('--random_state', type=int, default=0, help='random state for the split of data')
opt = parser.parse_args()

# ========================== TRAINING AND TEST DATA ========================== #
mortgage_data = MortgageData()
dataset_train = Mortgage(mortgage_data=mortgage_data, train=True, random_state=opt.random_state)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

dataset_test = Mortgage(mortgage_data=mortgage_data, train=False, random_state=opt.random_state)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

print('training set size %d' % len(dataset_train))
print('test set size %d' % len(dataset_test))

n_batch = np.int(np.ceil(len(dataset_train)/opt.batch_size))
n_batch_test = np.int(np.ceil(len(dataset_test)/opt.batch_size))
n_input = mortgage_data.n_input

# ========================== NETWORK AND OPTIMIZER ========================== #
network = eval("%s(n_input=%d, n_classes=opt.n_classes)" % (opt.model_name, n_input))
network.apply(init_weight)
network = network.float()

if opt.model is not None:
    network.load_state_dict(torch.load(opt.model))
    print("Weights from %s loaded" % opt.model)

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

# ====================== DEFINE STUFF FOR LOGS ====================== #
log_path = os.path.join('log', opt.model_type)
if not os.path.exists(log_path):
    os.mkdir(log_path)

save_path = os.path.join('trained_models', opt.model_type)
if not os.path.exists(save_path):
    os.mkdir(save_path)

log_file = os.path.join(log_path, '%s_%s_%s.txt' % (opt.model_type, opt.model_name, opt.optimizer))
if not os.path.exists(log_file):
    with open(log_file, 'a') as log:
        log.write(str(opt) + '\n\n')
        log.write(str(network) + '\n')
        log.write("train labels %s\n" % np.bincount(dataset_train.y))
        log.write("test labels %s\n\n" % np.bincount(dataset_test.y))

log_train_file = "./log/%s/logs_train_%s_%s.csv" % (opt.model_type, opt.model_name, opt.optimizer)
log_test_file = "./log/%s/logs_test_%s_%s.csv" % (opt.model_type, opt.model_name, opt.optimizer)

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
for epoch in range(opt.st_epoch, opt.n_epoch):
    # TRAINING
    st_time = time.time()
    network.train()
    value_meter_train.reset()
    loss_train = 0

    # LEARNING RATE SCHEDULE
    if (epoch+1) % opt.lr_decay_freq == 0:
        opt.lr /= opt.lr_decay_fact
        optimizer = get_optimizer(opt.optimizer, opt.lr, opt.momentum)

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
    
        print('[train epoch %d/%d ; batch: %d/%d] train loss: %.3g' % (epoch+1, opt.n_epoch, batch+1, n_batch, loss.item()))

    loss_train = float(loss_train.cpu())/n_batch
    dt = time.time()-st_time
    s_time = "%d min %d sec" % (dt//60, dt%60)
    
    print('='*40)
    print('[train epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train,
                                                                         value_meter_train.f1_macro, s_time))
    for i in range(opt.n_classes):
        print('cat %d: %s' % (i, value_meter_train.sum[i]))
    print('='*40)
    
    with open(log_file, 'a') as log:
        log.write('[train epoch %d/%d] | loss %.5g | f1_macro %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train,
                                                                                 value_meter_train.f1_macro, s_time) +  '\n')
        for i in range(opt.n_classes):
            log.write('cat %d: %s' % (i, value_meter_train.sum[i]) + '\n')

        row_train = {'model': opt.model_name, 
                     'random_state': opt.random_state,
                     'date': get_time(),
                     'n_epoch': opt.n_epoch,
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
print('[test epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, opt.n_epoch, loss_test,
                                                                    value_meter_test.f1_macro, s_time))
for i in range(opt.n_classes):
    print('cat %d: %s' % (i, value_meter_test.sum[i]))
print('='*40)

with open(log_file, 'a') as log:
    log.write('[test epoch %d/%d] | loss %.3g | f1_macro %.3g | time %s' % (epoch+1, opt.n_epoch, loss_test,
                                                                            value_meter_test.f1_macro, s_time) + '\n')
    for i in range(opt.n_classes):
        log.write('cat %d: %s' % (i, value_meter_test.sum[i]) + '\n')

row_test = {'model': opt.model_name, 
            'random_state': opt.random_state,
            'date': get_time(),
            'n_epoch': opt.n_epoch,
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
torch.save(network.state_dict(), os.path.join(save_path, '%s_%s.pth' % (opt.model_name, opt.optimizer)))

