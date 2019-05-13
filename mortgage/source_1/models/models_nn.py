# -*- coding: utf-8 -*-
"""
Models for classifying images
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#########################
# Simple architectures
#########################

class MLPNet3(nn.Module):
    def __init__(self, n_input, n_classes):
        super(MLPNet3, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def get_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MLPNet3Drop(nn.Module):
    def __init__(self, n_input, n_classes, p=0.2):
        super(MLPNet3Drop, self).__init__()
        self.p = p
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.drop1 = nn.Dropout(self.p)
        self.fc2 = nn.Linear(16, 16)
        self.drop2 = nn.Dropout(self.p)
        self.fc3 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    def get_features(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop1(F.relu(self.fc2(x)))
        return x

class MLPNet4(nn.Module):
    def __init__(self, n_input, n_classes):
        super(MLPNet4, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def get_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class MLPNet4Drop(nn.Module):
    def __init__(self, n_input, n_classes, p=0.2):
        super(MLPNet4Drop, self).__init__()
        self.p = p
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.drop1 = nn.Dropout(self.p)
        self.fc2 = nn.Linear(16, 16)
        self.drop2 = nn.Dropout(self.p)
        self.fc3 = nn.Linear(16, 16)
        self.drop3 = nn.Dropout(self.p)
        self.fc4 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    def get_features(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        return x

class MLPNet5(nn.Module):
    def __init__(self, n_input, n_classes):
        super(MLPNet5, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    def get_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class MLPNet5Drop(nn.Module):
    def __init__(self, n_input, n_classes, p=0.2):
        super(MLPNet5Drop, self).__init__()
        self.p = p
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, 16)
        self.drop1 = nn.Dropout(self.p)
        self.fc2 = nn.Linear(16, 16)
        self.drop2 = nn.Dropout(self.p)
        self.fc3 = nn.Linear(16, 16)
        self.drop3 = nn.Dropout(self.p)
        self.fc4 = nn.Linear(16, 16)
        self.drop4 = nn.Dropout(self.p)
        self.fc5 = nn.Linear(16, n_classes)
    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.drop4(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x
    def get_features(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.drop4(F.relu(self.fc4(x)))
        return x

