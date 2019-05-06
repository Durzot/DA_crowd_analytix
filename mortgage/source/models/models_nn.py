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

