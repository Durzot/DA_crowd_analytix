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

class MLPNet2(nn.Module):
    def __init__(self, n_input, n_classes):
        super(MLPNet2, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_input, n_input)
        self.fc2 = nn.Linear(n_input, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def get_features(self, x):
        x = F.relu(self.fc1(x))
        return x

