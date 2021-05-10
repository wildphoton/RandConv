#!/usr/bin/env python
"""
Created by zhenlinxu on 01/07/2020
"""
import torch
import torch.nn as nn

class EuclideanDistance(nn.Module):
    def forward(self, x, y):
        return torch.dist(x, y, 2)/x.size(0)
