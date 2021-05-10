#!/usr/bin/env python
"""
Created by zhenlinxu on 05/04/2020
"""
import torch

def accuracy(output, target):
    """Computes the accuracy in percents """
    with torch.no_grad():
        # topk = (1,)
        # maxk = max(topk)
        if not len(output.shape) == len(target.shape):
            _, pred = output.max(1)
        else:
            pred = output
        acc = pred.eq(target).sum().detach().cpu().item()/pred.size(0)
        return acc
