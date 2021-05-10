#!/usr/bin/env python
"""
Created by zhenlinx on 11/13/2020
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
function for creating LOG kernel comes from https://gist.github.com/Seanny123/10452462
"""
range_inc = lambda start, end: range(start, end + 1)  # Because this is easier to write and read


def create_log(sigma, size=7):
    w = math.ceil(float(size) * float(sigma))

    # If the dimension is an even number, make it uneven
    if (w % 2 == 0):
        w = w + 1

    # Now make the mask
    l_o_g_mask = []

    w_range = int(math.floor(w / 2))
    # 	print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range_inc(-w_range, w_range):
        for j in range_inc(-w_range, w_range):
            l_o_g_mask.append(l_o_g(i, j, sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w, w)
    return l_o_g_mask


def l_o_g(x, y, sigma):
    # Formatted this way for readability
    nom = ((y ** 2) + (x ** 2) - 2 * (sigma ** 2))
    denom = ((2 * math.pi * (sigma ** 6)))
    expo = math.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    return nom * expo / denom


class LaplacianOfGaussianFiltering:
    def __init__(self, size=3, sigma=1.0, input_channel=3, normalization=True,
                 identity_prob=0.0, device='cpu'):
        super(LaplacianOfGaussianFiltering, self).__init__()

        _kernel = torch.from_numpy(create_log(sigma, size=size)).unsqueeze(0).unsqueeze(0).float()
        self.kernel = _kernel.repeat(input_channel, 1, 1, 1).to(device)
        self.device = device
        self.size = size
        self.normalization = normalization
        self.identity_prob = identity_prob
        self.input_channel = input_channel

    def __call__(self, input):
        if (self.identity_prob > 0 and torch.rand(1) > self.identity_prob):
            return input
        output = F.conv2d(input.unsqueeze(0).to(self.device), self.kernel, groups=self.input_channel, padding=self.size // 2)
        if self.normalization:
            output = (output - output.mean(dim=(1, 2, 3))) / output.std(dim=(1, 2, 3))
        return output.squeeze(0).detach().cpu()
