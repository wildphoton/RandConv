#!/usr/bin/env python
"""
Created by zhenlinxu on 11/24/2019
"""

from .digit_net import DigitNet
from .alexnet import alexnet
from .vgg import get_vgg
from . import resnet
network_map = {
    'alexnet': alexnet,
    'digit': DigitNet,
}
for _name in resnet.__all__:
    network_map[_name] = resnet.__dict__[_name]


def get_network(name, **kwargs):
    if 'vgg' in name:
        return get_vgg(name, **kwargs)

    if name not in network_map:
        raise ValueError('Name of network unknown %s' % name)


    return network_map[name](**kwargs)


