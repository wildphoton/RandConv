#!/usr/bin/env python
"""
MNIST with 10K samples

Created by zhenlinx on 1/22/20
"""
import os
from torchvision.datasets import MNIST

class MNIST10K(MNIST):
    def __init__(self, **kwargs):
        super(MNIST10K, self).__init__(**kwargs)
        if self.train:
            self.data = self.data[:10000]
            self.targets = self.targets[:10000]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')
