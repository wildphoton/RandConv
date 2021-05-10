#!/usr/bin/env python
"""
Created by zhenlinx on 12/10/19
"""
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image



class SynthDigit(data.Dataset):
    """A synthetic counterpart of the SVHN dataset used in the paper
    "Unsupervised Domain Adaptation by Backpropagation" by Yaroslav Ganin and Victor Lempitsky.
    Download from
        https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view
        linked from Yaroslav's homepage http://yaroslav.ganin.net/
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN_PyTorch/master/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, target_transform=None):
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "synth_train_32x32.mat" if train else "synth_test_32x32.mat"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError("Dataset not found at {}".format(
                os.path.join(self.root, self.filename)
            ))

        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        self.data = np.transpose(self.data, (3, 2, 0, 1))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))


if __name__ == '__main__':
    data = SynthDigit(root='../data')