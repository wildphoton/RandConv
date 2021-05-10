#!/usr/bin/env python
"""
Created by zhenlinxu on 11/23/2019
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ImageList(Dataset):
    """
    A generic image dataset where image and labels are loaded from a list file in this way: ::

        folder1/AAAimage.png 1
        folder2/BBBimage.png 2
        ...

    Args:
        root (string): Root directory path.
        list (string): List file path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        preload (bool): if preload all images into memory

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

    """


    def __init__(self, root, lists, transform=None, target_transform=None, loader=default_loader,
                 preload=False, pre_transform=False, n_samples=None):
        self.transform = transform
        self.root = root
        self.loader = loader
        self.pre_transform = pre_transform
        image_files, labels = self.parse_list(root, lists)
        if preload:
            print("preloading")
            self.images = [loader(image) for image in image_files]
            if pre_transform:
                print("pre-transforming")
                self.images = [pre_transform(image) for image in self.images]
        else:
            self.images = image_files
        self.labels = labels

        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload
        self.n_samples = n_samples # set this variable if you want the dataset to be recurrent

    def __getitem__(self, item):
        if self.n_samples:
            item = item % len(self.images)

        image = self.images[item] if self.preload else self.loader(self.images[item])
        label = self.labels[item]
        if not self.preload and self.pre_transform is not None:
            image = self.pre_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (image, label)

    def __len__(self):
        return self.n_samples if self.n_samples else len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def parse_list(root, listfiles):
        image_paths = []
        labels = []

        if isinstance(listfiles, str):
            listfiles = [listfiles]

        for file in listfiles:
            with open(file, 'r') as f:
                for l in f.readlines():
                    relative_path, label = l.strip().split(' ')
                    image_paths.append(os.path.join(root, relative_path))
                    labels.append(int(label))
        return image_paths, labels