#!/usr/bin/env python
"""
Created by zhenlinxu on 11/23/2019
"""

from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
from torchvision.datasets.usps import USPS

# from .pacs import PACS
from .transforms import GreyToColor, IdentityTransform, ToGrayScale
from .mnist_m import MNISTM
from .synth_digit import SynthDigit
from .mnist_c import MNISTC
from .mnist_10k import MNIST10K

mnist = 'mnist'
mnist10k = 'mnist10k'
mnist_m = 'mnist_m'
mnist_c = 'mnist_c'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'
imagenet = 'imagenet'
pacs = 'pacs'
standard = 'standard'

dataset_std = {imagenet: (0.485, 0.456, 0.406),
               standard: (0.5, 0.5, 0.5),
               }

dataset_mean = {imagenet: (0.229, 0.224, 0.225),
                standard: (0.5, 0.5, 0.5),
                }

def get_dataset(name, **kwargs):

    if pacs in name:
        # available domains: cartoon, photo, sketch, art_painting
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        domain = name.split('-')[1]
        kwargs['domain'] = domain
        if 'pre_transform' not in kwargs:
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                ToGrayScale(3) if grey else IdentityTransform(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean[imagenet], dataset_std[imagenet])
            ])
            kwargs['pre_transform'] = transform
        if 'grey' in kwargs:
            del kwargs['grey']
        data = PACS(**kwargs)
    elif name == mnist:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNIST(**kwargs)

    elif name == mnist10k:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNIST10K(**kwargs)

    elif name == mnist_m:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']

        data = MNISTM(**kwargs)
    elif name == svhn:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if not kwargs['train']:
            kwargs['split'] = 'test'
        del kwargs['train']
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = SVHN(**kwargs)

    elif name == usps:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            # transforms.Normalize(dataset_mean[name], dataset_std[name])
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
            # transforms.Normalize(dataset_mean[mnist], dataset_std[mnist])

        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = USPS(**kwargs)

    elif name == synth:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'download' in kwargs:
            del kwargs['download']
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = SynthDigit(**kwargs)

    elif name == mnist_c:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNISTC(**kwargs)
    else:
        raise NotImplementedError('{} data does not exists'.format(name))
    return data
