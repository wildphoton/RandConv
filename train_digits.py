#!/usr/bin/env python
"""
Train rand-cnn on digits datasets
Created by zhenlinx on 11/22/19
"""
import os
import sys
sys.path.append(os.path.abspath(''))
import random
import argparse

from torch.utils.data import dataloader
from lib.datasets import get_dataset
from lib.datasets.transforms import GreyToColor, IdentityTransform, ToGrayScale, LaplacianOfGaussianFiltering

from randconv_trainer import *
from lib.networks import get_network

def main(args):
    print("Random Seed: ", args.rand_seed)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        if args.gpu_ids >= 0:
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)

    data_dir = "./data"
    mnist_c_dir = './data/mnist_c' # or change it to where you downloaded the mnist_c

    domains = ['mnist', 'mnist_m', 'svhn', 'usps', 'synth']
    args.n_classes = 10
    args.data_name = 'digits'
    args.image_size = 32
    image_size = (32, 32)

    if args.source == 'mnist10k':
        domains[0] = 'mnist10k'

    if args.multi_aug:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            GreyToColor(),
            ## for multiaug, the following two data transforms will be run on GPU to speedup preprocessing (see rand_cnn.py)
            # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # transforms.RandomGrayscale(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            if args.color_jitter else IdentityTransform(),
            ToGrayScale(3) if args.grey else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            LaplacianOfGaussianFiltering(size=3, sigma=1.0, identity_prob=0.5) if args.LoG else IdentityTransform(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        ToGrayScale(3) if args.grey else IdentityTransform(),
        transforms.ToTensor(),
        GreyToColor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("\n=========Preparing Data=========")

    assert args.source in domains, 'data_name can only be one of {}'.format(domains)
    trainset = get_dataset(args.source, root=data_dir, train=True, download=True, transform=train_transform)
    validsets = {domain: get_dataset(domain, root=data_dir, train=False, download=True, transform=test_transform) for domain in domains}

    trainloaders = [torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)]
    validloaders = {d: torch.utils.data.DataLoader(
        validsets[d], batch_size=512, shuffle=False, num_workers=2, pin_memory=True) for d in validsets}


    # Model
    print("\n=========Building Model=========")
    net = get_network(args.net, num_classes=args.n_classes, pretrained=args.pretrained)
    trainer = RandCNN(args)
    trainer.train(net, trainloaders, validloaders, testloaders=None, data_mean=(0.5, 0.5, 0.5), data_std=((0.5, 0.5, 0.5)))


    if args.test_corrupted:
        from lib.datasets.mnist_c import _CORRUPTIONS
        testdata = {type: get_dataset('mnist_c', root=mnist_c_dir, type=type, train=False, download=False, transform=train_transform) for type in _CORRUPTIONS[1:]}
        testloaders = {name: torch.utils.data.DataLoader(testdata[name], batch_size=256, shuffle=False, num_workers=2) for
                       name in testdata.keys()}
    else:
        testdata = {d: get_dataset(d, root=data_dir, train=False, download=True, transform=train_transform) for d in domains}
        testloaders = {d: torch.utils.data.DataLoader(testdata[d], batch_size=256, shuffle=False, num_workers=2) for d in domains}

    trainer.run_testing(net, testloaders)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    add_basic_args(parser)
    add_rand_layer_args(parser)
    parser.add_argument('--source', '-sc', type=str, default='mnist10k', help='souce domain for training')
    parser.add_argument('--target', '-tg', type=str, default='usps',
                        help='when target domain is given, use rest domains for training; '
                             'only effective when multi_source is true')
    args = parser.parse_args()

    main(args)
