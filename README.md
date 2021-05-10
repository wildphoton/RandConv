# RandConv

Official repo for [Robust and Generalizable Visual Representation Learning via Random Convolutions (ICLR2021)](https://openreview.net/forum?id=BVSM0x3EDK6)

Update 05/10: Code for RandConv and training scripts on digits data are available now! 
Scripts for PACS and imagenet are on the way.

## Requirements
See `requirements.txt`. Note that Pytorch v1.7 was used for testing.

## Running RandConv on Digits data
* MNIST-C has to be manually downloaded from https://github.com/google-research/mnist-c. Unzip the data into ./data/MNIST-M or change the data path in `train_digits.py`.
* `exp_mnist10k.sh` provided bash commands for reproduce digits experiments in the paper. You can select the specific settings by (un)commenting lines. `bash exp_mnist10k.sh 0` will run selected settings on GPU 0. 