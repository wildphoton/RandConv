#!/usr/bin/env python
"""
Created by zhenlinxu on 11/23/2019
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.alexnet
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, num_classes=1000, feature_size=6, model_path=None):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): number of classes
        model_path (string): path to pretrain model (using official model if is None)
    """
    model = AlexNet()
    if pretrained:
        if model_path is None:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        else:
            print("Loading model {}".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

    if feature_size == 6:
        model.classifier[6] = nn.Linear(4096, num_classes)
        # nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
        # nn.init.constant_(model.classifier[-1].bias, 0.)
    else:
        print("Adpating new classifier")
        fc_size = feature_size*feature_size*256
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size, num_classes),
        )
        for m in model.classifier.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    return model

