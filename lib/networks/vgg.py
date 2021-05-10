'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vggs
from torchvision.models.vgg import *

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
#
# class VGG(nn.Module):
#     def __init__(self, vgg_name, in_channels=3, num_classes=10):
#         super(VGG, self).__init__()
#         self.in_channels = in_channels
#         self.features = self._make_layers(cfg[vgg_name])
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Linear(512, num_classes)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(512 * 7 * 7, 4096),
#         #     nn.ReLU(True),
#         #     nn.Dropout(),
#         #     nn.Linear(4096, 4096),
#         #     nn.ReLU(True),
#         #     nn.Dropout(),
#         #     nn.Linear(4096, num_classes),
#         # )
#
#     def forward(self, x):
#         x = self.features(x)
#
#         x = F.avg_pool2d(x, kernel_size=x.size(2), stride=x.size(2))
#         # x = self.avgpool(x)
#         # x = torch.flatten(x, 1)
#         x = x.view(x.size(0), -1)
#
#         x = self.classifier(x)
#         return x
#
#     def _make_layers(self, cfg, batch_norm=True):
#         layers = []
#         in_channels = self.in_channels
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 if batch_norm:
#                     layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                                nn.BatchNorm2d(x),
#                                nn.ReLU(inplace=True)]
#                 else:
#                     layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                                nn.ReLU(inplace=True)]
#                 in_channels = x
#         return nn.Sequential(*layers)
#
#
#
# def vgg19(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration "E")
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     # if pretrained:
#     #     kwargs['init_weights'] = False
#     model = VGG('VGG19', **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
#     return model

#
# def vgg19_bn(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
#     return model

def get_vgg(arch, pretrained=False, progress=True, num_classes=1000, init_weights=True, **kwargs):
    net = globals()[arch](pretrained=pretrained, progress=progress, num_classes=1000, **kwargs)
    if num_classes != 1000:
        net.classifier[-1] = nn.Linear(4096, num_classes)
    if init_weights:
        nn.init.normal_(net.classifier[-1].weight, 0, 0.01)
        nn.init.constant_(net.classifier[-1].bias, 0)
    return net

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
