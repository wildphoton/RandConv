'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url, model_urls

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def _resnet(arch, block, layers, pretrained, progress, model_path=None, **kwargs):
    useful_key =  ['num_classes', 'zero_init_residual', 'groups', 'width_per_group',
                       'replace_stride_with_dilation', 'norm_layer']
    irrelevant_key = [key for key in kwargs if key not in useful_key]
    for key in irrelevant_key:
        del kwargs[key]

    if pretrained:
        num_classes = kwargs['num_classes']
        kwargs['num_classes'] = 1000


        model = ResNet(block, layers, **kwargs)
        if model_path is None:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        else:
            print("Loading model from {}".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = ResNet(block, layers, **kwargs)

    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

