#!/usr/bin/env python
"""
Created by zhenlinx on 11/14/19
"""
import torch
# from scipy.ndimage.morphology import binary_dilation
from torchvision.transforms.functional import to_grayscale
from PIL import Image, ImageDraw
import random
import numpy as np
from lib.utils.laplacianofgaussianfiltering import LaplacianOfGaussianFiltering

class ToBinary(object):
    """Convert multiclass label to binary
    """

    def __init__(self, target_id):
        self.target_id = target_id

    def __call__(self, target):
        return (target == self.target_id).long()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GreyToColor(object):
    """Convert Grey Image label to binary
    """

    def __call__(self, image):
        if len(image.size()) == 3 and image.size(0) == 1:
            return image.repeat([3, 1, 1])
        elif len(image.size())== 2:
            return
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + '()'

class IdentityTransform():
    """do nothing"""

    def __call__(self, image):
        return image



class ToGrayScale():
    def __init__(self, outout_channel=3):
        self.output_channel = outout_channel

    def __call__(self, image):
        return to_grayscale(image, self.output_channel)


class AddGeometricPattern:
    def __init__(self, intensity, shape='rect', num=20, region='bg', maxsize=2):
        self.shape = shape
        self.num = num
        self.region = region
        self.intensity = intensity
        self.maxsize = maxsize

    def __call__(self, image):
        if self.shape is None:
            return image

        pattern = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(pattern)
        img_size = image.size
        size_min = 2
        # size_max = int(np.asarray(image).shape[0] * 0.07)
        size_max = self.maxsize
        # print(size_min, size_max, img_size)

        for i in range(self.num):
            bbox_size = random.randint(size_min, size_max)
            loc_x = random.randint(0, img_size[0] - bbox_size - 1)
            loc_y = random.randint(0, img_size[1] - bbox_size - 1)
            if self.shape == 'rect':
                draw.rectangle((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size),
                               outline=(self.intensity))
            elif self.shape == 'cross':
                # draw.line((loc_x, loc_y + bbox_size // 2, loc_x + bbox_size, loc_y + bbox_size // 2), fill=self.intensity, width=1)
                # draw.line((loc_x + bbox_size // 2, loc_y, loc_x + bbox_size // 2, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x + bbox_size, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'corner_tl':
                draw.line((loc_x, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y), fill=self.intensity, width=1)
            elif self.shape == 'corner_br':
                draw.line((loc_x+ bbox_size, loc_y, loc_x, loc_y), fill=self.intensity, width=1)
                draw.line((loc_x+ bbox_size, loc_y, loc_x + bbox_size, loc_y+ bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'line_up':
                draw.line((loc_x + bbox_size, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'line_down':
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size), fill=self.intensity, width=1)

            else:
                raise NotImplementedError("Pattern {} is not implemented".format(self.shape))
        # draw.rectangle((20, 20, 22, 22), fill=None, outline=(128))
        img_arr = np.asarray(image)
        pattern_arr = np.asarray(pattern)
        fg_mask = img_arr > 0
        pattern_mask = pattern_arr/255.0

        if self.region == 'bg':
            img = Image.fromarray(np.uint8(img_arr*fg_mask+(1-fg_mask)*pattern_arr))
        elif self.region == 'fg':
            img = Image.fromarray(np.uint8(img_arr * (1 - pattern_mask)))
            # img = Image.fromarray(np.uint8(img_arr - pattern_arr*fg_mask))
            # img = Image.fromarray(np.uint8(img_arr + pattern_arr * (1 - fg_mask)))

        return img
