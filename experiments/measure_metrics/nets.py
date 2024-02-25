import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from os.path import join as ospj
import copy




######################## CLASSIFIER ########################
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        # Employed activation function 
        self.actv = actv
        # Normalize true or false
        self.normalize = normalize
        # Downsample in the resnet block 
        self.downsample = downsample
        # If the shortcut should be learnt
        self.learned_sc = dim_in != dim_out
        # Build the networks used in the residual block
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        # If the shortcut is to be learned 
        if self.learned_sc:
            x = self.conv1x1(x)
        # If downsampling is to be performed 
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        # Blocks of normalization - activation - convolution - pool
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        # Return output normalized to unit variance
        return x / math.sqrt(2)  


class Classifier(nn.Module):
    def __init__(self, img_size=96, num_domains=2, max_conv_dim=512, dim_in=3):
        super().__init__()
        self.img_size = img_size
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim

        # Main convolutional block
        module = []
        module += [torch.nn.Conv2d(dim_in, 64, kernel_size=3, padding=1)]  # BxCx1x1
        module += [nn.LeakyReLU(0.2)]

        # First convolution 
        in_fm = 64 
        out_fm = in_fm*2

        for i in range(5):
            module += [ResBlk(in_fm, out_fm, downsample=True)]
            in_fm = out_fm 
            out_fm = out_fm*2 if out_fm*2 <= self.max_conv_dim else out_fm
        module += [nn.LeakyReLU(0.2)]
        module += [torch.nn.Conv2d(out_fm, out_fm, 3)]  # BxCx1x1
        module += [nn.LeakyReLU(0.2)]
        self.conv = torch.nn.Sequential(*module)

        # Fully-connected layer
        self.fc = torch.nn.Linear(out_fm,2)

    def forward(self, x):
        out = self.conv(x).view(x.shape[0], -1)
        out = self.fc(out)
        return out
