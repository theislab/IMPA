import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


# Basic blocks encoder
class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='Batch'):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ReLUINSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='Batch'):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, in_channels, out_channels, stride=1):
        return [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(in_channels, out_channels, stride)
        model += [nn.BatchNorm2d(out_channels)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(out_channels, out_channels)
        model += [nn.BatchNorm2d(out_channels)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        # Residual connection 
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).to(self.device))
        return x + noise


# Upsampling layers 

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                    padding=padding, output_padding=output_padding, bias=True)]
        # In the deconvolutions you apply layer normalization 
        model += [LayerNorm(out_channels)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
    
  def forward(self, x):
        return self.model(x)
