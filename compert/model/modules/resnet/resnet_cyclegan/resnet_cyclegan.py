from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from .resnet_cyclegan_modules import *
# from resnet_cyclegan_modules import *


class ResnetEncoderCycleGAN(nn.Module):
    def __init__(self,
            in_channels: int = 3,
            init_fm: int = 64,
            n_conv: int = 3,
            n_residual_blocks: int = 6, 
            in_width: int = 96,
            in_height: int = 96,
            variational: bool = True):
        
        self.in_channels = in_channels
        self.init_fm = init_fm  # First number of feature maps
        self.n_conv = n_conv 
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.variational = variational
        super(ResnetEncoderCycleGAN, self).__init__()
    
        self.modules = []

        # The kind of normalization provided is batch normalization 
        self.norm_layer = nn.BatchNorm2d()

        # Initial layer with a large convolutional kernel
        self.model = [nn.ReflectionPad2d(3),  # padding +3 to half the spatial dimension
                 nn.Conv2d(self.in_channels, self.init_fm, kernel_size=7, padding=0, bias=False),
                 self.norm_layer(self.init_fm),
                 nn.ReLU(True)]

        # Convolutional layer 
        for i in range(self.n_conv):  # add downsampling layers
            mult = 2 ** i
            self.model += [nn.Conv2d(self.init_fm * mult, self.init_fm * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      self.norm_layer(self.init_fm * mult * 2),
                      nn.ReLU(True)]

        # Residual network  
        mult = 2 ** self.n_conv
        for i in range(self.n_residual_blocks):  # add ResNet blocks
            self.model += [ResnetBlock(self.init_fm * mult, norm_layer=self.norm_layer, use_dropout=False, use_bias=False)]
        
        self.resnet = nn.Sequential(*self.model)
    
    def forward(self, x):
        z = self.resnet(x)
        return z


class ResnetDecoderCycleGAN(nn.Module):
    def __init__(self, 
                out_channels: int = 5,
                init_fm: int = 64,
                n_conv: int = 3,
                out_width: int = 64,
                out_height: int = 64,
                variational: bool = True,
                decoding_style = 'sum',
                extra_fm=0):

        super(ResnetDecoderCycleGAN, self).__init__()

        self.out_channels = out_channels 
        self.n_conv = n_conv 
        self.init_fm = init_fm
        self.out_width, self.out_height = out_width, out_height 
        self.variational = variational
        self.norm_layer = nn.BatchNorm2d
        self.decoding_style = decoding_style
        model = []

        # Extra fm controls amount of added dimensions in case of latent concatenation 
        self.extra_fm = extra_fm

        for i in range(self.n_conv):  # add upsampling layers
            mult = 2 ** (self.n_conv - i)
            model += [nn.ConvTranspose2d(self.init_fm * mult + self.extra_fm, int(self.init_fm * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      self.norm_layer(int(self.init_fm * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        # Last comvolution with a large kernel 
        model += [nn.Conv2d(self.init_fm+self.extra_fm, self.out_channels, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()]

        self.deconv = nn.Sequential(*model)

    def forward_sum(self, z):
        # Reshape to height x width
        X = self.deconv(z)
        return X 
    
    def forward_concat(self, z, y_drug, y_moa):
        # Reshape to height x width
        for layer in self.deconv[:-1]:
            # Upsample drug labs
            y_drug_unsqueezed = y_drug.view(y_drug.size(0), y_drug.size(1), 1, 1)
            y_drug_broadcast = y_drug_unsqueezed.repeat(1, 1, z.size(2), z.size(3))

            # Upsample moa labs
            y_moa_unsqueezed = y_moa.view(y_moa.size(0), y_moa.size(1), 1, 1)
            y_moa_broadcast = y_moa_unsqueezed.repeat(1, 1, z.size(2), z.size(3))

            z = layer(torch.cat([z, y_drug_broadcast, y_moa_broadcast], dim=1))
        X = self.deconv[-1](z)
        return X 

    def forward(self, z, y_drug, y_moa):
        if self.decoding_style == 'sum':
            return self.forward_sum(z)
        else:
            return self.forward_concat(z, y_drug, y_moa) 


if __name__ == '__main__':
    enc = ResnetEncoderCycleGAN(in_channels = 3,
                init_fm = 64,
                n_conv = 3,
                n_residual_blocks = 4, 
                in_width = 96,
                in_height = 96,
                variational = False)

    dec = ResnetDecoderCycleGAN(out_channels = 3,
                init_fm = 64,
                n_conv = 3,
                out_width = 96,
                out_height = 96,
                variational = False) 
    
    x = torch.Tensor(64, 3, 96, 96)
    res = enc(x)
    x = dec(res)
    print(x.shape)
