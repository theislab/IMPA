import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


"""
The latent discriminator network
"""

class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='None'):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module):
  def __init__(self, init_fm, out_fm, depth, num_outputs):
    super(DiscriminatorNet, self).__init__()
    self.init_fm = init_fm
    self.out_fm = out_fm 
    self.depth = depth 
    self.num_outpus = num_outputs

    # First number of feature maps 
    in_fm = self.init_fm 

    model = []
    for i in range(depth):
        model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1)]
        if i == 0:
            in_fm = out_fm

    # Final convolution to produce number of outputs on the feature map dimension
    model += [nn.Conv2d(in_fm, num_outputs, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1)) # BxCx1x1 --> BxC
    return out


"""
The covariate encoding network 
"""

class LabelEncoder(nn.Module):
    def __init__(self, output_dim, input_fm, output_fm):
        super(LabelEncoder, self).__init__()

        self.output_dim = output_dim
        self.input_fm = input_fm
        self.output_fm = output_fm
        
        # Depth 
        depth = int(np.log2(self.output_dim//3))  # Requited for upsampling
        
        # Initial feature map setup
        in_fm = self.input_fm
        out_fm = self.output_fm

        # Initialize the modules 
        self.modules = [torch.nn.ConvTranspose2d(in_fm, out_fm, kernel_size = 3, stride = 2, padding=0),
                        torch.nn.ReLU()] # 1x1 --> 3x3
        for i in range(depth):
            self.modules.append(torch.nn.ConvTranspose2d(out_fm, out_fm, kernel_size = 4, stride = 2, padding=1))
            self.modules.append(torch.nn.ReLU())
        self.transp = torch.nn.Sequential(*self.modules)

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        x = self.transp(z)
        return x

class LabelEncoderLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LabelEncoderLinear, self).__init__()
        self.input_fm = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim))

    def forward(self, z):
        out = self.mlp(z)
        return out



"""Disentanglement classifier for classification 
"""

class DisentanglementClassifier(nn.Module):
    def __init__(self, init_dim, init_fm, out_fm, num_outputs):
        super(DisentanglementClassifier, self).__init__()
        self.init_dim = init_dim  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.out_fm = out_fm  # Output feature maps 
        self.num_outpus = num_outputs  # Number of classes for the classification 

        # First number of feature maps 
        in_fm = self.init_fm 

        model = []
        for i in range(2):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=1, padding=1, norm='Instance')]
            if i == 0:
                in_fm = out_fm
        
        flattened_dim = self.init_dim *  out_fm
        # Compile model 
        self.conv = nn.Sequential(*model)
        
        # Linear classification layer 

        self.linear = torch.nn.Linear(flattened_dim, self.num_outpus)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        return out


"""GAN discriminator for prestine vs true
"""

class GANDiscriminator(nn.Module):
    def __init__(self, init_dim, init_ch, init_fm):
        super(GANDiscriminator, self).__init__()
        self.init_dim = init_dim  # Spatial dimension
        self.init_ch = init_ch  # Input feature maps (3) 
        self.init_fm = init_fm  # The number of feature maps in the first layer 

        # Modifiable numbers of feature maps
        in_fm = self.init_ch
        out_fm = self.init_fm

        model = []
        for i in range(4):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='Instance')]
            in_fm = out_fm
            out_fm = in_fm *2
        
        flattened_dim = (self.init_dim//16)**2  * in_fm

        # Compile model 
        self.conv = nn.Sequential(*model)
        
        # Linear classification layer 
        self.linear = torch.nn.Linear(flattened_dim, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        return self.linear(out)


if __name__ == '__main__':
    # x = torch.rand(3, 512, 6, 6)
    # dis = DiscriminatorNet(512, 256, 3, 4)
    # print(dis(x).shape)

    x = torch.rand(64, 3, 96, 96)
    enc = GANDiscriminator(init_dim=96, init_ch=3, init_fm=64)
    print(enc(x).shape)
    print(enc)