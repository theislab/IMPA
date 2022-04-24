import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


# The discriminator network 

class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='None'):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if 'norm' == 'Instance':
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
    for i in range(depth-1):
        model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1, norm='Instance')]
        if i == 0:
            in_fm = out_fm

    # Reach dimensionality 1x1 wi
    model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=0, norm='Instance')]
    # Final convolution to produce number of outputs on the feature map dimension
    model += [nn.Conv2d(in_fm, num_outputs, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1)) # BxCx1x1 --> BxC
    return out


# The covariate encoding net 

class LabelEncoder(nn.Module):
    def __init__(self, output_dim, input_fm, output_fm):
        super(LabelEncoder, self).__init__()

        self.output_dim = output_dim
        self.input_fm = input_fm
        self.output_fm = output_fm
        
        # Depth 
        depth = int(np.log2(self.output_dim//3)) 
        
        # Initial feature map setup
        in_fm = self.input_fm
        out_fm = self.output_fm

        # Initialize the modules 
        self.modules = [torch.nn.ConvTranspose2d(in_fm, out_fm, kernel_size = 3, stride = 2, padding=0),
                        torch.nn.ReLU()] # 1x1 --> 3x3
        for i in range(depth):
            self.modules.append(torch.nn.ConvTranspose2d(self.output_fm, self.output_fm, kernel_size = 4, stride = 2, padding=1))
            self.modules.append(torch.nn.ReLU())
        self.transp = torch.nn.Sequential(*self.modules)

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        x = self.transp(z)
        return x


if __name__ == '__main__':
    # x = torch.rand(3, 512, 12, 12)
    # dis = DiscriminatorNet(512, 256, 4, 4)
    # print(dis(x).shape)
    x = torch.rand(3, 512)
    enc = LabelEncoder(output_dim=6, input_fm=512, output_fm=256)
    print(enc(x).shape)
    
