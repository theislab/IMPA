from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from .resnet_drit_modules import *
# from resnet_drit_modules import *

class ResnetDritEncoder(torch.nn.Module):
    
    def __init__(self,
            in_channels: int = 3,
            init_fm: int = 64,
            n_conv: int = 3,
            n_residual_blocks: int = 6, 
            in_width: int = 96,
            in_height: int = 96,
            variational: bool = True):
        
        super(ResnetDritEncoder, self).__init__()

        self.in_channels = in_channels
        self.init_fm = init_fm  # First number of feature maps
        self.n_conv = n_conv 
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.variational = variational

        # Encoding layers 
        self.modules = []
        in_fm = self.init_fm

        # Add the first module convolution with leaky relu and a kernel size of 7 (capture high-level spatial features)
        self.modules += [LeakyReLUConv2d(self.in_channels, in_fm, kernel_size=7, stride=2, padding=3)]

        # Add an additional number of convolutions with reduced kernel size 
        for i in range(1, self.n_conv):
            mult = 2 if not (self.variational and i == self.n_conv-1) else 4  # The product is by 4 if variational (double fm for var and mean)
            self.modules += [ReLUINSConv2d(in_fm, in_fm * mult, kernel_size=3, stride=2, padding=1)]  # Convolution 
            in_fm = in_fm * mult 
        
        # Build residual network
        for i in range(0, self.n_residual_blocks):
            self.modules += [INSResBlock(in_fm, in_fm)]
            if i == self.n_residual_blocks-1:
                self.modules += [GaussianNoiseLayer()]  # Add Gauss after the last residual net 
        
        self.conv = nn.Sequential(*self.modules)

    def forward(self, X):
        # Encode the image 
        z = self.conv(X)  
        # Derive the encodings for the mean and the log variance
        if self.variational:
            mu, log_sigma = z.chunk(2, dim=1)
            return mu, log_sigma
        return z


class ResnetDritDecoder(nn.Module):
    def __init__(self, 
                out_channels: int = 5,
                init_fm: int = 64,
                n_conv: int = 3,
                n_residual_blocks: int = 4, 
                out_width: int = 64,
                out_height: int = 64,
                decoding_style = 'sum', 
                concatenate_one_hot = True,
                extra_fm = 6,
                normalize=False):

        super(ResnetDritDecoder, self).__init__()

        self.out_channels = out_channels
        self.n_conv = n_conv 
        self.init_fm = init_fm*(2**(self.n_conv-1))  # The first number of feature vectors 
        self.n_residual_blocks = n_residual_blocks  
        self.out_width, self.out_height = out_width, out_height 
        self.decoding_style = decoding_style
        self.concatenate_one_hot = concatenate_one_hot
        self.extra_fm = extra_fm  # How many extra feature maps die to concatenation 
        self.normalize = normalize
        
        # Initial number of feature maps
        in_fm = self.init_fm  
        self.modules = []

        # Residual blocks
        residual_connections = []
        for i in range(0, self.n_residual_blocks):
            # Will perform a single concatenation before the residual block, so all the blocks are a single all together 
            residual_connections += [INSResBlock(in_fm+self.extra_fm, in_fm+self.extra_fm)]
        # Residual connections are treated as a whole layer within the module 
        residual_connections = torch.nn.Sequential(*residual_connections)
        self.modules += [residual_connections]

        # Output feature channels of the residual block 
        in_fm = in_fm+self.extra_fm
        for i in range(0, self.n_conv):
            self.modules += [ReLUINSConvTranspose2d(in_fm+self.extra_fm, in_fm//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            in_fm = in_fm//2
        
        # Output channel 
        self.modules += [nn.ConvTranspose2d(in_fm+self.extra_fm, self.out_channels, kernel_size=1, stride=1, padding=0)] + \
                            [nn.Sigmoid() if not self.normalize else nn.Tanh()]
        self.deconv = nn.Sequential(*self.modules)
    
    def forward(self, z, y_drug):
        if self.decoding_style == 'sum':
            return self.forward_sum(z, y_drug)
        else:
            return self.forward_concat(z, y_drug) 

    def forward_sum(self, z, y_drug):
        z = z + y_drug
        X = self.deconv(z)
        return X, z
    
    def forward_concat(self, z, y_drug):
        z_init = None
        for i, layer in enumerate(self.deconv[:-1]):

            # Upsample drug labs (originally a linear vector)
            y_drug_unsqueezed = y_drug.view(y_drug.size(0), y_drug.size(1), 1, 1)
            y_drug_broadcast = y_drug_unsqueezed.repeat(1, 1, z.size(2), z.size(3)).float()

            z_concat = torch.cat([z, y_drug_broadcast], dim=1)
            z = layer(z_concat)
            
            # Output first latent concatenation 
            if i == 0:
                z_init = z_concat
            
        X = self.deconv[-1](z)
        return X, z_init


# if __name__ == '__main__':
#     enc = ResnetDritEncoder(in_channels = 3,
#                 init_fm = 64,
#                 n_conv = 3,
#                 n_residual_blocks = 4, 
#                 in_width = 96,
#                 in_height = 96,
#                 variational = False)

#     # dec = ResnetDritDecoder(out_channels = 3,
#     #             init_fm = 64,
#     #             n_conv = 5,
#     #             n_residual_blocks = 4, 
#     #             out_width = 96,
#     #             out_height = 96,
#     #             variational = False,
#     #             batch_norm_layers_ae = False,
#     #             dropout_ae = False,
#     #             dropout_rate_ae = 0) 
    
#     x = torch.Tensor(64, 3, 96, 96)
#     res = enc(x)
#     # x = dec(res)
#     print(res.shape)