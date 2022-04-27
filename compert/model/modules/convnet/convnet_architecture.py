from collections import OrderedDict
from turtle import forward

import numpy as np
import torch
import torch.nn.functional as F

#-------------------------------------------------------------------------------------

"""
Convolutional layer with residual connection 
"""

# Convolutional layer with residual connection 
class ResidualLayer(torch.nn.Module):
    """
    Simple residual block 
    """
    def __init__(self, in_channels, out_channel):
        super(ResidualLayer, self).__init__()
        # Residual unit 
        self.resblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channel, kernel_size = 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size = 1)
            )
        self.activation_out = torch.nn.LeakyReLU()

    def forward(self, X):
        out = self.resblock(X)
        out += X[:,:out.shape[1],:,:]  # Residual connection 
        out = self.activation_out(out)
        return out

#-------------------------------------------------------------------------------------


class Encoder(torch.nn.Module):
    def __init__(self,
                in_channels: int = 5,
                init_fm: int = 64,
                n_conv: int = 3,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                variational: bool = True, 
                batch_norm_layers_ae: bool = False,
                dropout_ae: bool = False,
                dropout_rate_ae: float = 0 ) -> None:
                
        super(Encoder, self).__init__() 
    
        self.in_channels = in_channels
        self.init_fm = init_fm  # First number of feature maps
        self.n_conv = n_conv 
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.variational = variational

        # Batch norm and dropout 
        self.batch_norm_layers_ae = batch_norm_layers_ae
        self.dropout_ae = dropout_ae
        self.dropout_rate_ae = dropout_rate_ae

        # List containing the modules 
        self.modules = []

        # Build convolutional layers 
        in_fm = self.in_channels  # 3
        out_fm = self.init_fm  # Feature maps in the first convolutional layer   

        # Build downsizing convolutions 
        for i in range(0, self.n_conv):
            self.modules += [torch.nn.Conv2d(in_fm, out_fm,
                                kernel_size=4, 
                                stride=2, padding=1)]    

            # BN                    
            if i==0 or self.batch_norm_layers_ae:
                self.modules += [torch.nn.BatchNorm2d(out_fm)]

            # Activation 
            self.modules += [torch.nn.ReLU()]

            # Dropout 
            if self.dropout_ae:
                self.modules += [torch.nn.Dropout(p=self.dropout_rate_ae, inplace=True)]

            # Update feature maps
            in_fm = out_fm 
            out_fm = out_fm*2 if not (self.variational and i == self.n_conv-2) else out_fm*4

        # Add residual blocks 
        for i in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(in_fm, in_fm))

        self.encoder = torch.nn.Sequential(*self.modules) 

    def forward(self, X):
        z = self.encoder(X)  # Encode the image 
        
        # Derive the encodings for the mean and the log variance
        if self.variational:
            mu, log_sigma = z.chunk(2, dim=1)
            return mu, log_sigma

        return z


class Decoder(torch.nn.Module):
    def __init__(self,
                out_channels: int = 5,
                init_fm: int = 64,
                n_conv: int = 3,
                n_residual_blocks: int = 6, 
                out_width: int = 64,
                out_height: int = 64,
                variational: bool = True,
                decoding_style = 'sum',
                extra_fm=0) -> None:

        super(Decoder, self).__init__() 
        
        self.out_channels = out_channels
        self.n_conv = n_conv 
        self.init_fm = init_fm*(2**(self.n_conv-1))  # The first number of feature vectors 
        self.n_residual_blocks = n_residual_blocks
        self.out_width, self.out_height = out_width, out_height
        self.decoding_style = decoding_style
        self.extra_fm = extra_fm    
        
        # Build convolutional dimensions
        self.modules = []

        # Layer to upscale latent sample 
        self.variational = variational

        # The feature map values 
        in_fm = self.init_fm
        out_fm = self.init_fm//2

        # Append the residual blocks
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(in_fm+self.extra_fm, in_fm))

        for i in range(0, self.n_conv):
            # Convolutional layer
            self.modules += [torch.nn.Sequential(torch.nn.ConvTranspose2d(in_fm+self.extra_fm, out_fm, kernel_size=4, stride=2, padding=1),
                                torch.nn.ReLU() if i<self.n_conv-1 else torch.nn.Sigmoid())]
            


            # Update the number of feature maps
            in_fm = out_fm
            if i == self.n_conv-2:
                out_fm = self.out_channels
            else:
                out_fm = out_fm//2 

        # Assemble the decoder
        self.decoder = torch.nn.Sequential(*self.modules)

    def forward_sum(self, z):
        # Reshape to height x width
        X = self.decoder(z)
        return X 
    
    def forward_concat(self, z, y_drug, y_moa):
        # Reshape to height x width
        for layer in self.decoder:
            # Upsample drug labs
            y_drug_unsqueezed = y_drug.view(y_drug.size(0), y_drug.size(1), 1, 1)
            y_drug_broadcast = y_drug_unsqueezed.repeat(1, 1, z.size(2), z.size(3)).float()

            # Upsample moa labs
            y_moa_unsqueezed = y_moa.view(y_moa.size(0), y_moa.size(1), 1, 1)
            y_moa_broadcast = y_moa_unsqueezed.repeat(1, 1, z.size(2), z.size(3)).float()

            z = layer(torch.cat([z, y_drug_broadcast, y_moa_broadcast], dim=1))
        return z 

    def forward(self, z, y_drug, y_moa):
        if self.decoding_style == 'sum':
            return self.forward_sum(z)
        else:
            return self.forward_concat(z, y_drug, y_moa)   


# if __name__ == '__main__':
#     enc = Encoder(in_channels = 3,
#                 init_fm = 64,
#                 n_conv = 3,
#                 n_residual_blocks = 6, 
#                 in_width = 96,
#                 in_height = 96,
#                 variational = True, 
#                 batch_norm_layers_ae = False,
#                 dropout_ae = False,
#                 dropout_rate_ae = 0)

#     dec = Decoder(out_channels = 3,
#                 init_fm = 64,
#                 n_conv = 3,
#                 n_residual_blocks = 6, 
#                 out_width = 96,
#                 out_height = 96,
#                 variational = False,
#                 batch_norm_layers_ae = False,
#                 dropout_ae = False,
#                 dropout_rate_ae = 0) 
    
#     x = torch.Tensor(64, 3, 96, 96)
#     print(enc)
#     print(dec)
#     # x_hat = dec(res)
#     # print(x_hat.shape)