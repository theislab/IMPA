""" Full assembly of the parts to form the complete network """

from .unet_modules import *
# from unet_modules import *
import torch

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, init_fm, n_conv, in_width, in_height, variational):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.init_fm = init_fm
        self.n_conv = n_conv
        self.in_width = in_width 
        self.in_height = in_height 
        self.variational = variational

        # The initial feature maps equate the number of channels of the image
        in_fm = self.init_fm

        # First convolution from the input channels to the first feature map 
        self.modules = [DoubleConv(in_channels, in_fm)]
        for i in range(self.n_conv):
            if i == 0:
                self.modules.append(Down(in_fm, in_fm))  # Feature maps double each time 
            else:
                mult = 2 if not (self.variational and i == self.n_conv-1) else 4 
                self.modules.append(Down(in_fm, in_fm*mult))  # Feature maps double each time 
                in_fm*=mult                

        self.module = torch.nn.Sequential(*self.modules)

    def forward(self, X):
        z = self.module(X)  # Encode the image 
        # Derive the encodings for the mean and the log variance
        if self.variational:
            mu, log_sigma = z.chunk(2, dim=1)
            return mu, log_sigma
        return z


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, init_fm, n_conv, out_width, out_height, variational, decoding_style='sum', concatenate_one_hot=True, extra_fm=0):
        super(UNetDecoder, self).__init__()
        self.out_channels = out_channels
        self.init_fm = init_fm
        self.n_conv = n_conv
        self.in_width = out_width 
        self.in_height = out_height 
        self.variational = variational
        self.decoding_style = decoding_style
        self.concatenate_one_hot = concatenate_one_hot
        self.extra_fm = extra_fm 

        # The initial feature maps equate the number of channels of the         
        self.in_channels = self.init_fm*2**(self.n_conv-1)

        # Build upampling convolutions 
        in_fm = self.in_channels
        self.modules = []
        
        # Upscale increasing number of feature maps based on concatenation 
        for _ in range(self.n_conv):
            self.modules.append(Up(in_fm+self.extra_fm, in_fm // 2))
            in_fm //= 2
        
        # Out conv is not transposing
        self.modules.append(OutConv(in_fm+self.extra_fm, self.out_channels))
        self.modules.append(torch.nn.Sigmoid())
        self.deconv = torch.nn.Sequential(*self.modules)        

    def forward_sum(self, z):
        X = self.deconv(z)
        return X 
    
    def forward_concat(self, z, y_drug, y_moa):
        # Concat label after every upscaling layer 
        for layer in self.deconv[:-1]:
            
            # Upsample drug labs
            y_drug_unsqueezed = y_drug.view(y_drug.size(0), y_drug.size(1), 1, 1)
            y_drug_broadcast = y_drug_unsqueezed.repeat(1, 1, z.size(2), z.size(3)).float()

            # Upsample moa labs
            y_moa_unsqueezed = y_moa.view(y_moa.size(0), y_moa.size(1), 1, 1)
            y_moa_broadcast = y_moa_unsqueezed.repeat(1, 1, z.size(2), z.size(3)).float()

            z = layer(torch.cat([z, y_drug_broadcast, y_moa_broadcast], dim=1))
        # Sigmoid pass
        X = self.deconv[-1](z)
        return X 

    def forward(self, z, y_drug, y_moa):
        if self.decoding_style == 'sum':
            return self.forward_sum(z)
        else:
            return self.forward_concat(z, y_drug, y_moa) 



if __name__ == '__main__':
    enc = UNetEncoder(3, 512, 64, 4, 96, 96, False)
    dec = UNetDecoder(3, 512, 64, 4, 96, 96, False)
    x = torch.rand(64, 3, 96, 96)
    z = enc(x)
    print(z.shape)
    x_out = dec(z)
    print(x_out.shape)
