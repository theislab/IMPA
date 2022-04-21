""" Full assembly of the parts to form the complete network """

from unet_modules import *
import torch

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, init_fm, n_conv, in_width, in_height, variational):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.init_fm = init_fm
        self.n_conv = n_conv
        self.in_width = in_width 
        self.in_height = in_height 
        self.variational = variational

        # The initial feature maps equate the number of channels of the 
        in_fm = self.init_fm

        # First convolution from the input channels to the first feature map 
        self.modules = [DoubleConv(in_channels, in_fm)]
        for _ in range(self.n_conv):
            self.modules.append(Down(in_fm, in_fm*2))  # Feature maps double each time 
            in_fm*=2

        self.module = torch.nn.Sequential(*self.modules)
        # Flattening layer 
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

        # Define a latent encoding 
        spatial_dim = self.in_width//(2**self.n_conv)  # The image gets downsampled for each convolution performed 
        self.flattened_dim = int(in_fm*(spatial_dim**2))  
        if self.variational:
            self.projection_layer = ProjectionHeadVAE(input_dim=self.flattened_dim,
                                                        output_dim=self.latent_dim)
        else:
            self.projection_layer = ProjectionHeadAE(input_dim=self.flattened_dim,
                                                        output_dim=self.latent_dim) 

    def forward(self, x):
        x = self.module(x)
        x = self.flatten(x)
        x = self.projection_layer(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, latent_dim, init_fm, n_conv, out_width, out_height, variational):
        super(UNetDecoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.init_fm = init_fm
        self.n_conv = n_conv
        self.in_width = out_width 
        self.in_height = out_height 
        self.variational = variational

        # The initial feature maps equate the number of channels of the         
        self.in_channels = self.init_fm*(2**self.n_conv)
        self.spatial_dim = self.in_width//(2**self.n_conv)  
        self.flattened_dim = int(self.in_channels*(self.spatial_dim**2))
        self.upsampling_layer = nn.Linear(self.latent_dim, self.flattened_dim, bias=True)

        # Build upampling convolutions 
        in_fm = self.in_channels
        self.modules = []
        for _ in range(self.n_conv):
            self.modules.append(Up(in_fm, in_fm // 2))
            in_fm //= 2
        self.modules.append(OutConv(in_fm, self.out_channels))
        self.module = torch.nn.Sequential(*self.modules)

        self.activation_last = torch.nn.Sigmoid()

    def forward(self, z):
        """Given a list of images xs and a latent vector, the network reconstructs the original image

        Args:
            xs (list): A list of latent images output of the encoder
        """
        # Go up from z
        x = self.upsampling_layer(z)
        x = x.view(-1, self.in_channels, self.spatial_dim, self.spatial_dim)
        x = self.module(x)
        return self.activation_last(x)


if __name__ == '__main__':
    enc = UNetEncoder(3, 512, 64, 4, 96, 96, False)
    dec = UNetDecoder(3, 512, 64, 4, 96, 96, False)
    x = torch.rand(64, 3, 96, 96)
    z = enc(x)
    x_out = dec(z)
    print(x_out.shape)


