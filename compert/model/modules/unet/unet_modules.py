""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Convolutional layer that is followed by ReLU and batch normalization  
    """
    def __init__(self, init_fm, out_fm, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = init_fm
        self.double_conv = nn.Sequential(
            nn.Conv2d(init_fm, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_fm, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_fm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # Upsample the image 
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ProjectionHeadAE(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128):
        super(ProjectionHeadAE, self).__init__()
        self.projection_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection_head(x)



class ProjectionHeadVAE(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128):
        super(ProjectionHeadVAE, self).__init__()
        self.mu = nn.Linear(input_dim, output_dim, bias=False)
        self.logvar = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return [self.mu(x), self.logvar(x)]
