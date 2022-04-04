import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from .projections import *

# Interpolation class for resizing

class Interpolate(nn.Module):
    """Upsampling layer reverting the max pooling operation 
    """
    def __init__(self, upscale: str = 'scale', size: Optional[int] = None):
        super().__init__()
        self.upscale = upscale
        self.size = size

        if self.upscale == 'size':
            assert self.size is not None

    def forward(self, x):
        if self.upscale == 'scale':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.upscale == 'size':
            return F.interpolate(x, size=(self.size, self.size), mode='nearest')


# Basic convolutional layers

def conv3x3(in_fm: int, out_fm: int, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_fm, out_fm, kernel_size=3, padding=1, groups=groups, bias=True
    )

def conv1x1(in_fm: int, out_fm: int) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_fm, out_fm, kernel_size=1, bias=True)


# Resnet blocks and main body 

class BasicBlock(nn.Module):
    """BasicBlock is used with resnet18 and resnet34
    """
    expansion: int = 1

    def __init__(
        self,
        in_fm: int,
        out_fm: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:

        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Initial resnet block 
        self.conv1 = conv3x3(in_fm, out_fm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_fm, out_fm)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        # if upscale is not None it will also be added to upsample
        out = x
        # We can both upscale and upsample. They are performed in different loations of the network
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_fm: int,
        out_fm: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(out_fm * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_fm, width)
        self.conv2 = conv3x3(width, width, groups)
        self.conv3 = conv1x1(width, out_fm * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        # if upscale is not None it will also be added to upsample
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int, 
        latent_dim: int, 
        init_fm: int,
        out_width: int,
        out_height: int,
        variational: bool,
        h_dim: int,  # Latent space dimension
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        # input_height: int = 32,  # Initial spatial dimension 
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 512,
        first_conv3x3: bool = False,
        remove_first_maxpool: bool = False
    ) -> None:

        super(ResNetDecoder, self).__init__()

        self.in_channels = out_channels 
        self.latent_dim = latent_dim
        self.init_fm = init_fm  # Final number of feature maps 
        self.variational = variational  
        self.input_height = out_height  # Assuming that the height and width of the images is the same 
        self.h_dim = h_dim 
        self.groups = groups
        self.in_planes = h_dim  # Will be modified 

        self.first_conv3x3 = first_conv3x3
        self.remove_first_maxpool = remove_first_maxpool
        self.upscale_factor = 8  # To what extent we upscale the features 
        num_out_filters = width_per_group * widen

        if not first_conv3x3:
            self.upscale_factor *= 2

        if not remove_first_maxpool:
            self.upscale_factor *= 2

        # Projection goes from latent dimension to resenet h dimensions 
        self.linear_projection = nn.Linear(self.latent_dim, self.h_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise initial convolution 
        self.conv1 = conv1x1(self.h_dim // 16, self.h_dim)
        
        # Reversed Resnet structure with Bottlenck layers 
        num_out_filters /= 2
        self.layer1 = self._make_layer(
            block, int(num_out_filters), layers[0], Interpolate(
                upscale='size', size=self.input_height // self.upscale_factor
            )
        )  # No matter the size of the encoder or decoder, the first step resize image to a correct aspect ratio
        num_out_filters /= 2
        self.layer2 = self._make_layer(block, int(num_out_filters), layers[1], Interpolate())
        num_out_filters /= 2
        self.layer3 = self._make_layer(block, int(num_out_filters), layers[2], Interpolate())
        num_out_filters /= 2
        self.layer4 = self._make_layer(block, int(num_out_filters), layers[3], Interpolate())

        self.conv2 = conv3x3(int(num_out_filters) * block.expansion, self.init_fm)
        self.final_conv = conv3x3(self.init_fm, self.in_channels)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        upscale: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        upsample = None

        if self.in_planes != planes * block.expansion or upscale is not None:
            # this is passed into residual block for skip connection
            upsample = []
            if upscale is not None:
                upsample.append(upscale)
            upsample.append(conv1x1(self.in_planes, planes * block.expansion))
            upsample = nn.Sequential(*upsample)

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                upsample,
                self.groups,
                self.init_fm,
                upscale,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.init_fm,
                    upscale=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.linear_projection(x))

        x = x.view(x.size(0), self.h_dim // 16, 4, 4)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.remove_first_maxpool:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = self.relu(x)

        if not self.first_conv3x3:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.final_conv(x)

        return x


def decoder18(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim=512,
                **kwargs):
    # layers list is opposite the encoder (in this case [2, 2, 2, 2])
    return ResNetDecoder(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim, 
                BasicBlock, [2, 2, 2, 2], **kwargs)


def decoder34(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim = 512, 
                **kwargs):
    # layers list is opposite the encoder (in this case [3, 6, 4, 3])
    return ResNetDecoder(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim, BasicBlock, [3, 6, 4, 3], **kwargs)


def decoder50(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim=2048,
                **kwargs):
    # layers list is opposite the encoder
    return ResNetDecoder(out_channels,
                latent_dim,
                init_fm,
                out_width,
                out_height,
                variational, 
                h_dim,
                Bottleneck, [3, 6, 4, 3], **kwargs)



# if __name__ == "__main__":
#     z = torch.randn(64, 128)
#     model = decoder50(out_channels = 3,
#                     latent_dim = 128,
#                     init_fm = 64,
#                     out_width = 96,
#                     out_height = 96,
#                     variational = True)
#     print(model(z).shape)
