"""
Resnet adapted from torchvision https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from json.encoder import HAS_UTF8
import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from .projections import *

# Basic convolutional layers

def conv3x3(in_fm: int, out_fm: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_fm, out_fm, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_fm: int, out_fm: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_fm, out_fm, kernel_size=1, stride=stride, bias=False)

# Resnet blocks and main body 

class BasicBlock(nn.Module):
    expansion: int = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        in_fm: int,
        out_fm: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
    
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2dò
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_fm, out_fm, stride)
        self.bn1 = norm_layer(out_fm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_fm, out_fm)
        self.bn2 = norm_layer(out_fm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        in_fm: int,
        out_fm: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Define the base number of feature maps of the current layer 
        width = int(out_fm * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_fm, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_fm * self.expansion)
        self.bn3 = norm_layer(out_fm * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        init_fm: int,
        in_width: int,
        in_height: int,
        variational:bool, 
        h_dim: 2048, 
        block: Type[Union[BasicBlock, Bottleneck]],  # Determined by the resnet type
        layers: List[int], # Determined by the resnet type
        groups: int = 1,
        widen: int = 1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        first_conv3x3: bool = True,
        remove_first_maxpool: bool = True,
    ) -> None:

        super(ResNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.init_fm = init_fm
        self.in_width = in_width 
        self.in_height = in_height 
        self.variational = variational 
        self.h_dim = h_dim
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Define the input feature maps as the custom number of groups times a widening factor if necessary 
        self.in_fm = self.init_fm * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = self.init_fm # By default it starts with 64

        # This parameter will increase at each block end 
        num_out_filters = self.init_fm * widen

        # Decide whether the first convolution has a kernel of 7 (with stride 2) or a kernel of 3
        if first_conv3x3:
            self.conv1 = nn.Conv2d(self.in_channels, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, num_out_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)

        # Choose whether a first maximum pooling should be performed
        if remove_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block to create the final resnet architecture
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.variational:
            self.projection_layer = ProjectionHeadVAE(input_dim=self.h_dim,
                                                        output_dim=self.latent_dim)
        else:
            self.projection_layer = ProjectionHeadVAE(input_dim=self.h_dim,
                                                        output_dim=self.latent_dim)


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], out_fm : int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """Compose a resnet layer 

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): The type of block between basic and bottle neck
            out_fm (int): _description_
            blocks (int): _description_
            stride (int, optional): _description_. Defaults to 1.
            dilate (bool, optional): _description_. Defaults to False.

        Returns:
            nn.Sequential: _description_
        """
        norm_layer = self._norm_layer  # Batch normalization or any normalization layer
        downsample = None  # Downsampling layer
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Downsampling only in the first block of a layer 
        if stride != 1 or self.in_fm != out_fm * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_fm, out_fm * block.expansion, stride),
                norm_layer(out_fm * block.expansion),
            )

        # Create the layers 
        layers = []
        layers.append(
            block(
                self.in_fm,
                out_fm,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.in_fm = out_fm * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_fm,
                    out_fm,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.projection_layer(x)
        return x


def resnet18(in_channels, 
            latent_dim,
            init_fm,
            in_width,
            in_height,
            variational,
            h_dim=512, 
            **kwargs):
    return ResNetEncoder(in_channels, latent_dim,
            init_fm,
            in_width,
            in_height,
            variational,
            h_dim, 
            BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(in_channels, 
            latent_dim,
            init_fm,
            in_width,
            in_height,
            variational,
            h_dim = 512, **kwargs):
    return ResNetEncoder(in_channels, 
            latent_dim,
            init_fm,
            in_width,
            in_height,
            variational,
            h_dim, 
            BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(in_channels, 
            latent_dim,
            init_fm,
            in_width,
            in_height,
            variational,
            h_dim=2048, 
            **kwargs):
    return ResNetEncoder(in_channels,
                latent_dim,
                init_fm,
                in_width,
                in_height,
                variational,
                h_dim,
                Bottleneck, [3, 4, 6, 3], **kwargs)


if __name__ == "__main__":
    imgs = torch.rand(16, 3, 96, 96)
    encoder = resnet18(in_channels=3, 
            latent_dim = 128,
            init_fm = 128,
            in_width = 96,
            in_height = 96,
            variational= True)
    
    # print(encoder)
    print(encoder(imgs)[0].shape)