"""*****************************************************************************
All models are stored here

Generators are generating fake images.

Generator - Generate images.

--------------------------------------------------------------------------------
Discriminator are distinguish between real and created fake images.

Discriminator - Discriminate between normal images.

*****************************************************************************"""

import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False, batch_norm=True):
    """Custom convolution layer followed by batch norm."""
    grouped_layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        grouped_layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*grouped_layers)


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False, batch_norm=True):
    """Custom deconvolution layer followed by batch norm."""
    grouped_layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        grouped_layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*grouped_layers)


class Generator(nn.Module):
    """Generator for transfering from normal to fire"""

    def __init__(self, conv_dim=32):
        super(Generator, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 32
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64

        # residual blocks
        self.resnet_block = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)  # Kernel 1x1, Stride 1, Padding 1 conv_div 64
        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):  # IN : (Batch, 3, 32, 32)
        x = F.relu(self.conv1(x))  # IN: (Batch, 64, 16, 16)
        x = F.relu(self.conv2(x))  # IN: (Batch, 128, 8, 8)

        x = x + F.relu(self.resnet_block(x))  # IN: (Batch, 128, 8, 8)

        x = F.relu(self.deconv1(x))  # (Batch, 64, 16, 16)
        x = F.tanh(self.deconv2(x))  # (Batch, 3, 32, 32)
        return x


class Discriminator(nn.Module):
    """Discriminator"""

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 32
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64
        self.conv4 = conv(conv_dim * 4, 1, 4, 1, 0, False)  # Kernel 4x4, Stride 1, Padding 0 conv_div 1

    def forward(self, x):  # IN : (Batch, 3, 32, 32)
        x = F.relu(self.conv1(x))  # (Batch, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (Batch, 64, 8, 8)
        x = F.relu(self.conv3(x))  # (Batch, 128, 4, 4)
        x = sigmoid(self.conv4(x).squeeze())
        return x

