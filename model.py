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
from torch import sigmoid, tanh

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


class Generator32(nn.Module):
    """Generator"""

    def __init__(self, conv_dim=32):
        super(Generator32, self).__init__()
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
        x = tanh(self.deconv2(x))  # (Batch, 3, 32, 32)
        return x

class Generator64(nn.Module):
    """Generator"""

    def __init__(self, conv_dim=32):
        super(Generator64, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)

        # residual blocks
        self.resnet_block = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)  # Kernel 1x1, Stride 1, Padding 1 conv_div 64
        # decoding blocks
        self.deconv1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv2 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):  # IN : (Batch, 3, 32, 32)
        x = F.relu(self.conv1(x))  # IN: (Batch, 64, 16, 16)
        x = F.relu(self.conv2(x))  # IN: (Batch, 128, 8, 8)
        x = F.relu(self.conv3(x))  # IN: (Batch, 128, 8, 8)

        x = x + F.relu(self.resnet_block(x))  # IN: (Batch, 128, 8, 8)

        x = F.relu(self.deconv1(x))  # (Batch, 64, 16, 16)
        x = F.relu(self.deconv2(x))  # (Batch, 64, 16, 16)
        x = tanh(self.deconv3(x))  # (Batch, 3, 32, 32)
        return x

class Discriminator64(nn.Module):
    """Discriminator"""

    def __init__(self, conv_dim=32):
        super(Discriminator64, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 32
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)  # Kernel 4x4, Stride 2, Padding 1 conv_div 64
        self.conv5 = conv(conv_dim * 8, 1, 4, 1, 0, False)  # Kernel 4x4, Stride 1, Padding 0 conv_div 1

    def forward(self, x):  # IN : (Batch, 3, 32, 32)
        x = F.relu(self.conv1(x))  # (Batch, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (Batch, 64, 8, 8)
        x = F.relu(self.conv3(x))  # (Batch, 128, 4, 4)
        x = F.relu(self.conv4(x))  # (Batch, 128, 4, 4)
        x = sigmoid(self.conv5(x).squeeze())
        return x

class Generator128(nn.Module):
    """Generator"""

    def __init__(self, conv_dim=64):
        super(Generator128, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 4)
        self.conv4 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv5 = conv(conv_dim * 4, conv_dim * 4, 4)

        # residual blocks
        self.resnet_block1 = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)  # Kernel 1x1, Stride 1, Padding 1 conv_div 64
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim * 4, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim * 2, 4)
        self.deconv4 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv5 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))  # IN : 128
        x2 = F.relu(self.conv2(x1))  # IN: 64
        x3 = F.relu(self.conv3(x2))  # IN: 32
        x4 = F.relu(self.conv4(x3))  # IN: 16
        x5 = F.relu(self.conv5(x4))  # IN: 8

        r = F.relu(self.resnet_block1(x5))  # IN: (Batch, 256, 4, 4)
        
        y1 = F.relu(self.deconv1(r))  # 8
        y2 = F.relu(self.deconv2(y1))  # 16
        y3 = F.relu(self.deconv3(y2))  # 32
        y4 = F.relu(self.deconv4(y3))  # 64
        y5 = tanh(self.deconv5(y4))  # 128
        
        return y5

class Discriminator128(nn.Module):
    """Discriminator"""

    def __init__(self, conv_dim=64):
        super(Discriminator128, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)  # Kernel 4x4, Stride 4, Padding 0 conv_div 64
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # Kernel 4x4, Stride 4, Padding 0 conv_div 128
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 4)  # Kernel 4x4, Stride 4, Padding 1 conv_div 128
        self.conv4 = conv(conv_dim * 2, conv_dim * 4, 4)  # Kernel 4x4, Stride 4, Padding 1 conv_div 128
        self.conv5 = conv(conv_dim * 4, conv_dim * 4, 4)  # Kernel 4x4, Stride 4, Padding 1 conv_div 128
        self.conv6 = conv(conv_dim * 4, 1, 4, 1, 0, False)  # Kernel 4x4, Stride 1, Padding 0 conv_div 1

    def forward(self, x):  
        x = F.relu(self.conv1(x))  # 128
        x = F.relu(self.conv2(x))  # 64
        x = F.relu(self.conv3(x))  # 32
        x = F.relu(self.conv4(x))  # 16
        x = F.relu(self.conv5(x))  # 8
        x = sigmoid(self.conv6(x).squeeze())
        return x