"""*****************************************************************************
Updated generator for 256x256 images

--------------------------------------------------------------------------------
Updated discriminator for 256x256 images

*****************************************************************************"""

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, activation='lrelu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.lrelu = nn.LeakyReLU(0.05, True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.activation == 'lrelu':
            return self.lrelu(x)
        elif self.activation == 'tan':
            return self.tanh(x)
        elif self.activation == 'no':
            return x

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(self.deconv(x))
        else:
            x = self.deconv(x)

        return self.lrelu(x)

class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        bn = nn.InstanceNorm2d(num_filter)
        lrelu = nn.LeakyReLU(0.2, True)
        pad = nn.ReflectionPad2d(1)

        self.resnet_block = nn.Sequential(
            pad,
            conv1,
            bn,
            lrelu,
            pad,
            conv2,
            bn,
            lrelu
        )

    def forward(self, x):
        return x + self.resnet_block(x)



class Generator256(nn.Module):
    """Generator for 256x256 pictures"""

    def __init__(self):
        super(Generator256, self).__init__()
        # encoding blocks
        self.pad = nn.ReflectionPad2d(3)

        self.conv1 = ConvBlock(3, 32, kernel_size=7, stride=1, padding=0)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)


        # residual blocks
        self.resnet_blocks = []
        for _ in range(6):
            self.resnet_blocks.append(ResnetBlock(128))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # decoding blocks
        self.deconv1 = DeconvBlock(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = DeconvBlock(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv3 = ConvBlock(32, 3, kernel_size=7, stride=1, padding=0, activation='tan', batch_norm=False)

    def forward(self, x):  
        x = self.conv1(self.pad(x))
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.resnet_blocks(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(self.pad(x))
        return x

    def normal_weight_init(self):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, 0.0, 0.02)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, 0.0, 0.02)
            if isinstance(m, ResnetBlock):
                nn.init.normal(m.conv.weight, 0.0, 0.02)
                nn.init.constant(m.conv.bias, 0)


class Discriminator256(nn.Module):
    """Discriminator for 256x256 pictures"""

    def __init__(self):
        super(Discriminator256, self).__init__()
        conv1 = ConvBlock(3, 64, kernel_size=4, stride=2, padding=1, batch_norm=False)
        conv2 = ConvBlock(64, 128, kernel_size=4, stride=2, padding=1)
        conv3 = ConvBlock(128, 256, kernel_size=4, stride=2, padding=1)
        conv4 = ConvBlock(256, 512, kernel_size=4, stride=1, padding=1)
        conv5 = ConvBlock(512, 1, kernel_size=4, stride=1, padding=1, activation='no', batch_norm=False)

        self.conv_blocks = nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5
        )

    def forward(self, x):
        return self.conv_blocks(x)

    def normal_weight_init(self):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, 0.0, 0.02)
