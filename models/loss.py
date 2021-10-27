
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# refer
# https://github.com/Tianxiaomo/pytorch-YOLOv4

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.layer = nn.ModuleList()
        self.layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if norm:
            self.layer.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.layer.append(nn.Mish())
        if activation == 'leaky':
            self.layer.append(nn.LeakyReLU(0.1))
        elif activation == 'linear':
            pass

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x

class MergeLayer(nn.Module):
    def __init__(self, out_channels, activation):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.BatchNorm2d(out_channels))

        if activation == "mish":
            self.conv.append(nn.Mish())
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1))
        elif activation == "linear":
            pass

    def forward(self, *args):
        x = torch.cat([*args], dim=1)

        for l in self.conv:
            x = l(x)
        return x

class BottleNeckLayer(nn.Module):
    def __init__(self, in_channels, neck_channels, activation):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, neck_channels, 1, 1, activation)
        self.conv2 = ConvLayer(neck_channels, in_channels, 3, 1, activation)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Up2DTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        return F.interpolate(x, size=[H*2, W*2], mode='nearest')

class BEVFeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super(BEVFeatureAggregator, self).__init__()
        self.conv1 = ConvLayer(512, 256, 1, 1, 'leaky')
        self.conv2 = ConvLayer(512, 256, 1, 1, 'linear')
        self.spp1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.spp2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.spp3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.spp4 = ConvLayer(256, 256, 1, 1, 'leaky')
        self.conv3 = ConvLayer(1024, 256, 1, 1, 'linear')
        self.merge1 = MergeLayer(512, 'leaky')
        self.conv4 = ConvLayer(512, 256, 1, 1, 'leaky')
        self.conv5 = ConvLayer(256, 128, 1, 1, 'linear')
        self.up = Up2DTensor()

        self.conv6 = ConvLayer(256, 128, 1, 1, 'linear')

        self.merge2 = MergeLayer(256, 'leaky')
        self.conv7 = ConvLayer(256, 128, 1, 1, 'leaky')
        self.conv8 = ConvLayer(128, 256, 3, 1, 'leaky')
        self.conv9 = ConvLayer(256, 128, 1, 1, 'leaky')
        self.conv10 = ConvLayer(128, 256, 3, 1, 'leaky')
        self.conv11 = ConvLayer(256, 128, 1, 1, 'leaky')
        self.conv12 = ConvLayer(128, 256, 3, 1, 'leaky')
        self.conv13 = ConvLayer(256, cfg.bfa_outchannels, 1, 1, 'linear', norm=False, bias=True)

    def forward(self, bev1, bev2, bev3, bev4):
        # 1/2 1/4 1/8 1/16
        x1 = self.conv1(bev4)
        x2 = self.conv2(bev4)
        m1 = self.spp1(x1)
        m2 = self.spp2(x1)
        m3 = self.spp3(x1)
        m4 = self.spp4(x1)
        x3 = self.conv3(torch.cat((m1,m2,m3,m4), dim=1))
        x4 = self.conv4(self.merge1(x2, x3))
        x5 = self.conv5(x4)

        x6 = self.conv6(bev3)
        x7 = self.conv7(self.merge2(self.up(x5), x6))
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)

        return x13
