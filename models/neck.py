
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

class FeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super(FeatureAggregator, self).__init__()
        self.bevshape = cfg.bevshape
        self.in_channels = cfg.neck_bev_in_channels
        self.route = cfg.neck_channels
        self.out_channels = cfg.neck_bev_out_channels
        
        self.initconv1 = ConvLayer(self.in_channels//4,self.in_channels//4, 7,2, 'linear', norm=False)
        self.initconv2 = ConvLayer(self.in_channels//4,self.in_channels//4, 5,2, 'linear', norm=False)
        self.initconv3 = ConvLayer(self.in_channels//2,self.in_channels//2, 3,2, 'linear', norm=False)
        self.initmerge = MergeLayer(self.in_channels, 'mish')
        self.initconv4 = ConvLayer(self.in_channels,self.route[0], 3,1, 'mish')

        self.conv1 = BottleNeckLayer(self.route[0],self.route[0]//2, 'mish')
        self.conv2 = ConvLayer(self.route[0],self.route[1], 3,2, 'mish')
        
        self.conv3 = BottleNeckLayer(self.route[1],self.route[1]//2, 'mish')
        self.conv4 = ConvLayer(self.route[1],self.route[2], 3,2, 'mish')

        self.conv5 = BottleNeckLayer(self.route[2],self.route[2]//2, 'mish')

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up2d = Up2DTensor()

        self.pathconv1 = ConvLayer(self.route[2],self.route[1], 1,1, 'linear', norm=False)
        self.pathconv2 = ConvLayer(self.route[1],self.route[1], 1,1, 'linear', norm=False)
        self.merge1 = MergeLayer(self.route[1] * 2, 'leaky')

        self.conv6 = BottleNeckLayer(self.route[1] * 2,self.route[1], 'leaky')

        self.pathconv3 = ConvLayer(self.route[1] * 2,self.route[1], 1,1, 'linear', norm=False)
        self.pathconv4 = ConvLayer(self.route[2],self.route[1], 1,1, 'linear', norm=False)
        self.pathconv5 = ConvLayer(self.route[1],self.route[1], 1,1, 'linear', norm=False)
        self.pathconv6 = ConvLayer(self.route[0],self.route[1], 1,1, 'linear', norm=False)
        self.merge2 = MergeLayer(self.route[1] * 4, 'leaky')

        self.conv7 = BottleNeckLayer(self.route[1] * 4,self.route[1], 'leaky')

        self.pathconv7 = ConvLayer(self.route[1] * 4,self.route[1], 1,1, 'linear', norm=False)
        self.pathconv8 = ConvLayer(self.route[1],self.route[1], 1,1, 'linear', norm=False)
        self.pathconv9 = ConvLayer(self.route[0],self.route[1], 1,1, 'linear', norm=False)
        self.merge3 = MergeLayer(self.route[1] * 3, 'leaky')

        self.conv8 = ConvLayer(self.route[1] * 3,self.route[3], 3,1, 'leaky')
        self.conv9 = BottleNeckLayer(self.route[3],self.route[3]//2, 'leaky')
        self.conv10 = ConvLayer(self.route[3],self.out_channels, 1,1, 'linear', norm=False, bias=True)

    def forward(self, input):
        i1 = self.initconv1(input[:, :self.in_channels//4, :, :])
        i2 = self.initconv2(input[:, self.in_channels//4:self.in_channels//2, :, :])
        i3 = self.initconv3(input[:, self.in_channels//2:, :, :])
        i4 = self.initmerge(i1,i2,i3)
        x0 = self.initconv4(i4)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        p1 = self.pathconv1(x5)
        p2 = self.maxpool2d(self.pathconv2(x3))
        x6 = self.merge1(p1, p2)
        x6 = self.conv6(x6)

        p3 = self.up2d(self.pathconv3(x6))
        p4 = self.up2d(self.pathconv4(x5))
        p5 = self.pathconv5(x3)
        p6 = self.maxpool2d(self.pathconv6(x1))
        x7 = self.merge2(p3, p4, p5, p6)
        x7 = self.conv7(x7)

        p7 = self.up2d(self.pathconv7(x7))
        p8 = self.up2d(self.pathconv8(x3))
        p9 = self.pathconv9(x1)
        x8 = self.merge3(p7, p8, p9)
        x8 = self.conv8(x8)
        x9 = self.conv9(x8)

        return self.conv10(x9)
