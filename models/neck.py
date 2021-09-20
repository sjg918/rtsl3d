
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# refer
# https://github.com/Vegeta2020/SE-SSD/

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super(ConvLayer, self).__init__()

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

class FeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super(FeatureAggregator, self).__init__()
        self.bevshape = cfg.bevshape
        self.in_channels = cfg.neck_bev_in_channels
        self.path_channels = cfg.neck_path_channels
        # 144 72 36 18
        # 120 60 30 15
        # 64 128 256 512
        self.initconv1 = ConvLayer(self.in_channels//4,self.in_channels//4, 7,2, 'mish')
        self.initconv2 = ConvLayer(self.in_channels//4,self.in_channels//4, 5,2, 'mish')
        self.initconv3 = ConvLayer(self.in_channels//2,self.in_channels//2, 3,2, 'mish')
        self.initconv4 = ConvLayer(self.in_channels,self.path_channels[0], 3,1, 'mish')

        self.pathconv1 = ConvLayer(self.path_channels[0],self.path_channels[0], 3,1, 'mish')
        self.pathconv2 = ConvLayer(self.path_channels[0],self.path_channels[0], 3,1, 'mish')
        self.pathconv3 = ConvLayer(self.path_channels[0],self.path_channels[1], 3,2, 'mish')
        
        self.pathconv4 = ConvLayer(self.path_channels[1],self.path_channels[1], 3,1, 'mish')
        self.pathconv5 = ConvLayer(self.path_channels[1],self.path_channels[1], 3,1, 'mish')
        self.pathconv6 = ConvLayer(self.path_channels[1],self.path_channels[2], 3,2, 'mish')

        self.pathconv4 = ConvLayer(self.path_channels[2],self.path_channels[2], 3,1, 'mish')

        self.conv1 = ConvLayer(self.path_channels[2],self.path_channels[1], 1,1, 'leaky')
        self.conv2 = ConvLayer(self.path_channels[1],self.path_channels[2], 3,1, 'leaky')
        self.conv5 = ConvLayer(self.path_channels[2],self.path_channels[1], 1,1, 'leaky')

    def forward(self, input):
