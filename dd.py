https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850

import numpy as np
from torch.nn import functional as F
import torch
import cv2

def read_calib_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R_rect': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

calibs = read_calib_file('C:\Datasets/KITTI_OD/calib/000008.txt')
img = cv2.imread('C:\Datasets/KITTI_OD/image_2/000008.png')
sgm = cv2.imread('C:\Datasets/KITTI_OD/image_sgm/000008.png')
# Projection matrix from rect camera coord to image2 coord
P2 = calibs['P2']
P2 = np.reshape(P2, [3, 4])
P2 = torch.from_numpy(P2).unsqueeze(0)

h, w, _ = img.shape
x_range = np.arange(h, dtype=np.float32)
y_range = np.arange(w, dtype=np.float32)
_, yy_grid  = np.meshgrid(y_range, x_range)

yy_grid = torch.from_numpy(yy_grid).unsqueeze(0)
fy =  P2[:, 1:2, 1:2] #[B, 1, 1]
cy =  P2[:, 1:2, 2:3] #[B, 1, 1]
Ty =  P2[:, 1:2, 3:4] #[B, 1, 1]



import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn

import cv2

import os

import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class myConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.layer = nn.ModuleList()
        self.layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if norm:
            self.layer.append(nn.BatchNorm2d(out_channels))
        elif activation == 'mish':
            self.layer.append(nn.Mish())
        elif activation == 'leaky':
            self.layer.append(nn.LeakyReLU(0.1))
        elif activation == 'relu':
            self.layer.append(nn.ReLU())
        elif activation == 'linear':
            pass
        else:
            raise Exception("cehck activation fucn.")

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x


class DispRefinementHead(nn.Module):
    def __init__(self, in_channels):
        super(DispRefinementHead, self).__init__()
        self.conv1 = myConv2D(in_channels + 1, in_channels//2, 3, 1, 'relu')
        self.conv2 = myConv2D(in_channels//2, in_channels, 3, 1, 'relu')
        self.conv3 = myConv2D(in_channels, in_channels//2, 3, 1, 'relu')
        self.conv4 = myConv2D(in_channels//2, in_channels, 3, 1, 'relu')
        self.conv5 = myConv2D(in_channels, in_channels//2, 3, 1, 'relu')
        self.conv6 = myConv2D(in_channels//2, in_channels, 3, 1, 'relu')
        self.conv7 = myConv2D(in_channels, 1, 1, 1, 'linear', norm=False, bias=True)


    def forward(self, feature, dispmap):
        dispmap = dispmap.to(torch.float32).unsqueeze(dim=1)
        feature = torch.cat((feature, dispmap), dim=1)
        x1 = self.conv1(feature)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        refinement = F.tanh(x7) * 10
        return dispmap + refinement


class LeftRightConsistencyFeature(nn.Module):
    def __init__(self, in_channels, batchsize, img_height, img_width, device):
        super(LeftRightConsistencyFeature, self).__init__()
        self.conv1 = myConv2D(in_channels * 2, in_channels, 1, 1, activation='leaky')
        self.conv2 = myConv2D(in_channels, in_channels, 3, 1, activation='leaky')

        B, H, W = batchsize, img_height//8, img_width//8
        self.x_base = torch.linspace(-1, 1, W).repeat(B, H, 1).to(dtype=torch.float32, device=device).detach()
        self.y_base = torch.linspace(-1, 1, H).repeat(B, W, 1).transpose(1, 2).to(dtype=torch.float32, device=device).detach()
        self.W = img_width

    def forward(self, left, right, disp):
        B, C, H, W = left.shape
        disp = F.interpolate(disp.unsqueeze(1), [H, W], mode='nearest')
        disp = disp / (self.W // W)
        disp = disp / W * -2
        flow_field = torch.stack((self.x_base + disp.squeeze(1), self.y_base), dim=3)

        x0 = F.grid_sample(right, flow_field, mode='bilinear', padding_mode='border', align_corners=True)
        x1 = self.conv1(torch.cat((left, x0), dim=1))
        x2 = self.conv2(x1)
        return x2


class HeightAwareFeature(nn.Module):
    def __init__(self, in_channels, batchsize, img_height, img_width, device):
        super(HeightAwareFeature, self).__init__()
        self.offset_create = nn.Sequential(
            myConv2D(in_channels, 1, 3, 1, activation='linear', norm=False, bias=False),
            nn.Sigmoid(),
        )
        self.extract = myConv2D(in_channels * 2, in_channels, 1, 1, activation='leaky')
        
        B, H, W = batchsize, img_height//16, img_width//16
        self.x_base = torch.linspace(-1, 1, W).repeat(B, H, 1).to(dtype=torch.float32, device=device).detach()
        self.y_base = torch.linspace(-1, 1, H).repeat(B, W, 1).transpose(1, 2).to(dtype=torch.float32, device=device).detach()

    def forward(self, x):
        B, _, H, W = x.size()
        offset = self.offset_create(x)
        offset = 0.1 * offset + 0.9 * offset.detach()
        flow_field = torch.stack((self.x_base, self.y_base + offset.squeeze(1)), dim=3)
        output = F.grid_sample(x, flow_field, mode='bilinear', padding_mode='border', align_corners=True)
        output = self.extract(torch.cat((x, output), dim=1))
        return output


class GlobalSPPFeature(nn.Module):
    def __init__(self, in_channels):
        super(GlobalSPPFeature, self).__init__()
        self.conv1 = myConv2D(in_channels, in_channels//2, 1, 1, 'leaky')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 9), stride=1, padding=(3//2,9//2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(5, 15), stride=1, padding=(5//2,15//2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(7, 21), stride=1, padding=(7//2,21//2))
        self.conv2 = myConv2D(in_channels * 2, in_channels//2, 1, 1, 'leaky')
        self.conv3 = myConv2D(in_channels//2, in_channels, 3, 1, 'leaky')
        self.conv4 = myConv2D(in_channels, in_channels//2, 1, 1, 'leaky')

    def forward(self, x):
        x1 = self.conv1(x)
        # SPP
        m1 = self.maxpool1(x1)
        m2 = self.maxpool2(x1)
        m3 = self.maxpool3(x1)
        spp = torch.cat([m3, m2, m1, x1], dim=1)
        # SPP end
        x2 = self.conv2(spp)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x4


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.lrconsistency = LeftRightConsistencyFeature(128, 1, 288, 1280, 'cpu')
        self.heightaware = HeightAwareFeature(256, 1, 288, 1280, 'cpu')
        self.globalspp = GlobalSPPFeature(512)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, left, right, disp):
        L = self.conv1(left)
        L = self.bn1(L)
        L = self.relu(L)
        L = self.maxpool(L)
        L = self.layer1(L)
        xL8 = self.layer2(L)
        fea1 = xL8

        R = self.conv1(right)
        R = self.bn1(R)
        R = self.maxpool(R)
        R = self.layer1(R)
        R = self.layer2(R)
        fea2 = self.lrconsistency(xL8,R,disp)

        xL16 = self.layer3(xL8)
        xL32 = self.layer4(xL16)
        fea3 = self.heightaware(xL16)
        fea4 = self.globalspp(xL32)
        return fea1, fea2, fea3, fea4

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


#img = cv2.imread('C:\Datasets/KITTI_OD/image_2/000008.png')
xleft = torch.randn(1, 3, 288, 1280)
xright = torch.randn(1, 3, 288, 1280)
sgm = cv2.imread('C:\Datasets/KITTI_OD/image_sgm/000008.png')

sgm = torch.from_numpy(sgm).permute(2, 0, 1).to(dtype=torch.float32)
sgm = sgm[0, :, :].unsqueeze(0)

block_class, layers = resnet_spec[18]
model = Resnet(block_class, layers)
fea1, fea2, fea3, fea4 = model(xleft, xright, sgm)
