
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# refer
# https://github.com/Vegeta2020/SE-SSD/

class FELayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation, norm=True, bias=False):
        super(FELayer, self).__init__()
        self.layer = nn.ModuleList()

        self.layer.append(nn.Linear(in_channels, out_channels, bias=bias))
        if norm:
            self.layer.append(nn.LayerNorm(out_channels))
        if activation == 'mish':
            self.layer.append(nn.Mish())
        elif activation == 'linear':
            pass

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x

class PoorFELayer(nn.Module):
    def __init__(self, in_points, in_channels, out_channels):
        super(PoorFELayer, self).__init__()
        self.in_points = in_points
        self.out_channels = out_channels
        self.fc1 = FELayer(in_points * in_channels, out_channels, 'mish')

    def forward(self, x):
        x1 = self.fc1(x.view(-1, self.in_points * self.in_channels).contiguous())
        return x1

class NormalFELayer(nn.Module):
    def __init__(self, in_points, in_channels, out_channels):
        super(NormalFELayer, self).__init__()
        self.in_points = in_points
        self.out_channels = out_channels
        self.fc1 = FELayer(in_channels, out_channels, 'mish')
        self.fc2 = FELayer(in_points * out_channels, out_channels, 'mish')

    def forward(self, x):
        x1 = self.fc1(x).view(-1, self.in_points * self.out_channels)
        return self.fc2(x1)

class RichFELayer(nn.Module):
    def __init__(self, in_points, in_channels, out_channels):
        super(RichFELayer, self).__init__()
        self.in_points = in_points
        self.out_channels = out_channels
        self.fc1 = FELayer(in_channels, out_channels, 'mish')
        self.fc2 = FELayer(out_channels, out_channels, 'mish')
        self.fc3 = FELayer(in_points * out_channels, out_channels, 'mish')
        self.fc4 = FELayer(out_channels, out_channels, 'mish')

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1).view(-1, self.in_points * self.out_channels)
        x3 = self.fc3(x2)
        return self.fc4(x1)

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(VoxelFeatureExtractor, self).__init__()
        ##self.back_bev_init_shape = cfg.back_bev_init_shape
        #self.in_channels = cfg.back_in_channels
        #self.voxelwise_channels = cfg.back_voxelwise_channels
        #self.bevin_channels = cfg.back_bev_in_channels
        ##self.bev_channels = cfg.back_bev_channels
        #self.bevshape = cfg.bevshape

        self.poorvfe = PoorFELayer(cfg.p2v_poor, cfg.back_in_channels, cfg.back_voxelwise_channels)
        self.normvfe = NormalFELayer(cfg.p2v_maxpoints, cfg.back_voxelwise_channels, cfg.back_voxelwise_channels)
        
        self.poorbfe = PoorFELayer(cfg.v2b_poor, cfg.back_voxelwise_channels, cfg.back_bev_in_channels // 2)
        self.normbfe = NormalFELayer(cfg.v2b_normal, cfg.back_voxelwise_channels, cfg.back_bev_in_channels)
        self.richbfe = RichFELayer(cfg.v2b_maxvoxels, cfg.back_voxelwise_channels, cfg.back_bev_in_channels * 2)

    def forward(self, voxels, voxelmask, bevsidx, bevcoors, bevmask):
        # step1.
        x1 = self.vfe1(voxels)
        x2 = self.vfe2(x1) * voxelmask
        x3 = self.vfe3(x2)
        x4 = self.vfe4(x3) * voxelmask + x2
        voxelwise = torch.max(x4, dim=1)[0]
        
        # step2.
        bevs = torch.zeros((bevsidx.shape[0], self.bevin_channels, self.voxelwise_channels),\
           dtype=torch.float32, device=voxels.device)
        bevs[:, :] = voxelwise[bevsidx[:, :]]
        
        # step3.
        x0 = bevs.permute(0, 2, 1).contiguous()
        x1 = self.bfe1_01(x0[:, 0:4, :]) * bevmask
        x1 = self.bfe2_01(x1)
        x2 = self.bfe1_02(x0[:, 4:8, :]) * bevmask
        x2 = self.bfe2_02(x2)
        x3 = self.bfe1_03(x0[:, 8:12, :]) * bevmask
        x3 = self.bfe2_03(x3)
        x4 = self.bfe1_04(x0[:, 12:16, :]) * bevmask
        x4 = self.bfe2_04(x4)
        x5 = self.bfe1_05(x0[:, 16:20, :]) * bevmask
        x5 = self.bfe2_05(x5)
        x6 = self.bfe1_06(x0[:, 20:24, :]) * bevmask
        x6 = self.bfe2_06(x6)
        x7 = self.bfe1_07(x0[:, 24:28, :]) * bevmask
        x7 = self.bfe2_07(x7)
        x8 = self.bfe1_08(x0[:, 28:32, :]) * bevmask
        x8 = self.bfe2_08(x8)
        x9 = self.bfe1_09(x0[:, 32:36, :]) * bevmask
        x9 = self.bfe2_09(x9)
        x10 = self.bfe1_10(x0[:, 36:40, :]) * bevmask
        x10 = self.bfe2_10(x10)
        x11 = self.bfe1_11(x0[:, 40:44, :]) * bevmask
        x11 = self.bfe2_11(x11)
        x12 = self.bfe1_12(x0[:, 44:48, :]) * bevmask
        x12 = self.bfe2_12(x12)
        x13 = self.bfe1_13(x0[:, 48:52, :]) * bevmask
        x13 = self.bfe2_13(x13)
        x14 = self.bfe1_14(x0[:, 52:56, :]) * bevmask
        x14 = self.bfe2_14(x14)
        x15 = self.bfe1_15(x0[:, 56:60, :]) * bevmask
        x15 = self.bfe2_15(x15)
        x16 = self.bfe1_16(x0[:, 60:64, :]) * bevmask
        x16 = self.bfe2_16(x16)

        x17 = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,
                         x9,x10,x11,x12,x13,x14,x15,x16], dim=-1)
        x17 = x17.view([x0.shape[0], 128]).contiguous()
        
        # step4.
        x18 = self.bfe3(x17)
        x19 = self.bfe3(x18) + x17
        bevmap = torch.zeros((self.bevshape[0], self.bevshape[1], 128),
                             dtype=torch.float32, device=voxels.device)
        bevmap[bevcoors[:, 0], bevcoors[:, 1]] = x19
        
        return bevmap.permute(2, 1, 0).contiguous()
