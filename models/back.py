
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FELayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(FELayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.BatchNorm1d(out_channels)
        self.mish = nn.Mish()
    def forward(self, input):
        voxel_count = input.shape[1]
        x = self.linear(input)
        if len(input.shape) == 3:
            x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif len(input.shape) == 2:
            x = self.norm(x)
        return self.mish(x)

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()
        self.fe = FELayer(in_channels, out_channels//2)

    def forward(self, input):
        voxel_count = input.shape[1]
        pointwise = self.fe(input)
        # element wise max pool
        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        repeated = aggregated.repeat(1, voxel_count, 1)
        concatenated = torch.cat([pointwise, repeated], dim=2)
        return concatenated

class VoxelFeatureExtractor(nn.Module):
    # from
    # https://github.com/Vegeta2020/SE-SSD/
    def __init__(self, in_channels, voxelshape, bevmerge=5):
        super(VoxelFeatureExtractor, self).__init__()
        self.vfe1 = VFELayer(in_channels + 3, 16)
        self.vfe2 = VFELayer(16, 64)
        self.vfe3 = FELayer(64, 64)
        self.vfe4 = FELayer(64, 64)

        self.voxelshape = voxelshape
        self.bevmerge = bevmerge
        self.bevshape = (voxelshape[1]//self.bevmerge, voxelshape[2]//self.bevmerge)
        self.idxmat = 65535 * torch.ones((self.bevshape[0], self.bevshape[1]), dtype=torch.long).detach()
        self.bevin_channels = voxelshape[0] * bevmerge * bevmerge

        self.bfe1_01 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_01 = FELayer(self.bevin_channels, 2)
        self.bfe1_02 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_02 = FELayer(self.bevin_channels, 2)
        self.bfe1_03 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_03 = FELayer(self.bevin_channels, 2)
        self.bfe1_04 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_04 = FELayer(self.bevin_channels, 2)
        self.bfe1_05 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_05 = FELayer(self.bevin_channels, 2)
        self.bfe1_06 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_06 = FELayer(self.bevin_channels, 2)
        self.bfe1_07 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_07 = FELayer(self.bevin_channels, 2)
        self.bfe1_08 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_08 = FELayer(self.bevin_channels, 2)
        self.bfe1_09 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_09 = FELayer(self.bevin_channels, 2)
        self.bfe1_10 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_10 = FELayer(self.bevin_channels, 2)
        self.bfe1_11 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_11 = FELayer(self.bevin_channels, 2)
        self.bfe1_12 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_12 = FELayer(self.bevin_channels, 2)
        self.bfe1_13 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_13 = FELayer(self.bevin_channels, 2)
        self.bfe1_14 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_14 = FELayer(self.bevin_channels, 2)
        self.bfe1_15 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_15 = FELayer(self.bevin_channels, 2)
        self.bfe1_16 = FELayer(self.bevin_channels, self.bevin_channels)
        self.bfe2_16 = FELayer(self.bevin_channels, 2)

        self.bfe3 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish()
            )
        self.bfe4 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish()
            )

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # tiled_actual_num: [N, M, 1]
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.long, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        return paddings_indicator

    def voxel2bev(self, voxels, coors, max_voxels=1000, num_features=64):
        bevs = torch.zeros((self.bevshape[0]*self.bevshape[1], max_voxels, num_features), dtype=torch.float32, device=coors.device)
        bevcoors = torch.zeros((self.bevshape[0]*self.bevshape[1], 2), dtype=torch.long, device=coors.device).detach()
        num_voxels_ = torch.zeros((self.bevshape[0]*self.bevshape[1]), dtype=torch.long, device=coors.device).detach()
        
        N = voxels.shape[0]
        ndim = 2
        ndim_minus_1 = ndim - 1
        coor = torch.zeros((2, 1), dtype=torch.long, device=coors.device).detach()
        bev_num = torch.zeros((1), dtype=torch.long, device=coors.device).detach()

        for i in range(N):
            coor[0], coor[1] = coors[i, 1] // self.bevmerge, coors[i, 2] // self.bevmerge
            bevidx = self.idxmat[coor[0], coor[1]]
            if bevidx == 65535:
                bevidx = bev_num
                bev_num = bev_num + 1
                self.idxmat[coor[0], coor[1]] = bevidx
                bevcoors[bevidx, 0] = coor[0]
                bevcoors[bevidx, 1] = coor[1]
            num = num_voxels_[bevidx]
            bevs[bevidx, num.to(torch.long)] =  voxels[i].to(coors.device)
            num_voxels_[bevidx] += 1
            continue
        
        bevs = bevs[:bev_num]
        bevcoors = bevcoors[:bev_num]
        num_voxels_ = num_voxels_[:bev_num]
        
        self.idxmat = self.idxmat * 0 + 65535
        return bevs, bevcoors, num_voxels_

    def forward(self, features, coors, num_voxels):
        # step1.
        voxel_per_points = features.shape[1]
        mask = self.get_paddings_indicator(num_voxels, voxel_per_points, axis=0)
        mask = mask.unsqueeze(-1).to(torch.float32).to(features.device)
        x1 = self.vfe1(features)
        x2 = self.vfe2(x1) * mask
        x3 = self.vfe3(x2)
        x4 = self.vfe4(x3) * mask + x2
        voxelwise = torch.max(x4, dim=1)[0]
        print('hi')
        # step2.
        bevs, bevcoors, num_voxels_ = self.voxel2bev(voxelwise, coors, self.bevin_channels, 64)
        x0 = bevs.to(features.device).permute(0, 2, 1).contiguous()
        print('hi')
        # step3.
        bevs_per_voxels = bevs.shape[1]
        mask = self.get_paddings_indicator(num_voxels_, bevs_per_voxels, axis=0)
        mask = mask.unsqueeze(dim=1).to(torch.float32).to(features.device)
        print('hi')
        x1 = self.bfe1_01(x0[:, 0:4, :]) * mask
        x1 = self.bfe2_01(x1)
        x2 = self.bfe1_02(x0[:, 4:8, :]) * mask
        x2 = self.bfe2_02(x2)
        x3 = self.bfe1_03(x0[:, 8:12, :]) * mask
        x3 = self.bfe2_03(x3)
        x4 = self.bfe1_04(x0[:, 12:16, :]) * mask
        x4 = self.bfe2_04(x4)
        x5 = self.bfe1_05(x0[:, 16:20, :]) * mask
        x5 = self.bfe2_05(x5)
        x6 = self.bfe1_06(x0[:, 20:24, :]) * mask
        x6 = self.bfe2_06(x6)
        x7 = self.bfe1_07(x0[:, 24:28, :]) * mask
        x7 = self.bfe2_07(x7)
        x8 = self.bfe1_08(x0[:, 28:32, :]) * mask
        x8 = self.bfe2_08(x8)
        x9 = self.bfe1_09(x0[:, 32:36, :]) * mask
        x9 = self.bfe2_09(x9)
        x10 = self.bfe1_10(x0[:, 36:40, :]) * mask
        x10 = self.bfe2_10(x10)
        x11 = self.bfe1_11(x0[:, 40:44, :]) * mask
        x11 = self.bfe2_11(x11)
        x12 = self.bfe1_12(x0[:, 44:48, :]) * mask
        x12 = self.bfe2_12(x12)
        x13 = self.bfe1_13(x0[:, 48:52, :]) * mask
        x13 = self.bfe2_13(x13)
        x14 = self.bfe1_14(x0[:, 52:56, :]) * mask
        x14 = self.bfe2_14(x14)
        x15 = self.bfe1_15(x0[:, 56:60, :]) * mask
        x15 = self.bfe2_15(x15)
        x16 = self.bfe1_16(x0[:, 60:64, :]) * mask
        x16 = self.bfe2_16(x16)

        x17 = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,
                         x9,x10,x11,x12,x13,x14,x15,x16], dim=-1)
        x17 = x17.view([x0.shape[0], 128]).contiguous()
        print('hi')
        # step4.
        x18 = self.bfe3(x17)
        x19 = self.bfe3(x18)
        bevmap = torch.zeros((self.bevshape[0], self.bevshape[1], 128), dtype=torch.float32, device=x17.device)
        bevmap[bevcoors[:, 0], bevcoors[:, 1]] = x19
        print('hi')
        return bevmap.permute(2, 1, 0).contiguous()
