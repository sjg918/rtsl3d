
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VFELayer(nn.Module):
    # from
    # https://github.com/Vegeta2020/SE-SSD/
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels//2, bias=False)
        self.norm = nn.BatchNorm1d(out_channels//2)
        self.mish = nn.Mish()
    def forward(self, inputs):
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = self.mish(x)
        # element wise max pool
        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        repeated = aggregated.repeat(1, voxel_count, 1)
        concatenated = torch.cat([pointwise, repeated], dim=2)
        return concatenated

class VoxelFeatureExtractor(nn.Module):
    # from
    # https://github.com/Vegeta2020/SE-SSD/
    def __init__(self, in_channels, voxelshape):
        super(VoxelFeatureExtractor, self).__init__()
        self.vfe1 = VFELayer(in_channels + 3, 16)
        self.vfe2 = VFELayer(16, 64)
        self.linear1 = nn.Linear(64, 64, bias=False)
        self.norm1 = nn.BatchNorm1d(64)
        self.mish = nn.Mish()

        self.vfe3 = VFELayer(64, 128)
        self.linear2 = nn.Linear(128, 128, bias=False)
        self.norm2 = nn.BatchNorm1d(128)
        self.idxmat = 65535 * torch.ones((voxelshape[1], voxelshape[2]), dtype=torch.long).detach()
        self.voxelshape = voxelshape

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        print(actual_num.shape)
        # tiled_actual_num: [N, M, 1]
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        print(max_num_shape)
        max_num = torch.arange(max_num, dtype=torch.long, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        print(paddings_indicator.shape)
        return paddings_indicator

    def voxel2bev(self, voxels, coors, max_voxels=40, num_features=4):
        bevs = torch.zeros((voxels.shape[0], max_voxels, num_features), dtype=torch.float32, device=voxels.device)
        bevcoors = torch.zeros((voxels.shape[0], 2), dtype=torch.long, device=voxels.device)
        num_voxels_per_bev = torch.zeros((voxels.shape[0]), dtype=torch.long, device=voxels.device)
        print('num_voxel', num_voxels_per_bev.shape)
        N = voxels.shape[0]
        ndim = 2
        ndim_minus_1 = ndim - 1
        coor = torch.zeros((2, 1), dtype=torch.long, device=voxels.device)
        bev_num = torch.zeros((1), dtype=torch.long)
        for i in range(N):
            coor[0], coor[1] = coors[i, 1], coors[i, 2]
            bevidx = self.idxmat[coor[0], coor[1]]
            if bevidx == 65535:
                bevidx = bev_num
                bev_num = bev_num + 1
                self.idxmat[coor[0], coor[1]] = bevidx
                bevcoors[bevidx, 0] = coor[0]
                bevcoors[bevidx, 1] = coor[1]
            num = num_voxels_per_bev[bevidx]
            bevs[bevidx, num.to(torch.long)] =  voxels[i]
            num_voxels_per_bev[bevidx] += 1
            continue
        print('num_voxel', num_voxels_per_bev.shape)
        bevs = bevs[:bev_num]
        bevcoors = bevcoors[:bev_num]
        num_voxels_per_bev = num_voxels_per_bev[:bev_num]
        print('num_voxel', num_voxels_per_bev.shape)
        self.idxmat = self.idxmat * 0 + 65535
        return bevs, bevcoors, num_voxels_per_bev

    def forward(self, features, coors, num_voxels):
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.to(torch.float32).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        features = torch.cat([features, features_relative], dim=-1)
        voxel_per_points = features.shape[1]

        mask = self.get_paddings_indicator(num_voxels, voxel_per_points, axis=0)
        mask = mask.unsqueeze(-1).to(torch.float32)
        
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear1(x)
        x = self.norm1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.mish(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        bevs, bevcoors, num_voxels_per_bev = self.voxel2bev(voxelwise, coors, self.voxelshape[0], 64)
        bevs_count = bevs.shape[1]
        mask = self.get_paddings_indicator(num_voxels_per_bev, bevs_count, axis=0)
        mask = mask.unsqueeze(-1).to(torch.float32)
        
        x = self.vfe3(bevs)
        x *= mask
        x = self.linear2(x)
        x = self.norm2(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.mish(x)
        x *= mask
        bevwise = torch.max(x, dim=1)[0]
        bevwise_ = torch.zeros((self.voxelshape[1], self.voxelshape[2], bevwise.shape[1]))
        bevwise_[bevcoors[:, 0], bevcoors[:, 1]] = bevwise
        
        return bevwise_.permute(2, 1, 0).contiguous()
