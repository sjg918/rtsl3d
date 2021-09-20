
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1 = FELayer(in_points * in_channels, out_channels, 'mish')

    def forward(self, x):
        x1 = self.fc1(x.view(-1, self.in_points * self.in_channels).contiguous())
        return x1

class NormalFELayer(nn.Module):
    def __init__(self, in_points, in_channels, out_channels):
        super(NormalFELayer, self).__init__()
        self.in_points = in_points
        self.in_channels = in_channels
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1 = FELayer(in_channels, out_channels, 'mish')
        self.fc2 = FELayer(out_channels, out_channels, 'mish')
        self.fc3 = FELayer(in_points * out_channels, out_channels, 'mish')
        self.fc4 = FELayer(out_channels, out_channels, 'mish')

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1).view(-1, self.in_points * self.out_channels)
        x3 = self.fc3(x2)
        return self.fc4(x3)

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(VoxelFeatureExtractor, self).__init__()
        self.bevshape = cfg.bevshape
        self.bev_in_channels = cfg.back_bev_in_channels

        self.poorvfe = PoorFELayer(cfg.p2v_poor, cfg.back_in_channels, cfg.back_voxelwise_channels)
        self.normvfe = NormalFELayer(cfg.p2v_maxpoints, cfg.back_in_channels, cfg.back_voxelwise_channels)
        
        self.poorbfe = PoorFELayer(cfg.v2b_poor, cfg.back_voxelwise_channels, cfg.back_bev_in_channels // 2)
        self.normbfe = NormalFELayer(cfg.v2b_normal, cfg.back_voxelwise_channels, cfg.back_bev_in_channels)
        self.richbfe = RichFELayer(cfg.v2b_maxvoxels, cfg.back_voxelwise_channels, cfg.back_bev_in_channels * 2)

    def forward(self, poorvoxels, normvoxels,\
            poorbevsidx, normbevsidx, richbevsidx, poorcoors, normcoors, richcoors):

        # step1.
        poorv = self.poorvfe(poorvoxels)
        normv = self.normvfe(normvoxels)
        concv = torch.cat((poorv, normv), dim=0)

        poorb = concv[poorbevsidx]
        normb = concv[normbevsidx]
        richb = concv[richbevsidx]
        
        # step2.
        poorb = concv[poorbevsidx]
        normb = concv[normbevsidx]
        richb = concv[richbevsidx]
        
        # step3.
        poorb = self.poorbfe(poorb)
        normb = self.normbfe(normb)
        richb = self.richbfe(richb)

        # step4.
        bevmap = torch.zeros((self.bevshape[0], self.bevshape[1], self.bev_in_channels * 2),
                             dtype=torch.float32, device=poorvoxels.device)
        bevmap[poorcoors[:, 0], poorcoors[:, 1], :16] = poorb
        bevmap[normcoors[:, 0], normcoors[:, 1], :32] = normb
        bevmap[richcoors[:, 0], richcoors[:, 1]] = richb
        
        return bevmap.permute(2, 1, 0).contiguous()
