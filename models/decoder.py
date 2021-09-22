
import torch
import torch.nn as nn
import numpy as np
import numba
import math
from torch import stack as tstack
import torch.nn.functional as F

from .iou3d_utils import iou3d_eval

class DecoderWithNMS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = (cfg.bevshape[0]//2, cfg.bevshape[1]//2)
        self.outchannels = cfg.neck_bev_out_channels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long)
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY

        self.vaildconf = cfg.vaildconf
        self.grid_x = torch.arange(cfg.bevshape[0]//2, dtype=torch.float32).repeat(cfg.batchsize, cfg.bevshape[1]//2, 1)
        self.grid_y = torch.arange(cfg.bevshape[1]//2, dtype=torch.float32).repeat(cfg.batchsize, cfg.bevshape[0]//2, 1).permute(0, 2, 1).contiguous()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.killiou = cfg.killiou

    def forward(self, output, target):
        # output shape : [batch, cfg.neck_bev_out_channels, cfg.bevshape[0], cfg.bevshape[1]]
        # target : list.-> len(batch).{ [numpy array.]-> number of objects(n) with coord(3) dims(3) ry(1) }
        
        output = output.permute(0, 2, 3, 1).contiguous()
        vaildmask = output[..., 0].sigmoid() > self.vaildconf

        output[..., :4] = output[..., :4].sigmoid()
        output[..., 7:] = output[..., 7:].tanh()

        pred = torch.zeros((self.batchsize, self.outsize[0], self.outsize[1], 8), dtype=torch.float32, device=output.device)
        pred[..., 0] = output[..., 0]
        pred[..., 1] = output[..., 1] + self.grid_x
        pred[..., 2] = output[..., 2] + self.grid_y
        pred[..., 3] = output[..., 3] * self.zsize
        pred[..., 4] = output[..., 4].exp() * self.meandims[0]
        pred[..., 5] = output[..., 5].exp() * self.meandims[1]
        pred[..., 6] = output[..., 6].exp() * self.meandims[2]
        pred[..., 7] = torch.atan2(output[..., 7], output[..., 8])
        
        vaildmask = pred[..., 0].sigmoid() > self.vaildconf
        pred = pred[vaildmask].view(-1, 8)
        _, sortidx = torch.sort(pred[:, 0], dim=0, descending=True)
        sortpred = pred[idx]

        cnt = 0
        while True:
            if cnt == sortpred.shape[0]:
                break
            
            if cnt > 0:
                box0 = sortpred[:cnt]
            box1 = sortpred[cnt]
            box2 = sortpred[cnt+1:]

            _, _, iou3d = iou3d_eval(box2, box1)
            vaildmask = iou3d.view(-1) > self.killiou

            cnt = cnt + 1
            continue

        return loss
