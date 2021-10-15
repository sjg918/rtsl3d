
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .iou3d_utils import iou3d_eval

class MultiLoss(nn.Module):
    def __init__(self, cfg, cuda_id):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = cfg.bfa_outshape
        self.grid_y = torch.arange(self.outsize[1], dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, self.outsize[0], 1).detach()#.cpu()
        self.grid_x = torch.arange(self.outsize[0], dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, self.outsize[1], 1).permute(0, 2, 1).contiguous().detach()#.cpu()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.outchannels = cfg.bfa_outchannels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0],\
            [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long, device='cuda:'+str(cuda_id)).detach()#.cpu()
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY
        self.vaildiou = cfg.vaildiou
        self.reduction = cfg.lossreduction
        self._alpha = cfg.alpha
        self._gamma = cfg.gamma

    def mySigmoidFocalLoss(self, input, target, reduction='mean'):
        input = input.sigmoid()
        input = input.flatten().unsqueeze(1)
        target = target.flatten().unsqueeze(1)
        CE_loss = target * input + (1 - target) * (1 - input)
        pt = torch.clamp(CE_loss, min=1e-10, max=1.0)
        alpha_weight = self._alpha * target + (1 - self._alpha) * (1 - target)
        out_loss = -alpha_weight * ((1-pt)**self._gamma) * torch.log(pt)
        if reduction == 'mean':
            return out_loss.mean()
        elif reduction == 'sum':
            return out_loss.sum()

    def ObjclsSigmoidFocalcrossentropyLoss(self, prediction_tensor, target_tensor, reduction='mean'):
        prediction_tensor = prediction_tensor.flatten().unsqueeze(1)
        target_tensor = target_tensor.flatten().unsqueeze(1)
        per_entry_cross_ent = torch.clamp(prediction_tensor, min=0) - prediction_tensor * target_tensor.type_as(prediction_tensor)
        per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(prediction_tensor)))

        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities))

        modulating_factor = torch.pow(1.0 - p_t, self._gamma)

        alpha_weight_factor = 1.0
        alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha)

        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
        if reduction == 'mean':
            return focal_cross_entropy_loss.mean()
        elif reduction == 'sum':
            return focal_cross_entropy_loss.sum()

    def Sinsmoothl1Loss(self, input, target, reduction='mean'):
        v = (input - target).sin()
        mask = v < 1
        v[mask] = v[mask] * v[mask] * 0.5
        v[mask != True] = v[mask != True] - 0.5
        if reduction == 'mean':
            return v.sum() / v.numel()
        elif reduction == 'sum':
            return v.sum()

    def forward(self, output, target):
        # output shape : [batch, cfg.neck_bev_out_channels, cfg.bevshape[0], cfg.bevshape[1]]
        # target : list.-> len(batch).{ [numpy array.]-> number of objects(n) with coord(3) dims(3) ry(1) }
        
        output = output.permute(0, 2, 3, 1).contiguous()

        # pred = torch.zeros((self.batchsize, self.outsize[0], self.outsize[1], 7), dtype=torch.float32, device=output.device)
        # pred[..., 0] = output[..., 1] + self.grid_x
        # pred[..., 1] = output[..., 2] + self.grid_y
        # pred[..., 2] = output[..., 3] * self.zsize
        # pred[..., 3] = output[..., 4].exp() * self.meandims[0]
        # pred[..., 4] = output[..., 5].exp() * self.meandims[1]
        # pred[..., 5] = output[..., 6].exp() * self.meandims[2]
        # pred[..., 6] = torch.atan2(output[..., 7], output[..., 8])

        #bkg_mask = torch.ones((self.batchsize, self.outsize[0], self.outsize[1]), dtype=torch.bool, device=output.device)
        
        objclsloss, xyzloss, hwlloss, angleclsloss, anglesinloss = 0, 0, 0, 0, 0
        for i in range(self.batchsize):
            target_ = target[i]
            tgtobjcls = torch.zeros((self.outsize[0], self.outsize[1]), dtype=torch.float32, device=output.device)

            # target scale !!!!!
            target_[:, 0] = target_[:, 0] * 2
            target_[:, 1] = target_[:, 1] * 2

            #iou3ds = torch.zeros((target_.shape[0], 1), dtype=torch.float32, device=output.device)
            discretecoord = target_[:, :2].floor().to(torch.long)

            tgtobjcls[discretecoord[:, 0], discretecoord[:, 1]] = 1

            # for k in range(discretecoord.shape[0]):
            #     pred_ = pred[i, discretecoord[k, 0], discretecoord[k, 1], :]
            #     _, _, iou3d = iou3d_eval(pred_.view(-1, 7), target_[k].view(-1, 7))
            #     iou3ds[k] = 1 - iou3d
            #     continue

            # dircoord = discretecoord.unsqueeze(1).repeat(1, 8, 1) +\
            #     self.dir.unsqueeze(0).repeat(discretecoord.shape[0], 1, 1)
            
            # vaildcoord_mask = ((dircoord[..., 0] > - 1) & (dircoord[..., 0] < self.maxX)) &\
            #     ((dircoord[..., 1] > - 1) & (dircoord[..., 1] < self.maxY))
            # vaildcoord_mask = vaildcoord_mask.view(dircoord.shape[0], 8)

            # for k in range(dircoord.shape[0]):
            #     x = (dircoord[k, vaildcoord_mask[k]])[:, 0]
            #     y = (dircoord[k, vaildcoord_mask[k]])[:, 1]
            #     pred_ = pred[i, x, y, :].view(-1, 7)
            #     _, _, iou3d = iou3d_eval(pred_, target_[k].view(-1, 7))
            #     vaildiou_mask = iou3d.view(-1) > self.vaildiou
            #     if vaildiou_mask.sum() == 0:
            #         continue
            #     x, y = x[vaildiou_mask], y[vaildiou_mask]
            #     bkg_mask[i, x, y] = False
            #     continue
            
            # if self.reduction == 'mean':
            #     iou3dloss = iou3ds.mean()
            # elif self.reduction == 'sum':
            #     iou3dloss = iou3ds.sum()
            

            # cls xyz hwl ry LOSS
            tgt = torch.zeros((target_.shape[0], self.outchannels), dtype=torch.float32, device=output.device)
            tgt[:, 1] = target_[:, 0] - target_[:, 0].floor()
            tgt[:, 2] = target_[:, 1] - target_[:, 1].floor()
            tgt[:, 3] = target_[:, 2] / self.zsize
            tgt[:, 4] = (target_[:, 3] / self.meandims[0]).log()
            tgt[:, 5] = (target_[:, 4] / self.meandims[1]).log()
            tgt[:, 6] = (target_[:, 5] / self.meandims[2]).log()
            tgtangleclass = (target_[:, 6] >= 0).to(dtype=torch.long, device=output.device)
            tgt[:, 8] = target_[:, 6].sin()

            out = output[i, discretecoord[:, 0], discretecoord[:, 1], :].view(-1, self.outchannels)

            objclsloss = objclsloss + self.ObjclsSigmoidFocalcrossentropyLoss(output[i, :, :, 0], tgtobjcls, reduction=self.reduction)
            xyzloss = xyzloss + F.mse_loss(out[:, 1:4].sigmoid(), tgt[:, 1:4], reduction=self.reduction)
            hwlloss = hwlloss + F.mse_loss(out[:, 4:7], tgt[:, 4:7], reduction=self.reduction) * 0.5
            angleclsloss = angleclsloss + F.cross_entropy(out[:, 7:9], tgtangleclass, reduction=self.reduction)
            anglesinloss = anglesinloss + self.Sinsmoothl1Loss(out[:, 9], tgt[:, 8], reduction=self.reduction)
            continue

        loss = objclsloss + xyzloss + hwlloss + angleclsloss + anglesinloss
        return loss, objclsloss, xyzloss, hwlloss, angleclsloss, anglesinloss


class MultiLossBce(nn.Module):
    def __init__(self, cfg, cuda_id):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = cfg.bfa_outshape
        self.grid_y = torch.arange(self.outsize[1], dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, self.outsize[0], 1).detach()#.cpu()
        self.grid_x = torch.arange(self.outsize[0], dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, self.outsize[1], 1).permute(0, 2, 1).contiguous().detach()#.cpu()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.outchannels = cfg.bfa_outchannels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0],\
            [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long, device='cuda:'+str(cuda_id)).detach()#.cpu()
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY
        self.vaildiou = cfg.vaildiou
        self.reduction = cfg.lossreduction
        self._alpha = cfg.alpha
        self._gamma = cfg.gamma

    def Sinsmoothl1Loss(self, input, target, reduction='mean'):
        v = (input - target).sin()
        mask = v < 1
        v[mask] = v[mask] * v[mask] * 0.5
        v[mask != True] = v[mask != True] - 0.5
        if reduction == 'mean':
            return v.sum() / v.numel()
        elif reduction == 'sum':
            return v.sum()

    def forward(self, output, target):
        output = output.permute(0, 2, 3, 1).contiguous()

        
        bkgloss, objclsloss, xyzloss, hwlloss, angleclsloss, anglesinloss = 0, 0, 0, 0, 0, 0
        for i in range(self.batchsize):
            target_ = target[i]
            tgtobjcls = torch.ones((self.outsize[0], self.outsize[1]), dtype=torch.bool, device=output.device)

            # target scale !!!!!
            target_[:, 0] = target_[:, 0] * 2
            target_[:, 1] = target_[:, 1] * 2

            discretecoord = target_[:, :2].floor().to(torch.long)
            tgtobjcls[discretecoord[:, 0], discretecoord[:, 1]] = False

            bkg = output[i, :, :, 0][tgtobjcls].sigmoid()
            clsscale = bkg.mean()
            clsscale = (clsscale * -3).exp() * 2.5
            bkgmask = bkg > 0.025
            if bkgmask.sum() == 0:
                pass
            else:
                bkg = bkg[bkgmask]
                bkgloss = bkgloss + F.binary_cross_entropy(bkg, torch.zeros(bkg.shape, dtype=torch.float32, device=output.device), reduction=self.reduction)

            # cls xyz hwl ry LOSS
            tgt = torch.zeros((target_.shape[0], self.outchannels), dtype=torch.float32, device=output.device)
            tgt[:, 0] = 1
            tgt[:, 1] = target_[:, 0] - target_[:, 0].floor()
            tgt[:, 2] = target_[:, 1] - target_[:, 1].floor()
            tgt[:, 3] = target_[:, 2] / self.zsize
            tgt[:, 4] = (target_[:, 3] / self.meandims[0]).log()
            tgt[:, 5] = (target_[:, 4] / self.meandims[1]).log()
            tgt[:, 6] = (target_[:, 5] / self.meandims[2]).log()
            tgtangleclass = (target_[:, 6] >= 0).to(dtype=torch.long, device=output.device)
            tgt[:, 8] = target_[:, 6].sin()

            out = output[i, discretecoord[:, 0], discretecoord[:, 1], :].view(-1, self.outchannels)

            objclsloss = objclsloss + F.binary_cross_entropy(out[:, 0].sigmoid(), tgt[:, 0], reduction=self.reduction) * clsscale
            xyzloss = xyzloss + F.mse_loss(out[:, 1:4].sigmoid(), tgt[:, 1:4], reduction=self.reduction)
            hwlloss = hwlloss + F.mse_loss(out[:, 4:7], tgt[:, 4:7], reduction=self.reduction) * 0.5
            angleclsloss = angleclsloss + F.cross_entropy(out[:, 7:9], tgtangleclass, reduction=self.reduction) * 0.5
            anglesinloss = anglesinloss + F.smooth_l1_loss(out[:, 9].sin(), tgt[:, 8].sin(), reduction=self.reduction) * 0.5
            #anglesinloss = anglesinloss + self.Sinsmoothl1Loss(out[:, 9], tgt[:, 8], reduction=self.reduction)
            continue

        loss = bkgloss + objclsloss + xyzloss + hwlloss + angleclsloss + anglesinloss
        return loss, bkgloss, objclsloss, xyzloss, hwlloss, angleclsloss, anglesinloss
