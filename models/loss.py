
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .iou3d_utils import iou3d_eval

class MultiLoss(nn.Module):
    def __init__(self, cfg, cuda_id):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = (cfg.bevshape[0]//2, cfg.bevshape[1]//2)
        self.grid_y = torch.arange(cfg.bevshape[0]//2, dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, cfg.bevshape[1]//2, 1).detach()#.cpu()
        self.grid_x = torch.arange(cfg.bevshape[1]//2, dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, cfg.bevshape[0]//2, 1).permute(0, 2, 1).contiguous().detach()#.cpu()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.outchannels = cfg.neck_bev_out_channels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0],\
            [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long, device='cuda:'+str(cuda_id)).detach()#.cpu()
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY
        self.vaildiou = cfg.vaildiou

    def forward(self, output, target):
        # output shape : [batch, cfg.neck_bev_out_channels, cfg.bevshape[0], cfg.bevshape[1]]
        # target : list.-> len(batch).{ [numpy array.]-> number of objects(n) with coord(3) dims(3) ry(1) }
        
        output = output.permute(0, 2, 3, 1).contiguous()
        output[..., :4] = output[..., :4].sigmoid()
        output[..., 7:] = output[..., 7:].tanh()

        pred = torch.zeros((self.batchsize, self.outsize[1], self.outsize[0], 7), dtype=torch.float32, device=output.device)
        pred[..., 0] = output[..., 1] + self.grid_x
        pred[..., 1] = output[..., 2] + self.grid_y
        pred[..., 2] = output[..., 3] * self.zsize
        pred[..., 3] = output[..., 4].exp() * self.meandims[0]
        pred[..., 4] = output[..., 5].exp() * self.meandims[1]
        pred[..., 5] = output[..., 6].exp() * self.meandims[2]
        pred[..., 6] = torch.atan2(output[..., 7], output[..., 8])

        bkg_mask = torch.ones((self.batchsize, self.outsize[1], self.outsize[0]), dtype=torch.bool, device=output.device)
        
        loss = 0
        for i in range(self.batchsize):
            output_, target_ = output[i], target[i]
            discretecoord = target_[:, :2].floor().to(torch.long)

            # background
            bkg_mask[i, discretecoord[:, 0], discretecoord[:, 1]] = False

            dircoord = discretecoord.unsqueeze(1).repeat(1, 8, 1) +\
                self.dir.unsqueeze(0).repeat(discretecoord.shape[0], 1, 1)
            
            vaildcoord_mask = ((dircoord[..., 0] > - 1) & (dircoord[..., 0] < self.maxX)) &\
                ((dircoord[..., 1] > - 1) & (dircoord[..., 1] < self.maxY))
            vaildcoord_mask = vaildcoord_mask.view(dircoord.shape[0], 8)

            for k in range(dircoord.shape[0]):
                x = (dircoord[k, vaildcoord_mask[k]])[:, 0]
                y = (dircoord[k, vaildcoord_mask[k]])[:, 1]
                pred_ = pred[i, x, y, :].view(-1, 7)
                _, _, iou3d = iou3d_eval(pred_, target_[k].view(-1, 7))
                vaildiou_mask = iou3d.view(-1) > self.vaildiou
                if vaildiou_mask.sum() == 0:
                    continue
                x, y = x[vaildiou_mask], y[vaildiou_mask]
                bkg_mask[i, x, y] = False
                continue

            bkg = output[i, :, :, 0][bkg_mask[i].clone()]
            bkgloss = F.binary_cross_entropy(bkg, torch.zeros_like(bkg))

            # 
            tgt = torch.zeros((target_.shape[0], self.outchannels), dtype=torch.float32, device=output.device)
            tgt[:, 0] = 1
            tgt[:, 1] = target_[:, 0] - target_[:, 0].floor()
            tgt[:, 2] = target_[:, 1] - target_[:, 1].floor()
            tgt[:, 3] = target_[:, 2] / self.zsize
            tgt[:, 4] = (target_[:, 3] / (self.meandims[0] + 1e-16)).log()
            tgt[:, 5] = (target_[:, 4] / (self.meandims[1] + 1e-16)).log()
            tgt[:, 6] = (target_[:, 5] / (self.meandims[2] + 1e-16)).log()
            tgt[:, 7]= (math.pi * 2 - target_[:, 6]).sin()
            tgt[:, 8]= (math.pi * 2 - target_[:, 6]).cos()

            out = output[i, discretecoord[:, 0], discretecoord[:, 1], :].view(-1, self.outchannels)
            
            clsloss = F.binary_cross_entropy(out[:, 0], tgt[:, 0])
            xyzloss = F.binary_cross_entropy(out[:, 1:4], tgt[:, 1:4])
            wlhloss = F.mse_loss(out[:, 1:4], tgt[:, 1:4]) * 0.5
            loss_im = F.mse_loss(out[:, 7], tgt[:, 7])
            loss_re = F.mse_loss(out[:, 8], tgt[:, 8])
            loss_im_re = (1. - torch.sqrt(out[:, 7] ** 2 + out[:, 8] ** 2)) ** 2
            loss_im_re_red = loss_im_re.mean()
            loss_eular = loss_im + loss_re + loss_im_re_red

            loss = loss + bkgloss + clsloss + xyzloss + wlhloss + loss_eular
            continue

        return loss

if __name__ == "__main__":

    x = torch.randn(5,8,2)
    
    x[:0]

    df=df
    #box_a = np.array([[21.4, 1., 56.6, 1.52563191, 1.6285674, 3.8831164, 0.],
    #                  [20.0, 2.50, 56.5, 1.62, 1.37, 4.0, 0.]], dtype=np.float32).reshape(-1, 7)
    #box_b = np.array([[20.86000061, 2.50999999, 56.68999863, 1.67999995, 1.38999999, 4.26000023, 3.04999995],
    #                [21.0, 1., 56.0, 1.52563191, 1.6285674,4.0, 0.]], dtype=np.float32).reshape(-1, 7)
 
    #box_a = torch.from_numpy(box_a)
    #box_b = torch.from_numpy(box_b)
 
    #iou3dloss = IoU3DLoss()
    #print(iou3dloss(box_a, box_b))
