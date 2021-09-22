
import torch
import torch.nn as nn
import numpy as np
import numba
import math
from torch import stack as tstack
import torch.nn.functional as F

def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.
    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners

def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners

def bbox_overlaps(bboxes1, bboxes2, offset, eps):
    """Calculate overlap between two set of bboxes.
    Args:
        bboxes1 (Tensor): shape (m, 4), (xmin, ymin, xmax, ymax)
        bboxes2 (Tensor): shape (n, 4)
    Returns:
        ious(Tensor): shape (m, n).
    """
    left_top = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])      # [m, n, 2]
    right_bottom = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [m, n, 2]

    wh = (right_bottom - left_top + offset).clamp(min=0)            # [m, n, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]                             # [m, n]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + offset) * (bboxes1[:, 3] - bboxes1[:, 1] + offset)   # [m,]
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + offset) * (bboxes2[:, 3] - bboxes2[:, 1] + offset)   # [n,]
    iou_bev = (overlap / (area1[:, None] + area2 - overlap)).clamp(min=eps, max=1-eps)   # [m, n]
    return iou_bev, overlap

def iou3d_eval(pred, target, iou_type="iou3d", offset=1e-6, eps=1e-6):
    """
        IoU: this is not strictly iou, it use standup boxes to cal iou.
            Computing the IoU between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
        Args:
            pred (Tensor): Predicted bboxes of format (x, y, z, w, l, h, ry) in velo coord, shape (m, 7).
            target (Tensor): Corresponding gt bboxes, shape (n, 7).
            eps (float): Eps to avoid log(0).
        Return:
            Tensor: Loss tensor. shape (m, n).
    """
    pred_corner_2d = center_to_corner_box2d(pred[:, 0:2], pred[:, 3:5], pred[:, -1])          # rotated corners: (m, 4, 2)
    pred_corner_sd = corner_to_standup_nd(pred_corner_2d)                                     # (m, 4), (xmin, ymin, xmax, ymax).
    target_corner_2d = center_to_corner_box2d(target[:, 0:2], target[:, 3:5], target[:, -1])  # (n, 4, 2)
    target_corner_sd = corner_to_standup_nd(target_corner_2d)                                 # (n, 4)
    iou_bev, overlap_bev = bbox_overlaps(pred_corner_sd, target_corner_sd, offset, eps)       # (m, n)

    pred_corner_2d = center_to_corner_box2d(pred[:, 1:3], pred[:, 4:6])                  # rotated corners: (m, 4, 2)
    pred_corner_sd = corner_to_standup_nd(pred_corner_2d)                                # (m, 4), (ymin, zmin, ymax, zmax).
    target_corner_2d = center_to_corner_box2d(target[:, 1:3], target[:, 4:6])            # (n, 4, 2)
    target_corner_sd = corner_to_standup_nd(target_corner_2d)                            # (n, 4)
    iou_face, overlap_face = bbox_overlaps(pred_corner_sd, target_corner_sd, offset, eps)  # (m, n)

    iou3d = None
    if iou_type == "iou3d":
        pred_height_min = (pred[:, 2] - pred[:, 5]).view(-1, 1)        # z - h, (m, 1)
        pred_height_max = pred[:, 2].view(-1, 1)                       # z
        target_height_min = (target[:, 2] - target[:, 5]).view(1, -1)  # (1, n)
        target_height_max = target[:, 2].view(1, -1)

        max_of_min = torch.max(pred_height_min, target_height_min)  # (m, 1)
        min_of_max = torch.min(pred_height_max, target_height_max)  # (1, n)
        overlap_h = torch.clamp(min_of_max - max_of_min, min=0)     # (m, n)
        overlap_3d = overlap_bev * overlap_h                        # (m, n)

        pred_vol = (pred[:, 3] * pred[:, 4] * pred[:, 5]).view(-1, 1)  # (m, 1)
        target_vol = (target[:, 3] * target[:, 4] * target[:, 5]).view(1, -1)  # (1, n)  -> broadcast (m, n)
        iou3d = (overlap_3d / torch.clamp(pred_vol + target_vol - overlap_3d, min=eps)).clamp(min=eps, max=1.0)

    return iou_bev, iou_face, iou3d

class IoU3DLoss(nn.Module):
    def __init__(self, iou_type='iou3d', offset=1e-6, eps=1e-6, loss_weight=1.0):
        super(IoU3DLoss, self).__init__()
        self.iou_type = iou_type
        self.eps = eps
        self.offset = offset
        self.loss_weight = loss_weight


    def forward(self, pred, target, weights=None, **kwargs):
        """
            pred: [m, 7], (x,y,z,w,l,h,ry) in velo coord.
            target: [m, 7], (x,y,z,w,l,h,ry) in velo coord.
            iou_type: "iou3d" or "iou_bev"
            Boxes in pred and target should be matched one by one for calculation of iou loss.
        """
        pred = pred.float()
        target = target.float()

        #valid_mask = (pred[:, 3] > 0) & (pred[:, 4] > 0) & (pred[:, 5] > 0)
        #pred = pred[valid_mask]
        #target = target[valid_mask]

        num_pos_pred = pred.shape[0]
        iou_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        if num_pos_pred > 0:
            diag = torch.arange(num_pos_pred)
            iou_bev, iou_face, iou3d = iou3d_eval(pred, target, self.iou_type, self.offset, self.eps)

            if self.iou_type == 'iou3d':
                log_iou3d = -iou3d.log()
                iou_loss = log_iou3d[diag, diag].sum() / num_pos_pred
            else:
                log_iou_face = - iou_face.log()
                log_iou_bev = - iou_bev.log()
                iou_loss = (log_iou_bev[diag, diag].sum() + log_iou_face[diag, diag].sum()) / num_pos_pred

        return iou_loss * self.loss_weight

class MultiLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = (cfg.bevshape[0]//2, cfg.bevshape[1]//2)
        self.grid_x = torch.arange(cfg.bevshape[0]//2, dtype=torch.float32).repeat(cfg.batchsize, cfg.bevshape[1]//2, 1)
        self.grid_y = torch.arange(cfg.bevshape[1]//2, dtype=torch.float32).repeat(cfg.batchsize, cfg.bevshape[0]//2, 1).permute(0, 2, 1).contiguous()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.outchannels = cfg.neck_bev_out_channels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long)
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY
        self.vaildiou = cfg.vaildiou

    def forward(self, output, target):
        # output shape : [batch, cfg.neck_bev_out_channels, cfg.bevshape[0], cfg.bevshape[1]]
        # target : list.-> len(batch).{ [numpy array.]-> number of objects(n) with coord(3) dims(3) ry(1) }
        
        output = output.permute(0, 2, 3, 1).contiguous()
        output[..., :4] = output[..., :4].sigmoid()
        output[..., 7:] = output[..., 7:].tanh()

        pred = torch.zeros((self.batchsize, self.outsize[0], self.outsize[1], 7), dtype=torch.float32, device=output.device)
        pred[..., 0] = output[..., 1] + self.grid_x
        pred[..., 1] = output[..., 2] + self.grid_y
        pred[..., 2] = output[..., 3] * self.zsize
        pred[..., 3] = output[..., 4].exp() * self.meandims[0]
        pred[..., 4] = output[..., 5].exp() * self.meandims[1]
        pred[..., 5] = output[..., 6].exp() * self.meandims[2]
        pred[..., 6] = torch.atan2(output[..., 7], output[..., 8])

        bkg_mask = torch.ones((self.batchsize, self.outsize[0], self.outsize[1]), dtype=torch.bool, device=output.device)
        
        loss = 0
        for i in range(self.batchsize):
            output_, target_ = output[i], target[i]
            discretecoord = target_[:, :2].floor().to(torch.long)

            # background
            bkg_mask[i, discretecoord[:, 1], discretecoord[:, 0]] = False

            dircoord = discretecoord.unsqueeze(1).repeat(1, 8, 1) + self.dir.unsqueeze(0).repeat(dircoord.shape[0], 1, 1)
            vaildcoord_mask = ((dircoord[..., 0] > - 1) & (dircoord[..., 0] < self.maxX)) &\
                ((dircoord[..., 1] > - 1) & (dircoord[..., 1] < self.maxY))

            for k in range(dircoord.shape[0]):
                x = (dircoord[k, vaildcoord_mask[k]])[:, 0]
                y = (dircoord[k, vaildcoord_mask[k]])[:, 1]
                pred_ = pred[i, y, x, :].view([-1, 7])
                _, _, iou3d = iou3d_eval(pred_, target_[k])
                vaildiou_mask = iou3d.view(1) > self.vaildiou
                x, y = x[vaildiou], y[vaildiou]
                bkg_mask[i, y, x] = False
                continue

            bkg = output[i, :, :, 0][bkg_mask[i]]
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

            out = output[i, discretecoord[:, 1], discretecoord[:, 0], :].view(-1, self.outchannels)
            
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
    mask = (x[..., 0] > - 1) & (x[..., 0] < 1)
    print(mask.shape)

    x = (x[0, mask[0]])[:, 0]
    print(x)

    df=df
    box_a = np.array([[21.4, 1., 56.6, 1.52563191, 1.6285674, 3.8831164, 0.],
                      [20.0, 2.50, 56.5, 1.62, 1.37, 4.0, 0.]], dtype=np.float32).reshape(-1, 7)
    box_b = np.array([[20.86000061, 2.50999999, 56.68999863, 1.67999995, 1.38999999, 4.26000023, 3.04999995],
                    [21.0, 1., 56.0, 1.52563191, 1.6285674,4.0, 0.]], dtype=np.float32).reshape(-1, 7)
 
    box_a = torch.from_numpy(box_a)
    box_b = torch.from_numpy(box_b)
 
    iou3dloss = IoU3DLoss()
    print(iou3dloss(box_a, box_b))
