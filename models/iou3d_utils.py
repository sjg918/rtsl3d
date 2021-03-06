
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
