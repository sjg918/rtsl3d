
import torch
import torch.nn as nn
import numpy as np
import numba
import math
from torch import stack as tstack
import torch.nn.functional as F

from .iou3d_utils import iou3d_eval

def roty(t):
        # Rotation about the y-axis.
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.
    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def lidar_to_camera(x, y, z, V2C, R0):
    p = np.array([x, y, z, 1])
    p = np.matmul(V2C, p)
    p = np.matmul(R0, p)
    p = p[0:3]
    return tuple(p)

def lidar_to_camera_box(boxes, V2C, R0):
# (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, ry = lidar_to_camera(x, y, z, V2C=V2C, R0=R0), h, w, l, ry
        # ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)

class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.cls_id = self.cls_type_to_id(self.type)
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = data[15] if data.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def cls_type_to_id(self, cls_type):
        # Car and Van ==> Car class
        # Pedestrian and Person_Sitting ==> Pedestrian Class
        CLASS_NAME_TO_ID = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Van': 0,
            'Person_sitting': 1
        }
        if cls_type not in CLASS_NAME_TO_ID.keys():
            return -1
        return CLASS_NAME_TO_ID[cls_type]

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0], self.t[1], self.t[2], self.ry))

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                    self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.t[0], self.t[1], self.t[2],
                    self.ry, self.score)
        return kitti_str

class DecoderWithNMS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.batchsize = cfg.batchsize
        self.outsize = (cfg.bevshape[0]//2, cfg.bevshape[1]//2)
        self.outchannels = cfg.neck_bev_out_channels
        self.dir = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=torch.long)
        self.minY = cfg.minY
        self.minZ = cfg.minZ
        self.maxX = cfg.outputmaxX
        self.maxY = cfg.outputmaxY
        cuda_id = cfg.cuda_ids[0]

        self.vaildconf = cfg.vaildconf
        self.grid_y = torch.arange(cfg.bevshape[0]//2, dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, cfg.bevshape[1]//2, 1).detach()#.cpu()
        self.grid_x = torch.arange(cfg.bevshape[1]//2, dtype=torch.float32, device='cuda:'+str(cuda_id))\
            .repeat(cfg.batchsize, cfg.bevshape[0]//2, 1).permute(0, 2, 1).contiguous().detach()#.cpu()
        self.zsize = cfg.maxZ - cfg.minZ
        self.meandims = cfg.meandims
        self.vaildconf = cfg.vaildconf

        self.saveresulttxtplace = cfg.saveresulttxtplace

    def forward(self, output, calib, writetxt=True, filenumber=None):
        # output shape : [batch, cfg.neck_bev_out_channels, cfg.bevshape[0], cfg.bevshape[1]]
        
        output = output.permute(0, 2, 3, 1).contiguous()
        pred = torch.zeros((self.batchsize, self.outsize[1], self.outsize[0], 8), dtype=torch.float32, device=output.device)

        pred[..., 0] = output[..., 0].sigmoid()
        pred[..., 1] = output[..., 1].sigmoid() + self.grid_x
        pred[..., 2] = output[..., 2].sigmoid() + self.grid_y + self.minY
        pred[..., 3] = output[..., 3].sigmoid() * self.zsize + self.minZ
        pred[..., 4] = output[..., 4].exp() * self.meandims[0]
        pred[..., 5] = output[..., 5].exp() * self.meandims[1]
        pred[..., 6] = output[..., 6].exp() * self.meandims[2]
        pred[..., 7] = torch.atan2(output[..., 7].tanh(), output[..., 8].tanh())
        
        vaildmask = pred[..., 0] > self.vaildconf
        if vaildmask.sum() == 0:
            if writetxt:
                writefilename = './' + self.saveresulttxtplace + '/' + str(filenumber) + '.txt'
                with open(writefilename, 'w') as det_file:
                    pass
            return 0

        pred = pred[vaildmask].view(-1, 8)
        _, sortidx = torch.sort(pred[:, 0], dim=0, descending=True)
        sortpred = pred[sortidx]
        
        cnt = 0
        while True:
            if cnt >= sortpred.shape[0]:
                break
            
            if cnt > 0:
                box0 = sortpred[:cnt].view(-1, 8)
            box1 = sortpred[cnt].view(-1, 8)
            box2 = sortpred[cnt+1:].view(-1, 8)

            _, _, iou3d = iou3d_eval(box2[:, 1:], box1[:, 1:])
            vaildmask = iou3d.view(-1) < self.vaildconf
            box2 = box2[vaildmask]

            if cnt > 0:
                sortpred = torch.cat([box0, box1, box2], dim=0).contiguous()
            else:
                sortpred = torch.cat([box1, box2], dim=0).contiguous()
            cnt = cnt + 1
            continue

        kittiobjects = []
        corners3d = []
        if sortpred.shape[0]:
            sortpred = sortpred.cpu().numpy()
            sortpred[:, 1:] = lidar_to_camera_box(sortpred[:, 1:], calib.V2C, calib.R0)

        for index, l in enumerate(sortpred):
            strcls = "Car"
            line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % strcls

            obj = Object3d(line)
            obj.t = l[1:4]
            obj.h, obj.w, obj.l = l[4:7]
            obj.ry = l[7]

            _, corners_3d = compute_box_3d(obj, calib.P)
            corners3d.append(corners_3d)
            kittiobjects.append(obj)
        
        if len(corners3d) > 0:
            corners3d = np.array(corners3d)
            img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

            #img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
            #img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
            #img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
            #img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

            #img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
            #img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
            #box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

        for i, obj in enumerate(kittiobjects):
            x, z, ry = obj.t[0], obj.t[2], obj.ry
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            obj.alpha = alpha
            obj.box2d = img_boxes[i, :]

        if writetxt:
            writefilename = './' + self.saveresulttxtplace + '/' + str(filenumber) + '.txt'
            with open(writefilename, 'w') as det_file:
                objects_pred_len = len(kittiobjects)
                for i in range(objects_pred_len):
                    strline = kittiobjects[i].to_kitti_format()
                    if 'Car' in strline:
                        det_file.write(strline)
                        if i == objects_pred_len - 1:
                            continue
                        else:
                            det_file.write('\n')
                    continue
            return 0

        return kittiobjects
