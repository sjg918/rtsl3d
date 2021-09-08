
import torch.utils.data as data
import numba
import random
from torch.nn import functional as F
import math

import os
from PIL import Image
import numpy as np

class Object2Cam(object):
    # refer
    # https://github.com/ghimiredhikura/Complex-YOLOv3
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
        CLASS_NAME_TO_ID = {
            'Car': 0,
            #'Pedestrian': 1,
            #'Cyclist': 2,
            #'Van': 0,
            #'Person_sitting': 1
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
    
    @staticmethod
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

    def compute_box_3d(self, P):
        ''' Takes an object and a projection matrix (P) and projects the 3d
            bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in in rect camera coord.
        '''
        # compute rotational matrix around yaw axis
        R = Calibration.roty(self.ry)

        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + self.t[0]
        corners_3d[1, :] = corners_3d[1, :] + self.t[1]
        corners_3d[2, :] = corners_3d[2, :] + self.t[2]
        # print 'cornsers_3d: ', corners_3d
        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane
        corners_2d = Object2Cam.project_to_image(np.transpose(corners_3d), P)
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)

class Object2Velo(object):
    # refer
    # https://github.com/jeasinema/VoxelNet-tensorflow
    def __init__(self):
        pass

    @staticmethod
    def camera_to_lidar(x, y, z, V2C=None, R0=None, P2=None):
        p = np.array([x, y, z, 1])
        if V2C is None or R0 is None:
            p = np.matmul(cnf.R0_inv, p)
            p = np.matmul(cnf.Tr_velo_to_cam_inv, p)
        else:
            R0_i = np.zeros((4, 4))
            R0_i[:3, :3] = R0
            R0_i[3, 3] = 1
            p = np.matmul(np.linalg.inv(R0_i), p)
            p = np.matmul(Calibration.inverse_rigid_trans(V2C), p)
        p = p[0:3]
        return tuple(p)

    @staticmethod
    def change_box3d_center_(box3d, src, dst):
        dst = np.array(dst, dtype=box3d.dtype)
        src = np.array(src, dtype=box3d.dtype)
        box3d[..., :3] += box3d[..., 3:6] * (dst - src)
        return box3d

    @staticmethod
    def camera_to_lidar_corner(corners, calib):
        cs = []
        for c in corners:
            x, y, z = c
            (x,y,z) = Object2Velo.camera_to_lidar(x, y, z, calib.V2C, calib.R0, calib.P)
            cs.append([x,y,z])
            continue
        cs = np.array(cs).reshape(-1, 3)
        return cs

    @staticmethod
    def camera_to_lidar_box(object2cam, calib):
        bbox_selected = []
        for obj in object2cam:
            if obj.cls_id != -1:
                bbox = []
                bbox.append(obj.cls_id)
                bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry, obj.get_obj_level()])
                bbox_selected.append(bbox)
            continue
        if len(bbox_selected) == 0:
            return None
        else:
            labels = np.array(bbox_selected, dtype=np.float32)        
        bbox_ = []
        cls_ = []
        dif_ = []
        for box in labels:
            cls, x, y, z, h, w, l, ry, dif = box
            (x, y, z) = Object2Velo.camera_to_lidar(x, y, z, V2C=calib.V2C, R0=calib.R0, P2=calib.P)
            bbox_.append([x, y, z, w, l, h, ry])
            cls_.append([cls])
            dif_.append([dif])
            continue
        bbox_ = np.array(bbox_).reshape(-1, 7)
        bbox_ = Object2Velo.change_box3d_center_(bbox_, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
        cls_ = np.array(cls_).reshape(-1, 1)
        dif_ = np.array(dif_).reshape(-1, 1)
        return bbox_, cls_, dif_

class Calibration(object):
    # refer
    # https://github.com/kuangliu/kitti-utils 
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = Calibration.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R_rect': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    def cart2hom(self, pts_3d):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
        return pts_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    @staticmethod
    def rotx(t):
        # 3D Rotation about the x-axis.
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    @staticmethod
    def roty(t):
        # Rotation about the y-axis.
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    @staticmethod
    def rotz(t):
        # Rotation about the z-axis.
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    @staticmethod
    def transform_from_rot_trans(R, t):
        ''' Transforation matrix from rotation matrix and translation vector. '''
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    @staticmethod
    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

class BoundaryCheck(object):
    # refer
    # https://github.com/maudzung/Complex-YOLOv4-Pytorch
    def __init__(self, cfg):
        self.minX = cfg.minX
        self.maxX = cfg.maxX
        self.minY = cfg.minY
        self.maxY = cfg.maxY
        self.minZ = cfg.minZ
        self.maxZ = cfg.maxZ

    def __call__(self, gt_boxes, points):
        mask = np.where((gt_boxes[:, 0] > self.minX) & (gt_boxes[:, 0] < self.maxX) &
                        (gt_boxes[:, 1] > self.minY) & (gt_boxes[:, 1] < self.maxy))
        Gt_Boxes = gt_boxes[mask]

        mask = np.where((points[:, 0] >= self.minX) & (points[:, 0] < self.maxX) &
                        (points[:, 1] >= self.minY) & (points[:, 1] < self.maxY) &
                        (points[:, 2] >= self.minZ) & (points[:, 2] < self.maxZ))
        PointCloud = points[mask]
        #PointCloud[:, 2] = PointCloud[:, 2] - self.minZ
        return Gt_Boxes, PointCloud

class Point2Voxel(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, cfg):
        self.voxelshape = cfg.voxelshape
        self.voxelrange = np.array(cfg.voxelrange)
        self.voxelsize = np.array(cfg.voxelsize)
        self.maxpoints = cfg.maxpoints
        self.maxvoxels = cfg.maxvoxels
        self.idxmat = 65535 * np.ones((self.voxelshape), dtype=np.uint16)

    @staticmethod
    @numba.jit(nopython=True)
    def kernel(
        points, voxel_size, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels):
        N = points.shape[0]

        ndim = 3
        idxtup = (2, 1, 0)
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
        coor = np.zeros(shape=(3,), dtype=np.int32)
        voxel_num = 0
        for i in range(N):
            for j in range(ndim):
                c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
                coor[idxtup[j]] = c
                continue
            voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
            if voxelidx == 65535:
                voxelidx = voxel_num
                if voxel_num >= max_voxels:
                    break
                voxel_num += 1
                coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
                coors[voxelidx] = coor

            num = num_points_per_voxel[voxelidx]
            if num < max_points:
                voxels[voxelidx, num] = points[i]
                num_points_per_voxel[voxelidx] += 1
            continue
        return voxel_num

    def __call__(self, points):
        voxelmap_shape = (self.voxelrange[3:] - self.voxelrange[:3]) / self.voxelsize
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        num_points_per_voxel = np.zeros(shape=(self.maxvoxels,), dtype=np.int32)
        voxels = np.zeros(shape=(self.maxvoxels, self.maxpoints, points.shape[-1]), dtype=points.dtype)
        coors = np.zeros(shape=(self.maxvoxels, 3), dtype=np.int32)
        coord_tovoxelidx = self.idxmat.copy()
        voxel_num = Point2Voxel.kernel(
            points, self.voxelsize, self.voxelrange, num_points_per_voxel,
            coord_tovoxelidx, voxels, coors, self.maxpoints, self.maxvoxels,)
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]
        return voxels, coors, num_points_per_voxel

class GlobalRandomFlip(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, probability=0.5):
        self.p = probability
    def __call__(self, gt_boxes, points):
        if np.random.uniform(0, 1) < self.p:
            if gt_boxes is not None:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
            points[:, 1] = -points[:, 1]
        return gt_boxes, points

class GlobalRandomScaling(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, probability=0.5, minscale=0.95, maxscale=1.05):
        self.p = probability
        self.minscale = minscale
        self.maxscale = maxscale
    def __call__(self, gt_boxes, points):
        if np.random.uniform(0, 1) < self.p:
            noise_scale = np.random.uniform(self.minscale, self.maxscale)
            points[:, :3] *= noise_scale
            if gt_boxes is not None:
                gt_boxes[:, :-1] *= noise_scale
        return gt_boxes, points

class GlobalRandomRotation(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, probability=0.5, minrad=-np.pi / 4, maxrad=np.pi / 4):
        self.p = probability
        self.minrad = minrad
        self.maxrad = maxrad
    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype,
            )
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype,
            )
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype,
            )
        else:
            raise ValueError("axis should in range")
        return points @ rot_mat_T
    def __call__(self, gt_boxes, points):
        if np.random.uniform(0, 1) < self.p:
            noise_rotation = np.random.uniform(self.minrad, self.maxrad)
            points[:, :3] = self.rotation_points_single_angle(points[:, :3], noise_rotation, axis=2)
            if gt_boxes is not None:
                gt_boxes[:, :3] = self.rotation_points_single_angle(gt_boxes[:, :3], noise_rotation, axis=2)
                gt_boxes[:, -1] += noise_rotation
        return gt_boxes, points

class GlobalRandomTranslate(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, probability=0.5, xscale=3.6, yscale=3.0, zscale=0.2):
        self.p = probability
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale
    def __call__(self, gt_boxes, points, noise_translate_std):
        if np.random.uniform(0, 1) < self.p:
            randomx = np.random.uniform(-self.xscale, self.xscale)
            randomy = np.random.uniform(-self.yscale, self.yscale)
            randomz = np.random.uniform(-self.zscale, self.zscale)
            points[:, 0] += randomx
            points[:, 1] += randomy
            points[:, 2] += randomz
            if gt_boxes is not None:
                gt_boxes[:, 0] += randomx
                gt_boxes[:, 1] += randomy
                gt_boxes[:, 2] += randomz
        return gt_boxes, points, noise_translate

class RandomMosaic(object):
    def __init__(self, cfg):
        self.voxelshape = cfg.voxelshape
        self.voxelsize = cfg.voxelsize
        self.minX = cfg.minX
        self.maxX = cfg.maxX
        self.minY = cfg.minY
        self.maxY = cfg.maxY
        self.minZ = cfg.minZ
        self.maxZ = cfg.maxZ

    def global_boundary_check(self, points, labels, calibs):
        mask = np.where((points[:, 0] >= self.minX) & (points[:, 0] < self.maxX) &
                        (points[:, 1] >= self.minY) & (points[:, 1] < self.maxY) &
                        (points[:, 2] >= self.minZ) & (points[:, 2] < self.maxZ))
        PointCloud = points[mask]
        #PointCloud[:, 2] = PointCloud[:, 2] - self.minZ

        Labels = []
        for label, calib in zip(labels, calibs):
            ls = []
            if label is not None:
                pass
            else:
                Labels.append(None)
            for o in label:
                x, y, z = Object2Velo.camera_to_lidar(o.t[0], o.t[1], o.t[2], calib.V2C, calib.R0, calib.P)
                if ((x > self.minX) & (x < self.maxX) &
                (y > self.minY) & (y < self.maxX) &
                (z > self.minZ) & (z < self.maxZ)):
                    ls.append(o)
                continue
            if len(ls) == 0:
                Labels.append(None)
            Labels.append(o)
            continue
        return PointCloud, Labels

    def global_shift(self, points, labels, rx, ry):
        cnt = 0
        PointCloud, Labels = [], []
        for point, label in zip(points, labels):
            if cnt == 0:
                point_ = point[:, 씨발 label이거 camera좌표임 ㅋㅋㅋㅋㅋㅋㅋ]
            cnt = cnt + 1
            continue
        return PointCloud, Labels

    def get_vaild_cehck_bev(self, corners3dvelo, rx, ry):
        info = []
        cnt = 0
        for corners3d in corners3dvelo:
            if corners3d is not None:
                pass
            else:
                info.append(None)
            for corners in corners3d:
                x1, y1 = corners[0, 1], corners[0, 2]

    def get_corners3d_velo(self, labels, calibs):
        corners3dcam = []
        for label, calib in zip(labels, calibs):
            cs = []
            if label is not None:
                pass
            else:
                corners3dcam.append(None)
                continue
            for l in label:
                _, corners3d_ = l.compute_box_3d(calib.P)
                cs.append(corners3d_)
                continue
            corners3dcam.append(cs)
            continue

        corners3dvelo = []
        for corners3d, calib in zip(corners3dcam, calibs):
            cs = []
            if corners3d is not None:
                pass
            else:
                corners3dvelo.append(None)
                continue
            for corners in corners3d:
                corners_ = Object2Velo.camera_to_lidar_corner(corners, calib)
                cs.append(corners_)
                continue
            corners3dvelo.append(cs)
            continue
        return corners3dvelo

    def __call__(self, points, labels, calibs):
        rx = np.random.uniform(0.2, 0.8) * self.voxelshape[2] * self.voxelsize[2]
        ry = np.random.uniform(0.2, 0.8) * self.voxelshape[1] * self.voxelsize[1]

        corners3dvelo = self.get_corners3d_velo(labels, calibs)
        self.get_vaild_cehck_bev(corners3dvelo, rx, ry)
        print(corners3dvelo)
        print(rx,ry)
        df=df

        
