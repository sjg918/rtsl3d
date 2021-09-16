
import torch.utils.data as data
import numba
import random
from torch.nn import functional as F
import math

import os
from PIL import Image
import numpy as np

# refers
# https://github.com/ghimiredhikura/Complex-YOLOv3
# https://github.com/Vegeta2020/SE-SSD
# https://github.com/jeasinema/VoxelNet-tensorflow
# https://github.com/kuangliu/kitti-utils 

class KittiScene(object):
    def __init__(self, filepath):
        self.points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

    def shift3d(self, x, y, z, kittiobjects=None):
        if kittiobjects is not None:
            for object in kittiobjects:
                object.shift3d(x, y, z)
        self.points[:, 0] = self.points[:, 0] + x
        self.points[:, 1] = self.points[:, 1] + y
        self.points[:, 2] = self.points[:, 2] + z

    def flip3d(self, kittiobjects=None):
        if kittiobjects is not None:
            for object in kittiobjects:
                object.flip3d(scene=True)
        self.points[:, 1] = - self.points[:, 1]

    def remove_object_innerpoints(self, kittiobjects):
        if kittiobjects is None:
            pass
        else:
            for object in kittiobjects:
                boundary = object.compute_boundary3d()
                self.points = KittiObjectUtils.remove_inner_points(boundary, self.points)
                continue

    def add_object_innerpoints(self, kittiobjects):
        if kittiobjects is None:
            pass
        else:
            for object in kittiobjects:
                innerpoints = object.innerpoints
                self.points = np.concatenate([self.points, innerpoints], axis=0)
                continue
    
    def merge_dif_scene(self, *args):
        for i in args:
            self.points = np.concatenate([self.points, i.points])

class KittiObject(object):
    def __init__(self, label_file_line, points, calib):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        
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
        self.camx = data[11]
        self.camy = data[12]
        self.camz = data[13]
        #self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = data[15] if data.__len__() == 16 else -1.0
        self.level_str, self.level = self.get_obj_level()

        (x, y, z) = KittiObjectUtils.camera_to_lidar(\
            self.camx, self.camy, self.camz, calib.V2C, calib.R0, calib.P)
        self.velox = x
        self.veloy = y
        self.veloz = z
        self.camcorners2d, self.camcorners3d = self.compute_box_3d(calib.P)
        self.velocorners3d = self.camera_corners_to_lidar(self.camcorners3d, calib.V2C, calib.R0, calib.P)
        self.innerpoints = KittiObjectUtils.compute_boundary_inner_points(self.compute_boundary3d(), points)

    def compute_boundary3d(self):
        minX, maxX = self.velocorners3d[:, 0].min(), self.velocorners3d[:, 0].max()
        minY, maxY = self.velocorners3d[:, 1].min(), self.velocorners3d[:, 1].max()
        minZ, maxZ = self.velocorners3d[:, 2].min(), self.velocorners3d[:, 2].max()
        return np.array([minX, maxX, minY, maxY, minZ, maxZ], dtype=np.float32)

    def shift3d(self, x, y, z):
        self.velox = self.velox + x
        self.veloy = self.veloy + y
        self.veloz = self.veloz + z
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] + x
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] + y
        self.velocorners3d[:, 2] = self.velocorners3d[:, 2] + z
        self.innerpoints[:, 0] = self.innerpoints[:, 0] + x
        self.innerpoints[:, 1] = self.innerpoints[:, 1] + y
        self.innerpoints[:, 2] = self.innerpoints[:, 2] + z
        
    def scale3d(self, scalex, scaley, scalez):
        self.h = self.h * scalex
        self.w = self.w * scaley
        self.l = self.l * scalez
        
        self.innerpoints[:, 0] = self.innerpoints[:, 0] - self.velox
        self.innerpoints[:, 1] = self.innerpoints[:, 1] - self.veloy
        self.innerpoints[:, 2] = self.innerpoints[:, 2] - self.veloz
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] - self.velox
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] - self.veloy
        self.velocorners3d[:, 2] = self.velocorners3d[:, 2] - self.veloz
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] * scalex
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] * scaley
        self.velocorners3d[:, 2] = self.velocorners3d[:, 2] * scalez
        self.innerpoints[:, 0] = self.innerpoints[:, 0] * scalex
        self.innerpoints[:, 1] = self.innerpoints[:, 1] * scaley
        self.innerpoints[:, 2] = self.innerpoints[:, 2] * scalez
        self.innerpoints[:, 0] = self.innerpoints[:, 0] + self.velox
        self.innerpoints[:, 1] = self.innerpoints[:, 1] + self.veloy
        self.innerpoints[:, 2] = self.innerpoints[:, 2] + self.veloz
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] + self.velox
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] + self.veloy
        self.velocorners3d[:, 2] = self.velocorners3d[:, 2] + self.veloz

    def flip3d(self, scene=False):
        if scene == True:
            self.veloy = - self.veloy
            self.ry = - self.ry + np.pi
            self.velocorners3d[:, 1] = - self.velocorners3d[:, 1]
            self.innerpoints[:, 1] = - self.innerpoints[:, 1]
        else:
            self.innerpoints[:, 0] = self.innerpoints[:, 0] - self.velox
            self.innerpoints[:, 1] = self.innerpoints[:, 1] - self.veloy
            self.velocorners3d[:, 0] = self.velocorners3d[:, 0] - self.velox
            self.velocorners3d[:, 1] = self.velocorners3d[:, 1] - self.veloy
            self.innerpoints[:, :3] = self.innerpoints[:, :3] @ KittiObjectUtils.rotz(-self.ry)
            self.innerpoints[:, 1] = - self.innerpoints[:, 1]
            self.innerpoints[:, :3] = self.innerpoints[:, :3] @ KittiObjectUtils.rotz(self.ry)
            self.velocorners3d[:, :3] = self.velocorners3d[:, :3] @ KittiObjectUtils.rotz(-self.ry)
            self.velocorners3d[:, 1] = - self.velocorners3d[:, 1]
            self.velocorners3d[:, :3] = self.velocorners3d[:, :3] @ KittiObjectUtils.rotz(self.ry)
            self.innerpoints[:, 0] = self.innerpoints[:, 0] + self.velox
            self.innerpoints[:, 1] = self.innerpoints[:, 1] + self.veloy
            self.velocorners3d[:, 0] = self.velocorners3d[:, 0] + self.velox
            self.velocorners3d[:, 1] = self.velocorners3d[:, 1] + self.veloy
            KittiObjectUtils.swap2flip(self.velocorners3d, 0, 1)
            KittiObjectUtils.swap2flip(self.velocorners3d, 2, 3)
            KittiObjectUtils.swap2flip(self.velocorners3d, 4, 5)
            KittiObjectUtils.swap2flip(self.velocorners3d, 6, 7)
            self.ry = self.ry + np.pi

    def rotate3d(self, angle):
        self.innerpoints[:, 0] = self.innerpoints[:, 0] - self.velox
        self.innerpoints[:, 1] = self.innerpoints[:, 1] - self.veloy
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] - self.velox
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] - self.veloy
        self.innerpoints[:, :3] = self.innerpoints[:, :3] @ KittiObjectUtils.rotz(angle)
        self.velocorners3d[:, :3] = self.velocorners3d[:, :3] @ KittiObjectUtils.rotz(angle)
        self.innerpoints[:, 0] = self.innerpoints[:, 0] + self.velox
        self.innerpoints[:, 1] = self.innerpoints[:, 1] + self.veloy
        self.velocorners3d[:, 0] = self.velocorners3d[:, 0] + self.velox
        self.velocorners3d[:, 1] = self.velocorners3d[:, 1] + self.veloy
        self.ry = self.ry - angle

    def get_numpy_kittiformat(self):
        return np.array([self.velox, self.veloy, self.veloz, self.w, self.l, self.h, self.ry], dtype=np.float32)

    def compute_box_3d(self, P):
        # compute rotational matrix around yaw axis
        R = KittiObjectUtils.roty(self.ry)

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
        corners_3d[0, :] = corners_3d[0, :] + self.camx
        corners_3d[1, :] = corners_3d[1, :] + self.camy
        corners_3d[2, :] = corners_3d[2, :] + self.camz
        # print 'cornsers_3d: ', corners_3d
        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane
        corners_2d = KittiObjectUtils.project_to_image(np.transpose(corners_3d), P)
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)

    def camera_corners_to_lidar(self, camcorners3d, V2C, R0, P):
        velocorners3d = []
        for corners in camcorners3d:
            x, y, z = corners
            (x, y, z) = KittiObjectUtils.camera_to_lidar(\
                x, y, z, V2C, R0, P)
            velocorners3d.append(np.array([x,y,z], dtype=np.float32))
        return np.stack(velocorners3d, axis=0)

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
            return 'Easy', 1
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            return 'Moderate', 2
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            return 'Hard', 3
        else:
            return 'UnKnown', 4

class KittiObjectUtils(object):
    def __init__(self):
        pass
    
    @staticmethod
    @numba.jit(nopython=True)
    def swap2flip(corners, idx1, idx2):
        tmp = corners[idx1].copy()
        corners[idx1] = corners[idx2]
        corners[idx2] = tmp

    @staticmethod
    @numba.jit(nopython=True)
    def remove_inner_points(boundary, points):
        minX, maxX, minY, maxY, minZ, maxZ = boundary
        mask = np.where((points[:, 0] < minX) | (points[:, 0] > maxX) |
                        (points[:, 1] < minY) | (points[:, 1] > maxY) |
                        (points[:, 2] < minZ) | (points[:, 2] > maxZ))
        outerpoints = points[mask]
        return outerpoints

    @staticmethod
    @numba.jit(nopython=True)
    def compute_boundary_inner_points(boundary, points):
        minX, maxX, minY, maxY, minZ, maxZ = boundary
        mask = np.where((points[:, 0] >= minX) & (points[:, 0] < maxX) &
                        (points[:, 1] >= minY) & (points[:, 1] < maxY) &
                        (points[:, 2] >= minZ) & (points[:, 2] < maxZ))
        innerpoints = points[mask]
        return innerpoints

    @staticmethod
    def verify_object_inner_boundary(boundary, kittiobjects):
        minX, maxX, minY, maxY, minZ, maxZ = boundary
        if kittiobjects is None:
            return None
        vl = []
        for o in kittiobjects:
            if ((o.velox > minX) & (o.velox < maxX) &
                (o.veloy > minY) & (o.veloy < maxY)):
                vl.append(o)
            continue

        if len(vl) == 0:
            return None
        else:
            return vl

    @staticmethod
    def cehck_overlap_object_and_merge(kittiobjectlist):
        vkitti = []
        stat = [True, True, True, True]
        for i in range(len(kittiobjectlist)):
            fnum = 0
            if kittiobjectlist[i] is None:
                stat[i] = False
                continue
            for object in kittiobjectlist[i]:
                x, y = object.velox, object.veloy
                flag = False
                for j in range(i+1, len(kittiobjectlist)):
                    if kittiobjectlist[j] is None:
                        break

                    if flag == True:
                        break
                    for object_ in kittiobjectlist[j]:
                        if flag == True:
                            break
                        x_, y_ = object_.velox, object_.veloy
                        if (abs(x - x_) < 1.5) & (abs(y - y_) < 1.5):
                            continue
                        else:
                            flag = True
                        continue
                    continue
                if flag == True:
                    fnum = fnum + 1
                    continue
                vkitti.append(object)
                continue
            if fnum == len(kittiobjectlist[i]):
                stat[i] = False
            continue
        if len(vkitti) == 0:
            vkitti = None
        return vkitti, stat

    @staticmethod
    def project_to_image(pts_3d, P):
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

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
            p = np.matmul(KittiObjectUtils.inverse_rigid_trans(V2C), p)
        p = p[0:3]
        return tuple(p)

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


class Calibration(object):
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = KittiObjectUtils.inverse_rigid_trans(self.V2C)
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

class VoxelGenerateUtils(object):
    # refer
    # https://github.com/Vegeta2020/SE-SSD
    def __init__(self, cfg):
        self.voxelsize = np.array(cfg.voxelsize)
        self.voxelrange = np.array(cfg.voxelrange)
        self.voxelshape = cfg.voxelshape
        self.p2v_maxpoints = cfg.p2v_maxpoints
        self.p2v_maxvoxels = cfg.p2v_maxvoxels
        self.p2v_idxmat = 65535 * np.ones((self.voxelshape), dtype=np.uint16)

        self.bevsize = cfg.bevsize
        self.bevmulti = cfg.bevmulti
        self.bevshape = cfg.bevshape
        self.v2b_maxvoxels = cfg.v2b_maxvoxels
        self.v2b_idxmat = 65535 * np.ones((self.bevshape), dtype=np.uint16)
        self.voxelwise_channels = cfg.back_voxelwise_channels

    @staticmethod
    @numba.jit(nopython=True)
    def point2voxel_kernel(
        points, voxel_size, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels):
        N = points.shape[0]

        ndim = 3
        idxtup = (2, 1, 0)
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)
        coor = np.zeros(shape=(3,), dtype=np.int64)
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

    def point2voxel(self, points):
        voxelmap_shape = (self.voxelrange[3:] - self.voxelrange[:3]) / self.voxelsize
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.long).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        num_points_per_voxel = np.zeros(shape=(self.p2v_maxvoxels,), dtype=np.long)
        voxels = np.zeros(shape=(self.p2v_maxvoxels, self.p2v_maxpoints, points.shape[-1]), dtype=np.float32)
        coors = np.zeros(shape=(self.p2v_maxvoxels, 3), dtype=np.long)
        voxel_num = VoxelGenerateUtils.point2voxel_kernel(
            points, self.voxelsize, self.voxelrange, num_points_per_voxel,
            self.p2v_idxmat, voxels, coors, self.p2v_maxpoints, self.p2v_maxvoxels,)
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]
        self.p2v_idxmat = self.p2v_idxmat * 0 + 65535
        return voxels, coors, num_points_per_voxel

    @staticmethod
    @numba.jit(nopython=True)
    def paddingmask_kernel(num_voxel, max_num):
        actual_num = np.expand_dims(num_voxel, axis=1)
        max_num = np.arange(max_num, dtype=np.int64).reshape(1, max_num)
        paddings_indicator = actual_num.astype(np.int64) > max_num
        return paddings_indicator

    @staticmethod
    @numba.jit(nopython=True)
    def voxel2bev_kernel(
        N, coors, bevsidx, bevcoors, num_voxels_per_bev,
        v2b_idxmat, bevmulti):

        ndim = 2
        ndim_minus_1 = ndim - 1
        coorx = 0
        coory = 0
        #coor = np.zeros((2), dtype=np.int64)
        bev_num = 0

        for i in range(N):
            coorx = coors[i, 1] // bevmulti[0]
            coory = coors[i, 2] // bevmulti[1]
            bevidx = v2b_idxmat[coorx, coory]
            if bevidx == 65535:
                bevidx = bev_num
                bev_num = bev_num + 1
                v2b_idxmat[coorx, coory] = bevidx
                bevcoors[bevidx, 0] = coorx
                bevcoors[bevidx, 1] = coory
            num = num_voxels_per_bev[bevidx]
            bevsidx[bevidx, num] = i
            num_voxels_per_bev[bevidx] += 1
            continue
        return bev_num

    def voxel2bev(self, N, coors):
        bevsidx = np.zeros((self.bevshape[0]*self.bevshape[1], self.v2b_maxvoxels), dtype=np.long)
        bevcoors = np.zeros((self.bevshape[0]*self.bevshape[1], 2), dtype=np.long)
        num_voxels_per_bev = np.zeros((self.bevshape[0]*self.bevshape[1]), dtype=np.long)
        bevmulti = np.array(self.bevmulti, dtype=np.long)
        bev_num = VoxelGenerateUtils.voxel2bev_kernel(
            N, coors, bevsidx, bevcoors, num_voxels_per_bev, self.v2b_idxmat, bevmulti)
        bevsidx = bevsidx[:bev_num]
        bevcoors = bevcoors[:bev_num]
        num_voxels_per_bev = num_voxels_per_bev[:bev_num]
        self.v2b_idxmat = self.v2b_idxmat * 0 + 65535
        return bevsidx, bevcoors, num_voxels_per_bev

class RandomMosaic(object):
    def __init__(self, cfg, probability=0.5):
        self.voxelshape = cfg.voxelshape
        self.voxelsize = cfg.voxelsize
        self.minX = cfg.minX
        self.maxX = cfg.maxX
        self.minY = cfg.minY
        self.maxY = cfg.maxY
        self.minZ = cfg.minZ
        self.maxZ = cfg.maxZ
        self.boundary = np.array([self.minX, self.maxX,
                                  self.minY, self.maxY, self.minZ, self.maxZ], dtype=np.float32)
        self.p = probability

    def __call__(self, points, labels):
        """
        input
        points-> [KittiScene, ...]
        labels-> [[KittiObject, ...], ...]

        output
        outpoints-> KittiScene
        labels-> [KittiObject, ...]
        info-> [6]
        0,1,2,3-> Indicates whether an object exists or does not exist in the area below. 
        4,5-> rx, ry
        ----------------
        | 3     | 2
        |       |
        |       |
        -------rxry-----
        | 1     | 0
        |       |
        |       |
        ----------------
        """

        # set rx,ry
        rx = np.random.uniform(0.2, 0.8) * self.voxelshape[2] * self.voxelsize[2]
        ry = np.random.uniform(0.2, 0.8) * self.voxelshape[1] * self.voxelsize[1]

        # remove outer points
        for scene in points:
            scene.points = KittiObjectUtils.compute_boundary_inner_points(self.boundary, scene.points)
            continue
  
        # remove object points
        for scene, object in zip(points, labels):
            scene.remove_object_innerpoints(object)
            continue

        # random horizontal flip
        for scene, object in zip(points, labels):
            if np.random.uniform(0, 1) < self.p:
                scene.flip3d(object)
            continue

        # random object flip and rotate
        for object in labels:
            if (object is not None):
                for o in object:
                    if (np.random.uniform(0, 1) < self.p):
                        o.flip3d(scene=False)
                    if (np.random.uniform(0, 1) < self.p):
                        angle = np.random.uniform(-np.pi /4 , np.pi /4)
                        o.rotate3d(angle)
                    continue
            continue

        # shift y-axis
        for scene, object in zip(points, labels):
            scene.shift3d(0, -self.minY, 0, object)
            continue

        # shift scene & object
        if np.random.uniform(0, 1) < self.p:
            cnt = 0
            for scene, object in zip(points, labels):
                if cnt == 0:
                    scene.shift3d(-rx, +ry, 0, object)
                elif cnt == 1:
                    scene.shift3d(-rx, -ry, 0, object)
                elif cnt == 2:
                    scene.shift3d(+rx, +ry, 0, object)
                elif cnt == 3:
                    scene.shift3d(+rx, -ry, 0, object)
                cnt = cnt + 1
                continue
        else:
            pass
        
        # remove outer points & object
        cnt = 0
        vlabels = []
        for scene, object in zip(points, labels):
            if cnt == 0:
                boundary_ = [self.minX, rx, ry, self.maxY-self.minY, self.minZ, self.maxZ]
            elif cnt == 1:
                boundary_ = [self.minX, rx, 0, ry, self.minZ, self.maxZ]
            elif cnt == 2:
                boundary_ = [rx, self.maxX, ry, self.maxY-self.minY, self.minZ, self.maxZ]
            elif cnt == 3:
                boundary_ = [rx, self.maxX, 0, ry, self.minZ, self.maxZ]
            boundary_ = np.array(boundary_, dtype=np.float32)

            scene.points = KittiObjectUtils.compute_boundary_inner_points(boundary_, scene.points)
            vlabels.append(KittiObjectUtils.verify_object_inner_boundary(boundary_, object))
            cnt = cnt + 1
            continue

        # check overlap object and merge
        vkitti, stat = KittiObjectUtils.cehck_overlap_object_and_merge(vlabels)
        
        # shift y-axis
        for scene in points:
            scene.shift3d(0, self.minY, 0)
            continue
        if vkitti is not None:
            for object in vkitti:
                object.shift3d(0, self.minY, 0)
                continue

        # merge points
        points[0].merge_dif_scene(points[1], points[2], points[3])
        points[0].add_object_innerpoints(vkitti)
        stat.append(rx)
        stat.append(ry)
        return points[0], vkitti, stat

        
class RandomMixup(object):
    def __init__(self, cfg):
        self.minX = cfg.minX
        self.maxX = cfg.maxX
        self.minY = cfg.minY
        self.maxY = cfg.maxY
        self.minZ = cfg.minZ
        self.maxZ = cfg.maxZ

    def __call__(self, points, labels, labels_, ra, rx, ry):
        if ra == 0:
            minX, maxX, minY, maxY = 0, rx, ry+self.minY, self.maxY
        elif ra == 1:
            minX, maxX, minY, maxY = 0, rx, self.minY, ry+self.minY
        elif ra == 2:
            minX, maxX, minY, maxY = rx, self.maxX, ry+self.minY, self.maxY
        elif ra == 3:
            minX, maxX, minY, maxY = rx, self.maxX, self.minY, ry+self.minY
        
        cx = minX + np.random.uniform(0.2, 0.8) * (maxX - minX)
        cy = minY + np.random.uniform(0.2, 0.8) * (maxY - minY)
        jumpguy = random.choice(labels_)
        shiftx, shifty = cx - jumpguy.velox, cy - jumpguy.veloy 
        jumpguy.shift3d(shiftx, shifty, 0)
        points.remove_object_innerpoints([jumpguy])
        points.add_object_innerpoints([jumpguy])
        if labels is not None:
            labels.append(jumpguy)
            return points, labels
        return points, [jumpguy]


class PyramidUtils(object):
    def __init__(self):
        pass

    @staticmethod
    @numba.jit(nopython=True)
    def compute_8plane_through_9corners(corners3d, cornersequence):
        """
        corners3d has 9 points. following order:
        1 -------- 0
        /|         /|
        2 -------- 3 .
        | |        | |    center -> 8
        . 5 -------- 4
        |/         |/
        6 -------- 7
        
        cornersequence. following order:
        0-> 801 / 1-> 812 / 2-> 823 / 3-> 830
        4-> 804 / 5-> 815 / 6-> 826 / 7-> 837
        """

        planes = np.zeros((8, 4), dtype=np.float32)
        for i in range(8):
            sequence = cornersequence[i]
            p1, p2, p3 = sequence
            point1, point2, point3 = corners3d[p1], corners3d[p2], corners3d[p3]
            v1 = point3 - point1
            v2 = point2 - point1
            cp = np.cross(v1, v2)
            a, b, c = cp
            d = np.dot(cp, point3)
            planes[i, :] = a, b, c, d
            continue
        return planes

    @staticmethod
    @numba.jit(nopython=True)
    def divid_object_to_6pyramids(points, planes, planesequence):
        numpoints = points.shape[0]
        topmask = np.zeros((numpoints), dtype=np.bool_)
        bottommask = np.zeros((numpoints), dtype=np.bool_)
        leftmask = np.zeros((numpoints), dtype=np.bool_)
        rightmask = np.zeros((numpoints), dtype=np.bool_)
        frontmask = np.zeros((numpoints), dtype=np.bool_)
        rearmask = np.zeros((numpoints), dtype=np.bool_)
        #topmask, bottommask, leftmask, rightmask, frontmask, rearmask = masks
        for i in range(6):
            sequence = planesequence[i]
            p1, p2, p3, p4 = sequence
            plane1, plane2, plane3, plane4 = planes[p1], planes[p2], planes[p3], planes[p4]
            a1, b1, c1, d1 = plane1
            a2, b2, c2, d2 = plane2
            a3, b3, c3, d3 = plane3
            a4, b4, c4, d4 = plane4
            
            if i == 0: # top
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 > 0) &
                        (a2 * x + b2 * y + c2 * z - d2 > 0) &
                        (a3 * x + b3 * y + c3 * z - d3 > 0) &
                        (a4 * x + b4 * y + c4 * z - d4 > 0)):
                        topmask[pn] = True
                    else:
                        continue
                    continue
            elif i == 1: # bottom
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 < 0) &
                        (a2 * x + b2 * y + c2 * z - d2 < 0) &
                        (a3 * x + b3 * y + c3 * z - d3 < 0) &
                        (a4 * x + b4 * y + c4 * z - d4 < 0)):
                        bottommask[pn] = True
                    else:
                        continue
                    continue
            elif i == 2: # right
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 >= 0) &
                        (a2 * x + b2 * y + c2 * z - d2 <= 0) &
                        (a3 * x + b3 * y + c3 * z - d3 < 0) & 
                        (a4 * x + b4 * y + c4 * z - d4 >= 0)):
                        rightmask[pn] = True
                    else:
                        continue
                    continue
            elif i == 3: # left
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 <= 0) &
                        (a2 * x + b2 * y + c2 * z - d2 >= 0) &
                        (a3 * x + b3 * y + c3 * z - d3 >= 0) & 
                        (a4 * x + b4 * y + c4 * z - d4 < 0)):
                        leftmask[pn] = True
                    else:
                        continue
                    continue
            elif i == 4: # rear
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 <= 0) &
                        (a2 * x + b2 * y + c2 * z - d2 >= 0) &
                        (a3 * x + b3 * y + c3 * z - d3 >= 0) & 
                        (a4 * x + b4 * y + c4 * z - d4 < 0)):
                        rearmask[pn] = True
                    else:
                        continue
                    continue
            elif i == 5: # front
                for pn in range(numpoints):
                    x, y, z, _ = points[pn]
                    if ((a1 * x + b1 * y + c1 * z - d1 >= 0) &
                        (a2 * x + b2 * y + c2 * z - d2 <= 0) &
                        (a3 * x + b3 * y + c3 * z - d3 >= 0) & 
                        (a4 * x + b4 * y + c4 * z - d4 < 0)):
                        frontmask[pn] = True
                    else:
                        continue
                    continue
            continue
        top = points[topmask]
        bottom = points[bottommask]
        left = points[leftmask]
        right = points[rightmask]
        front = points[frontmask]
        rear = points[rearmask]
        return top, bottom, left, right, front, rear

class RandomPyramid(object):
    def __init__(self, cfg, dropprob=0.5, sparsifyprob=0.5):
        self.dropp = dropprob
        self.sparsifyp = sparsifyprob
        self.cornersequence = np.array([[8,0,1], [8,1,2], [8,2,3], [8,3,0], 
                                       [8,0,4], [8,1,5], [8,2,6], [8,3,7]], dtype=np.int16)
        self.planesequence = np.array([[0,1,2,3], [0,1,2,3], [1,3,4,7],
                        [1,3,5,6], [0,2,4,5], [0,2,6,7]], dtype=np.int16)

    def __call__(self, points, labels):
        if labels is None or len(labels) == 1:
            return points, labels
        numobject = len(labels)

        # gen 6pyramid numpy
        pyramidlist = []
        for object in labels:
            x, y, z = object.velox, object.veloy, object.veloz
            z = z + object.h / 2
            center = np.array([x, y, z]).reshape(1, 3)
            corners = np.concatenate([object.velocorners3d, center], axis=0)
            planes = PyramidUtils.compute_8plane_through_9corners(corners, self.cornersequence)
            t,b,L,R,f,r = PyramidUtils.divid_object_to_6pyramids(object.innerpoints, planes, self.planesequence)
            pyramidlist.append([t,b,L,R,f,r])
            continue

        # random swap
        if numobject < 2:
            pass
        else:
        #for num in range(numobject):
            num = np.random.randint(0, numobject)
            idx = np.random.randint(0, 6)
            choicelist = [i for i in range(numobject)]
            choicelist.remove(num)
            choice = random.choice(choicelist)
            tmp = pyramidlist[num][idx]
            pyramidlist[num][idx] = pyramidlist[choice][idx]
            pyramidlist[choice][idx] = tmp

            x1, y1, z1, ry1 = labels[num].velox, labels[num].veloy, labels[num].veloz, labels[num].ry
            l1, w1, h1 = labels[num].l, labels[num].w, labels[num].h
            x2, y2, z2, ry2 = labels[choice].velox, labels[choice].veloy, labels[choice].veloz, labels[choice].ry
            l2, w2, h2 = labels[choice].l, labels[choice].w, labels[choice].h
            dx, dy, dz, dry = x2 - x1, y2 - y1, z2 - z1, ry2 - ry1
            dl, dw, dh = l2 / l1, w2 / w1, h2 / h1
            if pyramidlist[num][idx].shape[0] != 0:
                pyramidlist[num][idx][:, 0] = pyramidlist[num][idx][:, 0] - dx
                pyramidlist[num][idx][:, 1] = pyramidlist[num][idx][:, 1] - dy
                pyramidlist[num][idx][:, 2] = pyramidlist[num][idx][:, 2] - dz

                pyramidlist[num][idx][:, 0] = pyramidlist[num][idx][:, 0] - x1
                pyramidlist[num][idx][:, 1] = pyramidlist[num][idx][:, 1] - y1
                pyramidlist[num][idx][:, 0] = pyramidlist[num][idx][:, 0] / dw
                pyramidlist[num][idx][:, 1] = pyramidlist[num][idx][:, 1] / dl
                pyramidlist[num][idx][:, 1] = pyramidlist[num][idx][:, 1] / dh
                pyramidlist[num][idx][:, :3] = pyramidlist[num][idx][:, :3] @ KittiObjectUtils.rotz(dry)
                pyramidlist[num][idx][:, 0] = pyramidlist[num][idx][:, 0] + x1
                pyramidlist[num][idx][:, 1] = pyramidlist[num][idx][:, 1] + y1
            if pyramidlist[choice][idx].shape[0] != 0:
                pyramidlist[choice][idx][:, 0] = pyramidlist[choice][idx][:, 0] + dx
                pyramidlist[choice][idx][:, 1] = pyramidlist[choice][idx][:, 1] + dy
                pyramidlist[choice][idx][:, 2] = pyramidlist[choice][idx][:, 2] + dz

                pyramidlist[choice][idx][:, 0] = pyramidlist[choice][idx][:, 0] - x2
                pyramidlist[choice][idx][:, 1] = pyramidlist[choice][idx][:, 1] - y2
                pyramidlist[choice][idx][:, 0] = pyramidlist[choice][idx][:, 0] * dw
                pyramidlist[choice][idx][:, 1] = pyramidlist[choice][idx][:, 1] * dl
                pyramidlist[choice][idx][:, 2] = pyramidlist[choice][idx][:, 2] * dh
                pyramidlist[choice][idx][:, :3] = pyramidlist[choice][idx][:, :3] @ KittiObjectUtils.rotz(-dry)
                pyramidlist[choice][idx][:, 0] = pyramidlist[choice][idx][:, 0] + x2
                pyramidlist[choice][idx][:, 1] = pyramidlist[choice][idx][:, 1] + y2

        ## random dropout
        #for num in range(numobject):
        #    if labels[num].level == 3:
        #        continue

        #    if np.random.uniform(0, 1) < self.dropp:
        #        idx = np.random.randint(0, 6)
        #        empty = np.zeros((0, 4), dtype=np.float32)
        #        pyramidlist[num][idx] = empty
        #        continue
        #    continue

        ## random sparsify
        #for num in range(numobject):
        #    if labels[num].level == 3:
        #        continue

        #    if np.random.uniform(0, 1) < self.sparsifyp:
        #        idx = np.random.randint(0, 6)
        #        tmp = pyramidlist[num][idx]
        #        if tmp.shape[0] == 0:
        #            continue
        #        mask = np.random.uniform(0, 1, size=tmp.shape[0])
        #        mask = (mask > 0.5)
        #        pyramidlist[num][idx] = tmp[mask]
        #    continue

        # merge
        for num, (t,b,L,R,f,r) in enumerate(pyramidlist):
            labels[num].innerpoints = np.concatenate([t,b,L,R,f,r], axis=0)
            continue

        # modify scene
        points.remove_object_innerpoints(labels)
        points.add_object_innerpoints(labels)
        return points, labels
