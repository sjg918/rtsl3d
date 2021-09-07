
import torch.utils.data as data
from .utils import *

import os
import numpy as np

_BASE_DIR = 'C:\Datasets/sample/'

class KittiDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        with open(_BASE_DIR + 'train.txt') as f:
            self.lists = f.readlines()
        self.pointroi = PointROI(cfg)
        self.point2voxel = Point2Voxel(cfg)

    def __len__(self):
        return len(self.lists)

    def read_velo(self, filename):
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return points

    def read_label(self, filename):
        lines = [line.rstrip() for line in open(filename)]
        objects = [Object2Cam(line) for line in lines]
        return objects

    def read_calib(self, filename):
        return Calibration(filename)

    def __getitem__(self, id):

        points = self.read_velo(_BASE_DIR + 'velodyne(lidar16ch)/' + self.lists[id][:6] + '.bin')
        labels = self.read_label(_BASE_DIR + 'label/' + self.lists[id][:6] + '.txt')
        calib  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[id][:6] + '.txt')

        # x, y, z, w, l, h, ry
        bbox_, cls_, dif_ = Object2Velo.camera_to_lidar_box(labels, calib)

        points = self.pointroi(points)
        voxels, coors, num_points_per_voxel = self.point2voxel(points)
        return voxels, coors, num_points_per_voxel