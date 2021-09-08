
import torch.utils.data as data
from .utils import *

import os
import numpy as np

_BASE_DIR = './samples/'

class KittiDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        with open(_BASE_DIR + 'train.txt') as f:
            self.lists = f.readlines()
        self.point2voxel = Point2Voxel(cfg)
        self.randommosaic = RandomMosaic(cfg)
    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):

        if self.mode == 'val':
            return 0
        
        Plist, Llist, Clist = [], [], []
        for i in range(4):
            id = random.randint(0, self.__len__() - 1)
            points = self.read_velo(_BASE_DIR + 'velodyne(lidar)/' + self.lists[id][:6] + '.bin')
            calib  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[id][:6] + '.txt')
            labels = self.read_label(_BASE_DIR + 'label/' + self.lists[id][:6] + '.txt', points, calib)
            Plist.append(points)
            Llist.append(labels)
            Clist.append(calib)
            # x, y, z, w, l, h, ry
            #bbox_, cls_, dif_ = Object2Velo.camera_to_lidar_box(labels, calib)
            #points = self.pointroi(points)
            #voxels, coors, num_points_per_voxel = self.point2voxel(points)
        
        kew = self.randommosaic(Plist, Llist, Clist)
        return 0

    def read_velo(self, filepath):
        points = KittiScene(filepath)
        return points

    def read_calib(self, filename):
        return Calibration(filename)

    def read_label(self, filename, points, calib):
        lines = [line.rstrip() for line in open(filename)]
        objects = [KittiObject(line, points.points, calib) for line in lines]
        olist = []
        for o in objects:
            if o.cls_id == -1:
                continue
            elif o.level == 4:
                continue
            else:
                olist.append(o)
        if len(olist) > 0:
            return olist
        else:
            return None
