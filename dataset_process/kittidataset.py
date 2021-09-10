
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
        self.point2voxel = Point2Voxel(cfg)
        self.randommosaic = RandomMosaic(cfg)
        self.randommixup = RandomMixup(cfg)
        self.randompyramid = RandomPyramid(cfg)
        self.boundary = np.array([cfg.minX, cfg.maxX,
                                  cfg.minY, cfg.maxY, cfg.minZ, cfg.maxZ], dtype=np.float32)
    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):

        if self.mode == 'val':
            return 0
        
        Plist, Llist = [], []
        for i in range(4):
            id = random.randint(0, self.__len__() - 1)
            points = self.read_velo(_BASE_DIR + 'velodyne(lidar)/' + self.lists[id][:6] + '.bin')
            calib  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[id][:6] + '.txt')
            labels = self.read_label(_BASE_DIR + 'label/' + self.lists[id][:6] + '.txt', points, calib)
            Plist.append(points)
            Llist.append(labels)
        
        points, labels, stat = self.randommosaic(Plist, Llist)
        ra, rx, ry = self.random_select_area(stat[:4]), stat[4], stat[5]
        if ra is not None:
            while True:
                id = random.randint(0, self.__len__() - 1)
                points_ = self.read_velo(_BASE_DIR + 'velodyne(lidar)/' + self.lists[id][:6] + '.bin')
                calib_  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[id][:6] + '.txt')
                labels_ = self.read_label(_BASE_DIR + 'label/' + self.lists[id][:6] + '.txt', points, calib)
                labels_ = KittiObjectUtils.verify_object_inner_boundary(self.boundary, labels_)
                if labels_ is not None:
                    break
                continue
            points, labels = self.randommixup(points, labels, labels_, ra, rx, ry)
        poitns, labels = self.randompyramid(points, labels)
        #print(points.points.shape)
        #df=df
        #points = self.pointroi(points)
        #voxels, coors, num_points_per_voxel = self.point2voxel(points)
        print('kewkewkew')
        df=df
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

    def random_select_area(self, stat):
        if False not in stat:
            return None
        
        idx = []
        for n, i in enumerate(stat):
            if i == False:
                idx.append(n)
            continue
        return random.choice(idx)
