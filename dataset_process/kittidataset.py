
import torch.utils.data as data
from .utils import *
from .visual import mkBVEmap

import os
import numpy as np
import torch
import cv2

_BASE_DIR = 'C:\Datasets/sample/'

class KittiDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        with open(_BASE_DIR + 'train.txt') as f:
            self.lists = f.readlines()
        self.voxelgenerateutils = VoxelGenerateUtils(cfg)
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
                labels_ = self.read_label(_BASE_DIR + 'label/' + self.lists[id][:6] + '.txt', points_, calib_)
                labels_ = KittiObjectUtils.verify_object_inner_boundary(self.boundary, labels_)
                if labels_ is not None:
                    break
                continue
            points, labels = self.randommixup(points, labels, labels_, ra, rx, ry)
        points_, labels_ = self.randompyramid(points, labels)

        #pts_ = KittiObjectUtils.compute_boundary_inner_points(self.boundary, points_.points)
        #map_ = mkBVEmap(pts_, labels_, self.cfg)

        #cv2.imshow('gg', map_)
        #cv2.waitKey(0)

        points_ = KittiObjectUtils.compute_boundary_inner_points(self.boundary, points_.points)
        voxels, coors, num_points_per_voxel = self.voxelgenerateutils.point2voxel(points_)
        voxelmask = VoxelGenerateUtils.paddingmask_kernel(num_points_per_voxel, self.cfg.p2v_maxpoints)

        points_mean =  np.sum(voxels[:, :, :3], axis=1, keepdims=True) / num_points_per_voxel.astype(np.float32).reshape(-1, 1, 1)
        features_relative = voxels[:, :, :3] - points_mean
        voxels = np.concatenate([voxels, features_relative], axis=-1)
        bevsidx, bevcoors, num_voxels_per_bev = self.voxelgenerateutils.voxel2bev(voxels.shape[0], coors)
        bevmask = VoxelGenerateUtils.paddingmask_kernel(num_voxels_per_bev, self.cfg.v2b_maxvoxels)
        target = 0
        return voxels, voxelmask, bevsidx, bevmask, target

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
            #elif o.level == 4:
            #    continue 
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
    
    @staticmethod
    def collate_fn(batch, cuda_idx=0):
        input = []
        targets = []
        for voxels, voxelmask, bevsidx, bevmask, target in batch:
            inputdict = {}
            inputdict['voxels'] = torch.from_numpy(voxels).to(torch.float32).cuda(cuda_idx)
            inputdict['voxelmask'] = torch.from_numpy(voxelmask).unsqueeze(dim=-1).to(torch.float32).cuda(cuda_idx)
            inputdict['bevsidx'] = torch.from_numpy(bevsidx).to(torch.float32).cuda(cuda_idx)
            inputdict['bevmask'] = torch.from_numpy(bevmask).unsqueeze(dim=1).to(torch.float32).cuda(cuda_idx)
            target = torch.from_numpy(target).unsqueeze(0).to(torch.float32)
            input.append(inputdict)
            continue
        targets = torch.cat(targets, dim=0)
        return input
