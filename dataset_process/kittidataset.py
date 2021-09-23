
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
        coors_re = coors.repeat(voxels.shape[1], axis=0).reshape(voxels.shape[0], voxels.shape[1], 3)
        poorvoxels, normvoxels, coors_, num_points_per_voxel_ =\
           VoxelGenerateUtils.get_dist_and_div_voxels(voxels, coors, coors_re, num_points_per_voxel, voxelmask, self.cfg.p2v_poor)
        voxelmask_ = VoxelGenerateUtils.paddingmask_kernel(num_points_per_voxel_, self.cfg.p2v_maxpoints)
        poorvoxelmask = voxelmask_.sum(axis=1) <= self.cfg.p2v_poor
        normvoxelmask = poorvoxelmask != True
        bevsidx, bevcoors, num_voxels_per_bev = self.voxelgenerateutils.voxel2bev(voxels.shape[0], coors_)
        bevmask = VoxelGenerateUtils.paddingmask_kernel(num_voxels_per_bev, self.cfg.v2b_maxvoxels)
        poorbevsidx, normbevsidx, richbevsidx, poorcoors, normcoors, richcoors =\
           VoxelGenerateUtils.div_bevs(bevsidx, bevcoors, num_voxels_per_bev, bevmask, self.cfg.v2b_poor, self.cfg.v2b_normal)
        
        voxelList= [poorvoxels,normvoxels]
        bevList = [poorbevsidx,normbevsidx,richbevsidx,poorcoors,normcoors,richcoors]
        target = [i.getnumpy_kittiformat_4train(self.cfg.minY, self.cfg.minZ) for i in labels_]
        target = np.concatenate(target).reshape(-1, 7)
        return voxelList, bevList, target

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
    def collate_fn(batch):
        inputs = []
        targets = []
        for voxelList, bevList, target in batch:
            inputdict = {}
            poorvoxels,normvoxels = voxelList
            inputdict['poorvoxels'] = torch.from_numpy(poorvoxels).to(torch.float32)
            inputdict['normvoxels'] = torch.from_numpy(normvoxels).to(torch.float32)
            poorbevsidx,normbevsidx,richbevsidx,poorcoors,normcoors,richcoors = bevList
            inputdict['poorbevsidx'] = torch.from_numpy(poorbevsidx).to(torch.long)
            inputdict['normbevsidx'] = torch.from_numpy(normbevsidx).to(torch.long)
            inputdict['richbevsidx'] = torch.from_numpy(richbevsidx).to(torch.long)
            inputdict['poorcoors'] = torch.from_numpy(poorcoors).to(torch.long)
            inputdict['normcoors'] = torch.from_numpy(normcoors).to(torch.long)
            inputdict['richcoors'] = torch.from_numpy(richcoors).to(torch.long)
            inputs.append(inputdict)

            target = torch.from_numpy(target).to(torch.float32, device)
            targets.append(target)
            continue
        return inputs, targets
