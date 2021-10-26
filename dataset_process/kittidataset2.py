
import torch.utils.data as data
from .utils import *
from .visual import *

import os
import numpy as np
import torch
import cv2

_BASE_DIR = '/home/user/DataSet/kitti_3dobject/KITTI/object/training/'

class KittiDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cuda_id = str(cfg.cuda_ids[0])
        self.cfg = cfg
        self.mode = mode
        if mode == 'train':
            with open(_BASE_DIR + 'train.txt') as f:
                self.lists = f.readlines()
        elif mode == 'val':
            with open(_BASE_DIR + 'val.txt') as f:
                self.lists = f.readlines()
        self.mosaicprobability = cfg.mosaicprobability
        self.voxelgenerateutils = VoxelGenerateUtils(cfg)
        self.basetrans = BaseTransform(cfg)
        self.randommixup = RandomMixup(cfg)
        self.boundary = np.array([cfg.minX, cfg.maxX,
                                  cfg.minY, cfg.maxY, cfg.minZ, cfg.maxZ], dtype=np.float32)
    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):

        if self.mode == 'val':
            points = self.read_velo2val(_BASE_DIR + 'velodyne(lidar)/' + self.lists[idx][:6] + '.bin')
            calib  = self.read_calib2val(_BASE_DIR + 'calib/' + self.lists[idx][:6] + '.txt')
            #labels = self.read_label(_BASE_DIR + 'label_2/' + self.lists[idx][:6] + '.txt', points, calib)
            labels = None
            points_ = KittiObjectUtils.compute_boundary_inner_points(self.boundary, points)
            mksparsebev(points_, self.cfg)
            df=df
            #return voxels, coors, num_points_per_voxel, calib, self.lists[idx][:6], points_
            if labels != None:
                target = [i.getnumpy_kittiformat_4train(0, 0) for i in labels]
                target = np.concatenate(target).reshape(-1, 7)
                return voxels, coors, num_points_per_voxel, calib, self.lists[idx][:6], points_, target
            else:
                return voxels, coors, num_points_per_voxel, calib, self.lists[idx][:6], points_, None
                
        while True:
            velo = self.read_velo(_BASE_DIR + 'velodyne(lidar16ch)/' + self.lists[idx][:6] + '.bin')
            pseudo = self.read_velo(_BASE_DIR + 'velodyne_pesudo/' + self.lists[idx][:6] + '.bin')
            calib  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[idx][:6] + '.txt')
            velolabels = self.read_label(_BASE_DIR + 'label_2/' + self.lists[idx][:6] + '.txt', velo, calib)
            pseudolabels = self.read_label(_BASE_DIR + 'label_2/' + self.lists[idx][:6] + '.txt', pseudo, calib)
            velo, pseudo, labels = self.basetrans(velo, pseudo, velolabels, pseudolabels)
            if labels is None:
                idx = random.randint(0, len(self.lists) - 1)
                continue
            else:
                break

        while True:
            id = random.randint(0, self.__len__() - 1)
            velo_ = self.read_velo(_BASE_DIR + 'velodyne(lidar16ch)/' + self.lists[id][:6] + '.bin')
            calib_  = self.read_calib(_BASE_DIR + 'calib/' + self.lists[id][:6] + '.txt')
            velolabels_ = self.read_label(_BASE_DIR + 'label_2/' + self.lists[id][:6] + '.txt', velo_, calib_)
            velolabels_ = KittiObjectUtils.verify_object_inner_boundary(self.boundary, velolabels_)
            if velolabels_ is not None:
                pseudo_ = self.read_velo(_BASE_DIR + 'velodyne_pesudo/' + self.lists[id][:6] + '.bin')
                pseudolabels_ = self.read_label(_BASE_DIR + 'label_2/' + self.lists[id][:6] + '.txt', pseudo_, calib_)
                pseudolabels_ = KittiObjectUtils.verify_object_inner_boundary(self.boundary, pseudolabels_)
                break
            continue
        velo, pseudo, labels = self.randommixup(velo, pseudo, labels, velolabels_, pseudolabels_)

        target = [i.getnumpy_kittiformat_4train(self.cfg.minY, self.cfg.minZ) for i in labels]
        target = np.concatenate(target).reshape(-1, 7)
        # VisBevMap(points_, labels_, self.cfg, self.lists[idx][:6])
        # df=df
        # remove outer points
        velo.points = KittiObjectUtils.compute_boundary_inner_points(self.boundary, velo.points)
        if pseudo is not None:
            pseudo.points = KittiObjectUtils.compute_boundary_inner_points(self.boundary, pseudo.points)

        velobev = mksparsebev(velo.points, self.cfg)
        if pseudo is not None:
            pseudobev = mksparsebev(pseudo.points, self.cfg)
            #velobev = np.concatenate((velobev, pseudobev), axis=0)
            return velobev, pseudobev, target
        return velobev, target

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

    def read_velo2val(self, filepath):
        return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

    def read_calib2val(self, filename):
        return Calibration2val(filename)

    def random_select_area(self, stat):
        if False not in stat:
            return None
        
        idx = []
        for n, i in enumerate(stat):
            if i == False:
                idx.append(n)
            continue
        return random.choice(idx)
    
    def collate_fn_cpu(self, batch):
        bev_list = []
        target_list = []
        for i, (bev, target) in enumerate(batch):
            bev_list.append(torch.from_numpy(bev).unsqueeze(0))
            target = torch.from_numpy(target)
            target_list.append(target)
            continue
        bev_list = torch.cat(bev_list, dim=0)
        return bev_list, target_list

    def collate_fn_cpu2(self, batch):
        velobev_list = []
        pseudobev_list = []
        target_list = []
        for i, (velobev, pseudobev, target) in enumerate(batch):
            velobev_list.append(torch.from_numpy(velobev).unsqueeze(0))
            pseudobev_list.append(torch.from_numpy(pseudobev).unsqueeze(0))
            target = torch.from_numpy(target)
            target_list.append(target)
            continue
        velobev_list = torch.cat(velobev_list, dim=0)
        pseudobev_list = torch.cat(pseudobev_list, dim=0)
        return velobev_list, pseudobev_list, target_list

    def collate_fn_gpu(self, batch):
        bev_list = []
        target_list = []
        for i, (bev, target) in enumerate(batch):
            bev_list.append(torch.from_numpy(bev).unsqueeze(0).to(dtype=torch.float32, device='cuda:'+self.cuda_id))
            target = torch.from_numpy(target).to(dtype=torch.float32, device='cuda:'+self.cuda_id)
            target_list.append(target)
            continue
        bev_list = torch.cat(bev_list, dim=0)
        return bev_list, target_list
