
import datetime

import torch
import torch.optim as optim
import torch.nn as nn

from CFG import cfg
from models import *
from dataset_process import *
import time

def load_model(m, p, cuda_ids):
    dict = torch.load(p, map_location='cuda:'+str(cuda_ids))
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)

def eval():
    # start
    print("\n-start- ", "(", datetime.datetime.now(), ")")

    # define model
    back = VoxelFeatureExtractor(cfg).cuda(cfg.cuda_ids[0])
    neck = FeatureAggregator(cfg).cuda(cfg.cuda_ids[0])
    load_model(back, './weights/rtsl3d/backnetw_20.pth', cfg.cuda_ids[0])
    load_model(neck, './weights/rtsl3d/necknetw_20.pth', cfg.cuda_ids[0])

    # define dataloader
    kitti_dataset = KittiDataset(cfg, mode='val')

    # define decoder
    decoder = DecoderWithNMS(cfg)

    back.eval()
    neck.eval()

    with torch.no_grad():
        for i in range(len(kitti_dataset)):
            stepstart = time.time()
            torch.cuda.synchronize()

            voxelList, bevList, calib, filenumber, target = kitti_dataset[i]
            poorvoxels,normvoxels = voxelList
            poorvoxels = torch.from_numpy(poorvoxels).to(dtype=torch.float32, device='cuda:'+kitti_dataset.cuda_id)
            normvoxels = torch.from_numpy(normvoxels).to(dtype=torch.float32, device='cuda:'+kitti_dataset.cuda_id)
            poorbevsidx,normbevsidx,richbevsidx,poorcoors,normcoors,richcoors = bevList
            poorbevsidx = torch.from_numpy(poorbevsidx).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)
            normbevsidx = torch.from_numpy(normbevsidx).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)
            richbevsidx = torch.from_numpy(richbevsidx).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)
            poorcoors = torch.from_numpy(poorcoors).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)
            normcoors = torch.from_numpy(normcoors).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)
            richcoors = torch.from_numpy(richcoors).to(dtype=torch.long, device='cuda:'+kitti_dataset.cuda_id)

            bevfeature = back(poorvoxels, normvoxels, poorbevsidx, normbevsidx, richbevsidx, poorcoors, normcoors, richcoors).unsqueeze(0)
            output = neck(bevfeature)
            decoder(output, calib, True, filenumber)

            torch.cuda.synchronize()
            print(time.time() - stepstart)
            
            continue

    print("\n-end- ", "(", datetime.datetime.now(), ")")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    eval()

