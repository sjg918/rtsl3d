
import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from CFG import cfg
from models import *
from dataset_process import *

def load_model(m, p, cuda_ids):
    dict = torch.load(p, map_location='cuda:'+str(cuda_ids))
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)

def train():
    #
    # torch.manual_seed(7)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(7)

    # start
    print("\n-start- ", "(", datetime.datetime.now(), ")")

    # define model
    back = VoxelFeatureExtractor(cfg).cuda(cfg.cuda_ids[0])
    neck = FeatureAggregator(cfg).cuda(cfg.cuda_ids[0])
    #back = nn.DataParallel(back, device_ids=cuda_ids)
    #head = nn.DataParallel(head, device_ids=cuda_ids)

    # define dataloader
    kitti_dataset = KittiDataset(cfg)
    kitti_loader = DataLoader(kitti_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,\
       pin_memory=False, drop_last=True, collate_fn=kitti_dataset.collate_fn_gpu)

    # define optimizer and scheduler
    back_optimizer = optim.Adam(back.parameters(), lr=cfg.learing_rate, betas=(0.9, 0.999), eps=1e-08)
    back_scheduler = optim.lr_scheduler.LambdaLR(back_optimizer, cfg.rampup_schedule)
    neck_optimizer = optim.Adam(neck.parameters(), lr=cfg.learing_rate, betas=(0.9, 0.999), eps=1e-08)
    neck_scheduler = optim.lr_scheduler.LambdaLR(neck_optimizer, cfg.rampup_schedule)

    # define loss function
    lossfunc = MultiLoss(cfg, cfg.cuda_ids[0])

    back.train()
    neck.train()

    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        print("\n(", str(epoch), "/", str(cfg.maxepoch), ") -epoch- ", "(", datetime.datetime.now(), ")")
        
        for cnt, (inputs, targets) in enumerate(kitti_loader):           
            # forward
            bevfeature = [back(input['poorvoxels'], input['normvoxels'],\
                input['poorbevsidx'], input['normbevsidx'], input['richbevsidx'],\
                input['poorcoors'], input['normcoors'], input['richcoors']).unsqueeze(0)\
                          for input in inputs]
            bevfeature = torch.cat(bevfeature, dim=0)
            output = neck(bevfeature)
            loss = lossfunc(output, targets)

            # backward
            loss.backward()
            back_optimizer.step()
            neck_optimizer.step()
            back.zero_grad()
            neck.zero_grad()
            
            # print steploss
            print("{}/{} steploss:{:.6f}".format(cnt, len(kitti_loader), loss.item()), end="\r")
            continue

        # learning rate scheduling
        back_scheduler.step()
        neck_scheduler.step()

        # save model
        if epoch % 10 == 0:
            torch.save(back.state_dict(), './weights/' + cfg.saveplace + '/backnetw_' + str(epoch) + '.pth')
            torch.save(neck.state_dict(), './weights/' + cfg.saveplace + '/necknetw_' + str(epoch) + '.pth')
            torch.save(back_optimizer.state_dict(), './weights/' + cfg.saveplace + '/backopti_' + str(epoch) + '.pth')
            torch.save(neck_optimizer.state_dict(), './weights/' + cfg.saveplace + '/neckopti_' + str(epoch) + '.pth')
            print('\n{}epoch model saved !'.format(epoch), "(", datetime.datetime.now(), ")")

        continue
    # end.
    torch.save(back.state_dict(), './weights/' + cfg.saveplace + '/backnetw_end.pth')
    torch.save(neck.state_dict(), './weights/' + cfg.saveplace + '/necknetw_end.pth')
    torch.save(back_optimizer.state_dict(), './weights/' + cfg.saveplace + '/backopti_' + str(epoch) + '.pth')
    torch.save(neck_optimizer.state_dict(), './weights/' + cfg.saveplace + '/neckopti_' + str(epoch) + '.pth')
    print("\n-end- ", "(", datetime.datetime.now(), ")")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()

