
from easydict import EasyDict

cfg = EasyDict()

# setting for device
cfg.cuda_ids = [0]

# setting for ROI
cfg.minX = 0
cfg.maxX = 72
cfg.minY = -30
cfg.maxY = 30
cfg.minZ = -3
cfg.maxZ = 1

# setting for point2voxel
cfg.voxelsize = (0.1, 0.1, 0.1)
cfg.voxelrange = (0, -30.0, -3.0, 72.0, 30.0, 1.0)
cfg.voxelshape = (
    int((cfg.voxelrange[5] - cfg.voxelrange[2]) / cfg.voxelsize[2]),
    int((cfg.voxelrange[4] - cfg.voxelrange[1]) / cfg.voxelsize[1]),
    int((cfg.voxelrange[3] - cfg.voxelrange[0]) / cfg.voxelsize[0]))
cfg.p2v_poor = 2
cfg.p2v_maxpoints = 8
cfg.p2v_maxvoxels = 40000

# setting for voxel2bev
cfg.bevsize = (0.5, 0.5)
cfg.bevmulti = (round(cfg.bevsize[0] / cfg.voxelsize[0]), round(cfg.bevsize[1] / cfg.voxelsize[1]))
cfg.bevshape = (cfg.voxelshape[1]//cfg.bevmulti[0], cfg.voxelshape[2]//cfg.bevmulti[1])
cfg.v2b_poor = 2
cfg.v2b_normal = 16
cfg.v2b_maxvoxels = 128

# setting for back_network
cfg.back_in_channels = 7
cfg.back_voxelwise_channels = 16
cfg.back_bev_in_channels = 32

# setting for neck_network
cfg.neck_bev_in_channels = cfg.back_bev_in_channels * 2
cfg.neck_channels = [128, 256, 384, 384]
cfg.neck_bev_out_channels = 1 + 3 + 3 + 2

# setting for loss_function
cfg.outputmaxX = cfg.bevshape[1] // 2
cfg.outputmaxY = cfg.bevshape[0] // 2
cfg.meandims = [1.6, 3.9, 1.56]
cfg.vaildiou = 0.5

# setting for training
cfg.num_cpu = 8
cfg.batchsize = 4
cfg.learing_rate = 0.001
cfg.maxepoch = 80
cfg.saveplace = 'rtsl3d'
cfg.rampup_epoch = 15
def sigmoid_shaped_schedule(i):
    if i < cfg.rampup_epoch:
        factor = math.exp(-5 * (1 - i/cfg.rampup_epoch) * (1 - i/cfg.rampup_epoch))
    else:
        factor = 1
    return factor
cfg.rampup_schedule = sigmoid_shaped_schedule

#
cfg.vaildconf = 0.5
