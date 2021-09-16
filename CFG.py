
from easydict import EasyDict

cfg = EasyDict()

# setting for ROI
cfg.minX = 0
cfg.maxX = 72
cfg.minY = -30
cfg.maxY = 30
cfg.minZ = -3
cfg.maxZ = 1

# setting for point2voxel
cfg.voxelsize = (0.2, 0.2, 0.2)
cfg.voxelrange = (0, -32.0, -3.0, 80.0, 32.0, 1.0)
cfg.voxelshape = (
    int((cfg.voxelrange[5] - cfg.voxelrange[2]) / cfg.voxelsize[2]),
    int((cfg.voxelrange[4] - cfg.voxelrange[1]) / cfg.voxelsize[1]),
    int((cfg.voxelrange[3] - cfg.voxelrange[0]) / cfg.voxelsize[0]))
cfg.p2v_maxpoints = 8
cfg.p2v_maxvoxels = 20000

# setting for voxel2bev
cfg.bevsize = (0.2, 0.2)
cfg.bevmulti = (round(cfg.bevsize[0] / cfg.voxelsize[0]), round(cfg.bevsize[1] / cfg.voxelsize[1]))
cfg.bevshape = (cfg.voxelshape[1]//cfg.bevmulti[0], cfg.voxelshape[2]//cfg.bevmulti[1])
cfg.v2b_maxvoxels = cfg.voxelshape[0] * cfg.bevmulti[0] * cfg.bevmulti[1]

# setting for back_network
cfg.back_in_channels = 7
cfg.back_voxelwise_channels = 64
cfg.back_bev_in_channels = cfg.v2b_maxvoxels
cfg.back_bev_init_shape = ()
cfg.back_bev_channels = 128
