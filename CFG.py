
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
cfg.voxelsize = (0.1, 0.1, 0.1)
cfg.voxelrange = (0, -30.0, -3.0, 72.0, 30.0, 1.0)
cfg.voxelshape = (40, 600, 720)
cfg.p2v_maxpoints = 16
cfg.p2v_maxvoxels = 100000

# setting for voxel2bev
cfg.bevsize = (0.5, 0.5)
cfg.bevmulti = (int(cfg.bevsize[0] / cfg.voxelsize[1]), int(cfg.bevsize[1] / cfg.voxelsize[2]))
cfg.bevshape = (cfg.voxelshape[1]//cfg.bevmulti[0], cfg.voxelshape[2]//cfg.bevmulti[1])
cfg.v2b_maxvoxels = cfg.voxelshape[0] * cfg.bevmulti[0] * cfg.bevmulti[1]

# setting for back_network
cfg.back_in_channels = 7
cfg.back_voxelwise_channels = 64
cfg.back_bev_in_channels = cfg.v2b_maxvoxels
cfg.back_bev_init_shape = ()
cfg.back_bev_channels = 128
