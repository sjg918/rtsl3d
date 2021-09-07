
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
cfg.maxpoints = 16
cfg.maxvoxels = 20000