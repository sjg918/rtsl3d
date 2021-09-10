
import numpy as np
import cv2

def mkBVEmap(points, labels, cfg):
    voxelshape = cfg.voxelshape
    voxelsize = cfg.voxelsize
    minX = cfg.minX
    maxX = cfg.maxX
    minY = cfg.minY
    maxY = cfg.maxY
    minZ = cfg.minZ
    maxZ = cfg.maxZ

    points_ = np.copy(points)
    # Discretize
    points_[:, 1] = points_[:, 1] - minY
    points_[:, 2] = points_[:, 2] - minZ
    points_[:, 0] = np.int_(np.floor(points_[:, 0] / voxelsize[0]))
    points_[:, 1] = np.int_(np.floor(points_[:, 1] / voxelsize[1]))

    # sort-3times
    indices = np.lexsort((-points_[:, 2], points_[:, 1], points_[:, 0]))
    points_ = points_[indices]

    # Height Map
    heightMap = np.zeros((voxelshape[2]+1, voxelshape[1]+1))

    _, indices = np.unique(points_[:, 0:2], axis=0, return_index=True)
    points_frac = points_[indices]
    max_height = float(np.abs(maxZ - minZ))
    heightMap[np.int_(points_frac[:, 0]), np.int_(points_frac[:, 1])] = points_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((voxelshape[2]+1, voxelshape[1]+1))
    densityMap = np.zeros((voxelshape[2]+1, voxelshape[1]+1))

    _, indices, counts = np.unique(points_[:, 0:2], axis=0, return_index=True, return_counts=True)
    points_top = points_[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(points_top[:, 0]), np.int_(points_top[:, 1])] = points_top[:, 3]
    densityMap[np.int_(points_top[:, 0]), np.int_(points_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, voxelshape[2], voxelshape[1]))
    RGB_Map[2, :, :] = densityMap[:voxelshape[2],:voxelshape[1]]
    RGB_Map[1, :, :] = heightMap[:voxelshape[2],:voxelshape[1]]
    RGB_Map[0, :, :] = intensityMap[:voxelshape[2],:voxelshape[1]]

    #RGB_Map = RGB_Map * 255
    #RGB_Map = np.array(RGB_Map, dtype=np.uint8)
    RGB_Map = np.transpose(RGB_Map, (1,2,0))
    map_ = RGB_Map.copy()
    for object in labels:
        x1, y1 = object.velocorners3d[0, 0], object.velocorners3d[0, 1]
        x2, y2 = object.velocorners3d[1, 0], object.velocorners3d[1, 1]
        x3, y3 = object.velocorners3d[2, 0], object.velocorners3d[2, 1]
        x4, y4 = object.velocorners3d[3, 0], object.velocorners3d[3, 1]
        map_ = cv2.line(map_, ( int((y1 - minY) * 10), int(x1 * 10)), ( int((y2 - minY)* 10), int(x2 * 10)), (0,0,255),1)
        map_ = cv2.line(map_, (int((y2 - minY) * 10), int(x2 * 10)), (int((y3 - minY)* 10), int(x3* 10)), (0,0,255),1)
        map_ = cv2.line(map_, (int((y3 - minY) * 10), int(x3 * 10)), (int((y4 - minY)* 10), int(x4* 10)), (0,0,255),1)
        map_ = cv2.line(map_, (int((y4 - minY) * 10), int(x4 * 10)), (int((y1 - minY)* 10), int(x1* 10)), (0,0,255),1)

        #x1, y1 = object.velox, object.veloy
        #map_ = cv2.circle(map_, int((y1-minY)*10) )

    return map_