
import numpy as np
import cv2
import torch

def mksparsebev(points, cfg):
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

    # RGB_Map = np.transpose(RGB_Map, (1,2,0))
    # RGB_Map = RGB_Map.copy()
    return RGB_Map

    # # sort-3times
    # indices = np.lexsort((-points_[:, 2], points_[:, 1], points_[:, 0]))
    # points_ = points_[indices]

    # # take max value and counting number of elements
    # max_height = float(np.abs(maxZ - minZ))
    # _, indices, counts = np.unique(points_[:, 0:2], axis=0, return_index=True, return_counts=True)
    # points_ = points_[indices]
    # numpoints = points_.shape[0]

    # # sparsebev Map
    # sparsebev = np.zeros((numpoints*3), dtype=np.float32)
    # sparsebev[:numpoints] = points_[:, 2] / max_height
    # sparsebev[numpoints:numpoints*2] = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    # sparsebev[numpoints*2:] = points_[:, 3]
    
    # # coors
    # coors = np.zeros((numpoints*3, 4), dtype=np.int32)
    # coors[:, 0] = points_[:, 0].tile(3)
    # coors[:, 1] = points_[:, 1].tile(3)
    # coors[:numpoints, 2] = 0
    # coors[numpoints:numpoints*2, 2] = 1
    # coors[numpoints*2:, 2] = 2
    # return sparsebev, coors

    

def VisBevMap(points, labels, cfg, filenumber, mode='save'):
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
    RGB_Map = RGB_Map.copy()

    if labels is not None:
        for object in labels:
            #print('hi', object.velox, object.veloy)
            x1, y1 = object.velocorners3d[0, 0], object.velocorners3d[0, 1]
            x2, y2 = object.velocorners3d[1, 0], object.velocorners3d[1, 1]
            x1 = (x1 / (maxX - minX)) * voxelshape[2]
            y1 = ((y1 - minY) / (maxY - minY)) * voxelshape[1]
            x2 = (x2 / (maxX - minX)) * voxelshape[2]
            y2 = ((y2 - minY) / (maxY - minY)) * voxelshape[1]
            #x3, y3 = object.velocorners3d[2, 0], object.velocorners3d[2, 1]
            #x4, y4 = object.velocorners3d[3, 0], object.velocorners3d[3, 1]
            #map_ = cv2.line(map_, ( int((y1 - minY) * 10), int(x1 * 10)), ( int((y2 - minY)* 10), int(x2 * 10)), (0,0,255),1)
            #map_ = cv2.line(map_, (int((y2 - minY) * 10), int(x2 * 10)), (int((y3 - minY)* 10), int(x3* 10)), (0,0,255),1)
            #map_ = cv2.line(map_, (int((y3 - minY) * 10), int(x3 * 10)), (int((y4 - minY)* 10), int(x4* 10)), (0,0,255),1)
            #map_ = cv2.line(map_, (int((y4 - minY) * 10), int(x4 * 10)), (int((y1 - minY)* 10), int(x1* 10)), (0,0,255),1)
            midx, midy = (x1 + x2) / 2, (y1 + y2) / 2

            x1, y1 = object.velox, object.veloy
            x1 = (x1 / (maxX - minX)) * voxelshape[2]
            y1 = ((y1 -minY) / (maxY - minY)) * voxelshape[1]
            cv2.circle(RGB_Map, (int(y1),int(x1)), 2, (0,0,1),2)
            cv2.arrowedLine(RGB_Map, (int((y1)),int(x1)), (int(midy), int(midx) ), (0,0,1),1)
    
    if mode == 'save':
        cv2.imwrite('./testimages/' + filenumber + '.png', (RGB_Map * 255).astype(np.uint8))
    elif mode == 'show':
        cv2.imshow('./testimages/' + filenumber + '.png', RGB_Map)
        cv2.waitKey()
        cv2.destroyAllWindows()
    #map = torch.from_numpy(RGB_Map)
    #cv2.imwrite('./testimages/' + filenumber + '.png', map_)

import matplotlib.pyplot as plt

def VisTrainLoss(dir):
    runningloss = []
    start = 1
    end = 80

    for i in range(start, end + 1):
        runningloss.append(torch.load(dir + 'runningloss' +str(i)+ '.pth'))

    #bkgloss = torch.cat([runningloss[i][:, 1] for i in range(0, end-start)], dim=0)
    #clsloss = torch.cat([runningloss[i][:, 2] for i in range(0, end-start)], dim=0)
    bkgloss = torch.tensor([runningloss[i][:, 0].mean() for i in range(0, end-start)])
    clsloss = torch.tensor([runningloss[i][:, 1].mean() for i in range(0, end-start)])
    xyzloss = torch.tensor([runningloss[i][:, 5].mean() for i in range(0, end-start)])
    bkgloss = bkgloss.cpu().numpy()
    clsloss = clsloss.cpu().numpy()
    xyzloss = xyzloss.cpu().numpy()
    plt.figure(1)
    plt.plot(bkgloss)
    plt.figure(2)
    plt.plot(clsloss)
    #plt.plot(xyzloss)
    plt.show()
