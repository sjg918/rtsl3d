import numpy as np
import os

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def velo_points_2_pano(points, v_res, h_res, v_fov, h_fov, depth=False):

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # project point cloud to 2D point map
    x_img = np.arctan2(-y, x) / (h_res * (np.pi / 180))
    y_img = -(np.arctan2(z, dist) / (v_res * (np.pi / 180)))
    
    # shift negative points to positive points (shift minimum value to 0)
    x_offset = h_fov[0] / h_res
    x_img = np.trunc(x_img - x_offset).astype(np.int32)
    y_offset = v_fov[1] / v_res
    y_fine_tune = 1
    y_img = np.trunc(y_img + y_offset + y_fine_tune).astype(np.int32)

    use_index = (x_img % 2 == 0) & (y_img % 4 == 0)
    use_point = points[use_index]
    
    return use_point

# bin file -> numpy array
home = '/home/user/DataSet/kitti_3dobject/KITTI/object/training/velodyne(lidar)'
newhome = '/home/user/DataSet/kitti_3dobject/KITTI/object/training/velodyne(lidar16ch)'
hi = os.listdir('/home/user/DataSet/kitti_3dobject/KITTI/object/training/velodyne(lidar)')

for i in hi:
    velo_points = load_from_bin(home + '/' + i)
    print(velo_points.shape)
    new_points = velo_points_2_pano(velo_points, 0.4, 0.35, (-24.9, 2.0), (-180, 180), False)
    print(new_points.shape)
    df=df
    new_points.tofile(newhome + '/' + i)
