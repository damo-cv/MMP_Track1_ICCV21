import pickle
import math
import time
import json
from collections import defaultdict
import numpy as np

def distace_batch(point1, point2):
    """Batch distance calculation using numpy."""

    dis = np.linalg.norm(point1-point2, axis=1)
    return dis

def cross_batch(p1, p2, q1, q2):
    """根据两个点表示的在3d空间中的射线，计算他们的公垂线中点位置和在对应射线中的位置"""

    v1 = p2 - p1
    v2 = q2 - q1
    # l1 = p1 + t1 * v1
    # l2 = q1 + t2 * v2
    a = np.sum(v1*v2, axis=1, keepdims=True)
    sample_num = len(a)
    b = np.sum(v1*v1, axis=1, keepdims=True)
    c = np.sum(v2*v2, axis=1, keepdims=True)
    d = np.sum((q1-p1)*v1, axis=1, keepdims=True)
    e = np.sum((q1-p1)*v2, axis=1, keepdims=True)
    t1 = np.zeros((sample_num, 1), dtype=np.float32)
    t2 = np.zeros((sample_num, 1), dtype=np.float32)
    isParallel = np.zeros((sample_num, 1), dtype=np.float32)
    chuizhi_indexes = np.argwhere(a == 0)
    normal_indexes = np.argwhere(np.abs(a*a-b*c) > 0.001)
    normal_indexes = np.setdiff1d(normal_indexes, chuizhi_indexes)
    all_indexes = np.arange(0, sample_num)
    parallel_indexes = np.setdiff1d(np.setdiff1d(all_indexes, chuizhi_indexes), normal_indexes)
    if np.sum(chuizhi_indexes) > 0:
        t1[chuizhi_indexes] = d[chuizhi_indexes]/b[chuizhi_indexes]
        t2[chuizhi_indexes] = -e[chuizhi_indexes]/c[chuizhi_indexes]
    if np.sum(normal_indexes) > 0:
        t1[normal_indexes] = (a[normal_indexes] * e[normal_indexes] - c[normal_indexes] * d[normal_indexes]) / (a[normal_indexes] * a[normal_indexes] - b[normal_indexes] * c[normal_indexes])
        t2[normal_indexes] = b[normal_indexes] * t1[normal_indexes] / a[normal_indexes] - d[normal_indexes] / a[normal_indexes]
    if np.sum(parallel_indexes) > 0:
        isParallel[parallel_indexes] = 1.0
        t2[parallel_indexes] = -d[parallel_indexes] / a[parallel_indexes]
    point1 = p1 + t1*v1
    point2 = q1 + t2*v2
    dis = distace_batch(point1, point2)

    return point1, point2, dis, isParallel

def get_line(x, camera_id, camera_param):
    """Get line in world"""
    # x: n * 2
    
    x_tmp = np.concatenate((x, np.ones((len(x), 1)).astype(np.float32)), axis=1)
    new_x = (x_tmp - camera_param[camera_id]['C']) * camera_param[camera_id]['FInv']
    sample_num = len(new_x)
    cam_origin = np.array([[0, 0, 0]])
    cam_origin = np.repeat(cam_origin, repeats=sample_num, axis=0)
    
    img_point_in_wld = camera_param[camera_id]['Rotation'] @ new_x.transpose(1,0)
    cam_orgin_in_wld = camera_param[camera_id]['Rotation'] @ cam_origin.transpose(1,0)
    
    img_point_in_wld = img_point_in_wld.transpose(1,0) + camera_param[camera_id]['Translation']
    cam_orgin_in_wld = cam_orgin_in_wld.transpose(1,0) + camera_param[camera_id]['Translation']
  
    return img_point_in_wld, cam_orgin_in_wld

def solve_two_points(x1, x2, camera_id1, camera_id2, camera_param):
    """
    Get 3d points with two sets of 2d ponits.
    Ref: https://blog.csdn.net/tiemaxiaosu/article/details/51734667
    """

    img_point_in_wld_1, cam_orgin_in_wld_1 = get_line(x1, camera_id1, camera_param)
    img_point_in_wld_2, cam_orgin_in_wld_2 = get_line(x2, camera_id2, camera_param)
    point1, point2, dis, isParallel = cross_batch(img_point_in_wld_1, cam_orgin_in_wld_1, img_point_in_wld_2, cam_orgin_in_wld_2)
    X = np.mean([point1, point2], axis=0)
    return X, dis

def get_map_coordinate(world_coordinate, camera_param):
    """Transform global world coordinate to map coordinate."""

    footpoint = (world_coordinate - camera_param['min_volume']) * camera_param['discretization_factor']
    return footpoint