from collections import defaultdict
import json
import numpy as np
import math

def parse_camera_param(camera_config_file):
    """Parse camera param from yaml file."""
    camera_parameters = defaultdict(dict)
    camera_configs = json.load(open(camera_config_file))
    camera_ids = list()

    for camera_param in camera_configs['Cameras']:

        cam_id = camera_param['CameraId']
        camera_ids.append(cam_id)

        camera_parameters[cam_id]['Translation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Translation'])[np.newaxis, :]
        camera_parameters[cam_id]['Rotation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Rotation']).reshape((3, 3))

        camera_parameters[cam_id]['FInv'] = np.asarray([
                1 / camera_param['IntrinsicParameters']['Fx'],
                1 / camera_param['IntrinsicParameters']['Fy'], 1
            ])[np.newaxis, :]
        camera_parameters[cam_id]['C'] = np.asarray([
                camera_param['IntrinsicParameters']['Cx'],
                camera_param['IntrinsicParameters']['Cy'], 0
            ])[np.newaxis, :]

        discretization_factorX = 1.0 / (
            (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) / (math.floor(
                (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) /
                camera_configs['Space']['VoxelSizeInMM']) - 1))
        discretization_factorY = 1.0 / (
            (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) / (math.floor(
                (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) /
                camera_configs['Space']['VoxelSizeInMM']) - 1))
        camera_parameters['discretization_factor'] = np.asarray([discretization_factorX, discretization_factorY, 1])

        camera_parameters['min_volume'] = np.asarray([
            camera_configs['Space']['MinU'], camera_configs['Space']['MinV'],
            camera_configs['Space']['MinW']
        ])
    
    return camera_parameters

def batch_euc_dist(point1, point2):
    point1_reshape = point1[:, np.newaxis, :]
    point2_reshape = point2[np.newaxis, :, :]
    sub = point1_reshape - point2_reshape
    dist = np.linalg.norm(sub, ord=2, axis=-1)
    return dist

def batch_cosine_dist(feat1, feat2):
    assert feat1.shape[1] == feat2.shape[1]
    feat1 = feat1 / np.linalg.norm(feat1, ord=2, axis=-1, keepdims=True)
    feat2 = feat2 / np.linalg.norm(feat2, ord=2, axis=-1, keepdims=True)
    sim_matrix = feat1 @ feat2.T
    return 1 - sim_matrix