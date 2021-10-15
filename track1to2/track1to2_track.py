import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from utils.utils import parse_camera_param

def global2pixel(person_coords, camera_id, camera_param_dict):
    # det : X Y Z
    world_coord = person_coords / camera_param_dict['discretization_factor'] + camera_param_dict['min_volume']
    trans_coord = world_coord - camera_param_dict[camera_id]['Translation']
    uvw = np.linalg.inv(camera_param_dict[camera_id]['Rotation']) @ trans_coord.transpose(1, 0)
    uvw = uvw.transpose(1, 0)
    pixel_coords = uvw / camera_param_dict[camera_id]['FInv'] / uvw[:, 2:3] + camera_param_dict[camera_id]['C']
    return pixel_coords[:, :2]

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

def batch_ious(det, gt):
    det[:, 2:4] += det[:, :2]
    gt[:, 2:4] += gt[:, :2]
    det = det[:, np.newaxis, :]
    gt = gt[np.newaxis, :, :]
    
    max_x1 = np.maximum(det[..., 0], gt[..., 0])
    min_x2 = np.minimum(det[..., 2], gt[..., 2])
    max_y1 = np.maximum(det[..., 1], gt[..., 1])
    min_y2 = np.minimum(det[..., 3], gt[..., 3])
    
    i = np.maximum(min_y2 - max_y1, 0) * np.maximum(min_x2 - max_x1, 0)
    a1 = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
    a2 = (gt[..., 3] - gt[..., 1]) * (gt[...,2] - gt[..., 0])
    u = a1 + a2 - i
    return i / u

def cos_match(det, res, unmatched_rids, unmatched_cids, cos_th):
    if len(unmatched_rids) == 0 or len(unmatched_cids) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    sub_det = det[unmatched_rids, :]
    sub_res = res[unmatched_cids, :]
    cosine_dist = batch_cosine_dist(sub_det, sub_res)
    matched_rids, matched_cids = linear_sum_assignment(cosine_dist)
    mask = cosine_dist[matched_rids, matched_cids] < cos_th
    matched_rids = matched_rids[mask]
    matched_cids = matched_cids[mask]
    matched_rids = unmatched_rids[matched_rids]
    matched_cids = unmatched_cids[matched_cids]
    return matched_rids, matched_cids

def pos_match(det, res, unmatched_rids, unmatched_cids, pos_th):
    if len(unmatched_rids) == 0 or len(unmatched_cids) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    sub_det = det[unmatched_rids, :]
    sub_res = res[unmatched_cids, :]
    euc_dist = batch_euc_dist(sub_det, sub_res)
    matched_rids, matched_cids = linear_sum_assignment(euc_dist)
    mask = euc_dist[matched_rids, matched_cids] < pos_th
    matched_rids = matched_rids[mask]
    matched_cids = matched_cids[mask]
    matched_rids = unmatched_rids[matched_rids]
    matched_cids = unmatched_cids[matched_cids]
    return matched_rids, matched_cids

def track1to2_track(dets, track1_res, camera_id, camera_param_file, cos_th, pos_th, cos_first=True):
    #track1_res: fid, tid, Y, X, Z
    #dets: fid, cat, x, y, w, h, score, feat 
    camera_param_dict = parse_camera_param(camera_param_file)
    data_for_projection = np.concatenate((track1_res[:, 3:4], track1_res[:, 2:3], track1_res[:, 4:5]), axis=1)
    pixel_coord = global2pixel(data_for_projection, camera_id, camera_param_dict)
    track1_pixel_res = np.hstack((track1_res[:, :2], pixel_coord))# x, y
    det_head_point_x = dets[:, 2] + dets[:, 4] / 2
    det_head_point_y = dets[:, 3] + dets[:, 5] / 2
    det_head_point = np.hstack((det_head_point_x.reshape(-1, 1), det_head_point_y.reshape(-1, 1)))
    det_head = np.hstack((dets[:, 0:1], det_head_point))
    
    fids = np.unique(track1_pixel_res[:, 0])
    res =  []
    tracks_dict = {}
    new_tid = 100
    for fid in sorted(fids):
        sub_det_head = det_head[det_head[:, 0] == fid]
        sub_det = dets[dets[:, 0] == fid]
        sub_track1_res = track1_res[track1_res[:, 0] == fid]
        if len(sub_det_head) == 0:
            continue
        det_size = len(sub_det)
        res_size = len(sub_track1_res)
        det_point = sub_det_head[:, 1:3]
        track1res = track1_pixel_res[track1_pixel_res[:, 0] == fid]
        tids = track1res[:, 1]
        track1res_point = track1res[:, 2:4]
        
        unmatched_rids = np.arange(det_size).astype(int)
        unmatched_cids = np.arange(res_size).astype(int)
        if cos_first:
            matched_rids, matched_cids = cos_match(sub_det[:, 7:], sub_track1_res[:, 5:], unmatched_rids, unmatched_cids, cos_th)
        else:
            matched_rids, matched_cids = pos_match(det_point, track1res_point, unmatched_rids, unmatched_cids, pos_th)
        
        unmatched_rids = set(list(range(det_size))) - set(matched_rids.tolist())
        unmatched_rids = np.array(list(unmatched_rids)).astype(int)
        unmatched_cids = set(list(range(res_size))) - set(matched_cids.tolist())
        unmatched_cids = np.array(list(unmatched_cids)).astype(int)
         
        if cos_first:
            matched_rids2, matched_cids2 = pos_match(det_point, track1res_point, unmatched_rids, unmatched_cids, pos_th)
        else:
            matched_rids2, matched_cids2 = cos_match(sub_det[:, 7:], sub_track1_res[:, 5:], unmatched_rids, unmatched_cids, cos_th) 
 
        matched_rids = np.hstack((matched_rids, matched_rids2))
        matched_cids = np.hstack((matched_cids, matched_cids2))
        
        matched_tids = tids[matched_cids]
        det_res = sub_det[matched_rids, 1:]
        track_res = np.concatenate((sub_det[matched_rids, 0:1], matched_tids.reshape(-1, 1), det_res), axis=1) 
        res.append(track_res)
    res = np.concatenate(res, axis=0)
    return res
            




