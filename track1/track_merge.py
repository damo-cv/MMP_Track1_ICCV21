import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
INFTY_COST = 1e+5

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_euc_distance(track1, track2):
    dists = []
    frame_ids1 = set(track1.keys())
    frame_ids2 = set(track2.keys())
    fid_intersect = frame_ids1 & frame_ids2
    if len(fid_intersect) == 0:
        return INFTY_COST
    for fid in fid_intersect:
        point1 = track1[fid][2:4]
        point2 = track2[fid][2:4]
        dist = np.linalg.norm(point1 - point2, ord=2)
        dists.append(dist)    
    return np.mean(dists)

def merge_tracks(tracks_dict, cand_tids):
    base_tid = cand_tids[0]
    all_frame_ids = set()
    for tid in cand_tids:
        frame_ids = set(tracks_dict[tid].keys())
        all_frame_ids = all_frame_ids | frame_ids
    
    res = []
    for fid in sorted(list(all_frame_ids)):
        row = []
        for tid in cand_tids:
            if fid in tracks_dict[tid]:
                row.append(tracks_dict[tid][fid][2:])
        row = np.mean(np.asarray(row), axis=0)
        fid_tid_pad = np.array([fid, base_tid])
        row = np.hstack((fid_tid_pad, row))
        res.append(row)
    return np.asarray(res)

def merge_overlap_track(det, threshold):
    match = []
    tids = np.unique(det[:, 1])
    tracks_dict = defaultdict(dict)
    for row in det:
        frame_idx = int(row[0])
        track_id = int(row[1])
        tracks_dict[track_id][frame_idx] = row
    
    cost_matrix = np.zeros((len(tids), len(tids))) + INFTY_COST
    for i in range(len(tids)-1):
        for j in range(i+1, len(tids)):
            track_i = tracks_dict[tids[i]]
            track_j = tracks_dict[tids[j]]
            cost_matrix[i, j] = get_euc_distance(track_i, track_j)
            cost_matrix[j, i] = cost_matrix[i, j]

    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='precomputed',
                                linkage='single').fit_predict(cost_matrix)
    
    labels = get_match(cluster_labels)

    results = []
    for l in labels:
        if len(l) == 1:
            tid = tids[l[0]]
            res = det[det[:, 1] == tid]
            results.append(res)
        else:
            cand_tids = [tids[i] for i in l]
            res = merge_tracks(tracks_dict, cand_tids)
            results.append(res)
    
    results = np.vstack(results)

    return results
