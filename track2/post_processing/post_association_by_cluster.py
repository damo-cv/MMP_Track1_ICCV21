import numpy as np
from sklearn.cluster import AgglomerativeClustering
INFTY_COST = 1e+5

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b.T)

def noverlap(period1, period2):
    #s1 = period1[0]
    #e1 = period1[-1]
    #s2 = period2[0]
    #e2 = period2[-1]
    s1 = np.min(period1)
    e1 = np.max(period1)
    s2 = np.min(period2)
    e2 = np.max(period2)
        
    if (0 < s2 - e1) or (0 < s1 - e2):
        return True
     
    return False

def reid_similarity(det1, det2, start_cols):
    feat1 = det1[:, start_cols:]
    #print(feat1.shape)
    feat2 = det2[:, start_cols:]
    avg_feat1 = np.mean(feat1, axis=0)
    avg_feat2 = np.mean(feat2, axis=0)
    return cosine_similarity(avg_feat1, avg_feat2)

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

def associate(det, threshold, start_cols, identity='tmp'):
    #processed_track_list = []
    tids = np.unique(det[:, 1])
    cost_m = np.ones((len(tids), len(tids))) * INFTY_COST
    edges = []
    min_dis = 1000
    for i in range(len(tids) - 1):
        trk_i = det[det[:, 1] == tids[i]]
        image_ids_i = trk_i[:, 0]
        # ignore len 1 track
        if (trk_i.shape[0] == 1): continue
        for j in range(i+1, len(tids)):
            trk_j = det[det[:, 1] == tids[j]]
            if (trk_j.shape[0] == 1): continue
            image_ids_j = trk_j[:, 0]
            if noverlap(image_ids_i, image_ids_j):
                similarity = reid_similarity(trk_i, trk_j, start_cols)
                cost_m[i,j] = similarity
                cost_m[j,i] = similarity
                min_dis = min(similarity, min_dis)
                
    print('min_dist: ', min_dis)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='precomputed',
                                linkage='average').fit_predict(cost_m)
    
    labels = get_match(cluster_labels)

    for l in labels:
        if len(l) > 1:
            base_tid = tids[l[0]]
            for id in l[1:]:
                det[det[:, 1] == tids[id], 1] = base_tid
    
    return det
