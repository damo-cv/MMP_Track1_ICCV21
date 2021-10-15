import numpy as np

def dist_3d(track1, track2, ios_th=0.8):
    assert len(set(track2.keys())) >= len(set(track1.keys()))
    time_intersect = len(set(track1.keys()) & set(track2.keys()))
    time_union = len(set(track1.keys()) | set(track2.keys()))
    ios = time_intersect / len(track1.keys())
    if ios < ios_th:
        return 1000000
    dists = []
    image_ids = set(track1.keys()) | set(track2.keys())
    for iid in image_ids:
        point1 = track1.get(iid, None) # Y X
        point2 = track2.get(iid, None)
        if point1 is not None and point2 is not None:
            dist = np.linalg.norm(point1 - point2)
            dists.append(dist)

    return np.mean(dists)

def nms_3d(tracks, length, dist_th, ios_th):
    length = np.array(length)
    order = length.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        dists = []
        for i in range(1, order.size):
            dist = dist_3d(tracks[order[i]], tracks[order[0]], ios_th)
            dists.append(dist)
        dists = np.array(dists)

        inds = np.where(dists > dist_th)[0]
        order = order[inds + 1]
    return keep

def track_nms(tracks, dist_th, ios_th):
    tracks_dict = {} # transfer to dict format
    for row in tracks:
        frame_idx = int(row[0])
        track_id = int(row[1])
        if track_id not in tracks_dict:
            tracks_dict[track_id] = {frame_idx : row[2:4]}
        else:
            tracks_dict[track_id][frame_idx] = row[2:4]
    tracks_dict = list(tracks_dict.values()) # store tracks as track-nms format
    length_list = [len(trk) for trk in tracks_dict] # track length as 3d nms score
    keep = nms_3d(tracks_dict, length_list, dist_th, ios_th)
    trk_ids = np.unique(tracks[:, 1])
    #print('after track nms, removing ', (len(trk_ids) - len(keep)) ,' tracks')
    valid_ids = trk_ids[keep]
    tracks = np.array([row for row in tracks if row [1] in valid_ids])
    return tracks
