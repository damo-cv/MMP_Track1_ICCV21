import collections
import numpy as np
from track_obj import TrackData, FrameData, FrameResultData
from sklearn.cluster import AgglomerativeClustering
from utils.utils import batch_cosine_dist
from track_imp import get_match

def get_sep_track(dets):
    #frame_id, tid , x, y, w, h, score, feat
    track_dict = {}
    tids = np.unique(dets[:, 1])
    for tid in tids:
        sub_dets = dets[dets[:, 1] == tid]
        info = {}
        info['frame_ids'] = sub_dets[:, 0].astype(int)
        info['bboxes'] = sub_dets[:, 2:6]
        info['bboxes'][:, 2:4] += info['bboxes'][:, 0:2]
        info['scores'] = sub_dets[:, 6]
        info['flags'] = sub_dets[:, 7]
        info['feats'] = sub_dets[:, 8:]
        track_dict[int(tid)] = info
    return track_dict
 
def get_mean_feats_by_cluster(feats):
    cost_matrix = batch_cosine_dist(feats.copy(), feats.copy())
    #cluster each track2 to two clusters, and get the average feat of the longest track
    cluster_labels = AgglomerativeClustering(n_clusters=2, affinity='precomputed',
                                linkage='average').fit_predict(cost_matrix)
    labels = get_match(cluster_labels)
    selected_labels = labels[0]
    if len(labels[1]) > len(selected_labels):
        selected_labels = labels[1]
    feats = [feats[i].reshape(1, -1) for i in selected_labels]
    feats = np.vstack(feats)
    mean_feats = np.mean(feats, axis=0)
    mean_feats = mean_feats / np.linalg.norm(mean_feats, ord=2)
    return mean_feats

def parse_track_data(camera_id, dets):
    #dets: frame_id, tid , x, y, w, h, score feat
    track_dict = get_sep_track(dets) #{tid: {'frame_ids':, ...}}
    track_data_dic = {}
    for tid in track_dict.keys():
        record_id = (camera_id, tid)
        trail_num = len(track_dict[tid]['frame_ids'])
        bbox_list = track_dict[tid]['bboxes'] # xyxy
        feat_list = track_dict[tid]['feats']
        flag_list = track_dict[tid]['flags']
        trail_data = collections.OrderedDict()
        head_list = np.zeros((len(bbox_list), 2), dtype=np.float32)
        head_list[:, 0] = (bbox_list[:, 0] + bbox_list[:, 2]) / 2
        head_list[:, 1] = (bbox_list[:, 1] + bbox_list[:, 3]) / 2
        for index, fid in enumerate(track_dict[tid]['frame_ids']):
            head = head_list[index]
            frame_bbox_data = FrameData()
            bbox = bbox_list[index]
            feat = feat_list[index]
            flag = flag_list[index]
            frame_bbox_data.set_bbox(bbox)
            frame_bbox_data.set_head(head) #not head acctually, replaced by the center point
            frame_bbox_data.set_feat(feat)
            frame_bbox_data.set_flag(flag)
            trail_data[fid] = frame_bbox_data

        track_data = TrackData()
        track_data.set_trail(trail_data)
        track_data.set_camera_id(camera_id)
        track_data.set_trail_num(trail_num)
        track_data.set_record_id(record_id)

        flags = track_dict[tid]['flags']
        feats = track_dict[tid]['feats']
        valid_feats = feats[flags==1, :]
         
        if len(valid_feats) <= 2:
            mean_feat = np.mean(valid_feats, axis=0)
            mean_feat = mean_feat / np.linalg.norm(mean_feat, ord=2)
        else:
            #cluster each track2 to two clusters, and get the average feat of the longest track
            mean_feat = get_mean_feats_by_cluster(valid_feats)
        
        track_data.set_mean_feat(mean_feat)
        track_data_dic[record_id] = track_data

    return track_data_dic

def get_global_results(global_result_dic, camera_ids):
    res = []
    for global_id in global_result_dic.keys():
        global_track_dict = global_result_dic[global_id]
        for frame_id in sorted(global_track_dict.keys()):
            frame_data = global_track_dict[frame_id]
            coordinate = frame_data.get_footpoint()
            feat = frame_data.get_feat()
            #output fid, tid, Y, X, Z
            out = [frame_id, global_id, coordinate[1], coordinate[0], coordinate[2]]
            out += feat.tolist()
            for cid in camera_ids:
                c_feat = frame_data.get_camera_feat(cid)
                if c_feat is not None:
                    out += c_feat.reshape(-1).tolist()
                else:
                    out += feat.tolist()
            res.append(out)
    res = np.array(res)
    return res
