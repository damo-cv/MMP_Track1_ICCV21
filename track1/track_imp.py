import collections
import time
import numpy as np
from transform import solve_two_points, get_map_coordinate
from track_obj import FrameResultData
from sklearn.cluster import AgglomerativeClustering
LARGE_COST = 10000.0

def get_reid_sim(track_data_a, track_data_b):
    """获取两个track的平均reid特征的cosine相似度，特征已经标准化"""
    feat_a = track_data_a.get_mean_feat()
    feat_b = track_data_b.get_mean_feat()
    sim_appearance = np.dot(feat_a, feat_b)
    return sim_appearance

def get_pos_cost(track_data_a, track_data_b, camera_param_dic):
    """获取两个track在空间坐标上的距离，顺便计算不同camera计算出的空间坐标"""

    temp_world_detection_base = {'pos_cost': LARGE_COST, 'max_pos_cost': LARGE_COST, 'min_pos_cost': LARGE_COST, 'all_pos_cost': [LARGE_COST], 'time_iou': 0.0, 'time_dist': 0, 'match_stamp_iou': 0.0, 'match_stamp_num': 0, 'person_height': 3.0}    # match_stamp_iou = 0.001 is dummy value
    stamp_a_list = track_data_a.get_trail_stamp_list()
    stamp_b_list = track_data_b.get_trail_stamp_list()

    # do they have overlap in time?
    if stamp_a_list[0] > stamp_b_list[-1] or stamp_a_list[-1] < stamp_b_list[0]:
        temp_world_detection_base['time_dist'] = max(stamp_a_list[0] - stamp_b_list[-1], stamp_b_list[0] - stamp_a_list[-1])
        return temp_world_detection_base

    camera_a = track_data_a.get_camera_id()
    camera_b = track_data_b.get_camera_id()
    inter_list = np.intersect1d(stamp_a_list, stamp_b_list)
    union_list = np.union1d(stamp_a_list, stamp_a_list)
    # the two tracks must have time overlap and not in the same camera.
    if camera_a != camera_b and len(inter_list) > 0:
        temp_world_detection = batch_compare_track(inter_list, union_list, track_data_a, track_data_b, camera_param_dic)
        temp_world_detection_base.update(temp_world_detection)
    temp_world_detection_base['time_iou'] = len(inter_list) / float(len(union_list))
    return temp_world_detection_base

def batch_compare_track(inter_list, union_list, track_data_a, track_data_b, camera_param_dic):
    """根据两个有overlap切camera id不同的track，利用他们的overlap部分计算出空间距离和空间中垂线坐标."""
    record_id_a = track_data_a.get_record_id()
    record_id_b = track_data_b.get_record_id()
    camera_a = track_data_a.get_camera_id()
    camera_b = track_data_b.get_camera_id()

    person_height = 3.0
    stamp_list = []
    head_a_list = []
    head_b_list = []
    for index_a in range(len(inter_list)):
        time_stamp = inter_list[index_a]
        stamp_data_a = track_data_a.get_stamp_data(time_stamp)
        head_a = stamp_data_a.get_head()
        stamp_data_b = track_data_b.get_stamp_data(inter_list[index_a])
        head_b = stamp_data_b.get_head()
        head_a_list.append(head_a)
        head_b_list.append(head_b)
        stamp_list.append(time_stamp)
    world_list, loss_list = solve_two_points(np.array(head_a_list).reshape(-1,2), np.array(head_b_list).reshape(-1,2), camera_a, camera_b, camera_param_dic)
    # loss_list 表示两个点在空间中的欧氏距离
    mean_loss = np.mean(loss_list)

    pos_cost = mean_loss
    if len(world_list) > 0:
        person_height = np.mean(np.array(world_list)[:, -1])
    all_pos_cost = {}
    for good_index, good_stamp in enumerate(stamp_list):
        track_data_a.add_map_time_stamp(good_stamp)
        stamp_data_a = track_data_a.get_stamp_data(good_stamp)
        stamp_data_a.put_temp_world_dict(record_id_a, record_id_b, world_list[good_index])
        stamp_data_b = track_data_b.get_stamp_data(good_stamp)
        stamp_data_b.put_temp_world_dict(record_id_b, record_id_a, world_list[good_index])
        track_data_b.add_map_time_stamp(good_stamp)
        all_pos_cost[good_stamp] = loss_list[good_index]
    temp_world_detection = {'pos_cost': pos_cost, 'max_pos_cost': np.max(loss_list), 'all_pos_cost': all_pos_cost, 'min_pos_cost': np.min(loss_list),
                 'person_height': person_height}
    return temp_world_detection

def get_tracks_sim(track_data_dic, camera_param_dic, track_list):
    """获取track两两之间的距离"""
    track_num = len(track_list)
    sim_list = [[{} for _ in range(track_num)] for _ in range(track_num)]
    reid_time, pos_time = 0.0, 0.0
    sim_dict = {}
    for index_a, record_a in enumerate(track_list):
        sim_list[index_a][index_a] = {'pos_cost': LARGE_COST, 'reid_sim': 0.0, 'match_stamp_iou': 1.0, 'time_iou': 1.0, 'time_dist': 0.0, 'camera_match': 1.0, 
           'max_pos_cost': LARGE_COST, 'match_stamp_num': 0.0, 'min_pos_cost': LARGE_COST, 'all_pos_cost': [], 'person_height': 3.0}
        camera_a = track_data_dic[record_a].get_camera_id()
        for i, record_b in enumerate(track_list[index_a+1:]):
            camera_b = track_data_dic[record_b].get_camera_id()
            start_time = time.time()
            sim_appearance = get_reid_sim(track_data_dic[record_a], track_data_dic[record_b])
            reid_time += (time.time() - start_time)
            start_time = time.time()
            temp_world_detection = get_pos_cost(track_data_dic[record_a], track_data_dic[record_b], camera_param_dic)
            pos_time += (time.time() - start_time)
            temp_world_detection['reid_sim'] = sim_appearance
            temp_world_detection['camera_match'] = camera_a == camera_b
            sim_list[index_a][index_a+1+i] = temp_world_detection
            sim_list[index_a+1+i][index_a] = temp_world_detection
            sim_dict[(record_a, record_b)] = temp_world_detection
            sim_dict[(record_b, record_a)] = temp_world_detection
    print('reid time %.4f s and pos time %.4f s' % (reid_time, pos_time))
    return sim_list, sim_dict

def get_cost_matrix(track_sim_list, track_data_dic, track_list):
    """综合reid和空间距离，以及track之间的关系，构建距离矩阵"""
    cost_matrix = np.zeros((len(track_sim_list), len(track_sim_list[0])), np.float32) + LARGE_COST
    track_num = len(track_sim_list)
    for index_a in range(track_num-1):
        for index_b in range(index_a+1, track_num):
            track_sim = track_sim_list[index_a][index_b]
            pos_cost = track_sim['pos_cost']
            min_pos_cost = track_sim['min_pos_cost']
            reid_sim = track_sim['reid_sim']
            cost_matrix[index_a][index_b] = 1 - reid_sim
            cost_matrix[index_b][index_a] = cost_matrix[index_a][index_b]
            
    return cost_matrix

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

def merge_by_sim(track_sim_list, track_data_dic, track_list, reid_th):
    """
    Merge by sim.
    Ref: https://stackoverflow.com/questions/30089675/clustering-cosine-similarity-matrix
    """

    print('start clustering')
    merge_start_time = time.time()
    cost_matrix = get_cost_matrix(track_sim_list, track_data_dic, track_list)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=reid_th, affinity='precomputed',
                                linkage='average').fit_predict(cost_matrix)
    labels = get_match(cluster_labels)
    # print(merged_index_list)
    
    print('we have %d global tracks after merge, time for merge %.4f s' % (len(labels), time.time()-merge_start_time))

    # get real data
    valid_global_list = []
    valid_count = 0
    for person_track_list in labels:
        temp = []
        for index in person_track_list:
            record_name = track_list[index]
            temp.append(record_name)
        if len(temp) > 1:
            cameras = set([t[0] for t in temp])
            if len(cameras) > 1:
                valid_count += 1
                valid_global_list.append(temp)
        #clustered_list.append(temp)
    print(f'after merge, %d valid global ids are created: {valid_global_list}' % valid_count)
    return valid_global_list

def get_average_result(clustered_record_list, track_data_dic, camera_param_dic, sim_dict, len_th):
    """获取最终经过cluster后目标的世界坐标了."""

    global_result_dic = {}
    for global_id, person_track_list in enumerate(clustered_record_list):
        stamp_list = []
        for record_id in person_track_list:
            stamp_list += track_data_dic[record_id].get_map_time_stamp()
        unique_stamp, unique_counts = np.unique(stamp_list, return_counts=True)
        # at least two cameras match, they should have the same time stamp
        common_stamp = unique_stamp[unique_counts > 1]

        global_trail_data = collections.OrderedDict()
        for time_stamp in common_stamp:
            # we are sure these common stamp should be ROUND_VALUE times
            world_list = []
            feat_dict = {}
            flag_dict = {}
            dist_list = []
            record_dict = {}
            record_list = []
            for record_id in person_track_list:
                track_data = track_data_dic[record_id]
                stamp_data = track_data.get_stamp_data(time_stamp)
                if stamp_data is None:
                    continue
                feat = stamp_data.get_feat()
                flag = stamp_data.get_flag()
                feat_dict[record_id[0]] = feat.reshape(1, -1)
                flag_dict[record_id[0]] = flag
                # temp_world_list contains every pair 3d data which has not been merged. We should delete the bad matches here.
                true_world_list = []
                true_record_list = []
                true_dist_list = []
                word_dict = stamp_data.get_temp_world_dict()
                
                for record_key in word_dict.keys():
                    record_b = record_key[1]
                    if record_b not in person_track_list:
                        continue
                    if (record_id, record_b) in record_dict or (record_b, record_id) in record_dict:
                        continue
                    
                    true_world_list.append(word_dict[record_key])
                    true_dist_list.append(sim_dict[(record_id, record_b)]['all_pos_cost'][time_stamp])
                    record_dict[(record_id, record_b)] = 1
                    true_record_list.append((record_id, record_b))
                world_list += true_world_list
                dist_list += true_dist_list
                record_list += true_record_list
            if len(world_list) == 0:
                continue
            new_world_list = []
            new_record_list = []
            for i, pos in enumerate(dist_list):
                if pos <= len_th:
                    new_world_list.append(world_list[i])
                    new_record_list.append(record_list[i])
            if not len(new_world_list):
                new_world_list.append(world_list[np.argmin(dist_list)])
                new_record_list.append(record_list[np.argmin(dist_list)])
            if len(new_world_list) > 2:
                new_world_list, labels = get_clustered_world_list(new_world_list)
                new_record_list = [new_record_list[i] for i in labels]
            common_world = np.mean(new_world_list, axis=0, keepdims=False)
            footpoint = get_map_coordinate(common_world, camera_param_dic)
            frame_result_data = FrameResultData()
            frame_result_data.set_world(common_world)
            frame_result_data.set_footpoint(footpoint)
            remain_camera_ids = set()
            for record in new_record_list:
                remain_camera_ids.add(record[0][0])
                remain_camera_ids.add(record[1][0])
            feat_list = []
            for cid in remain_camera_ids:
                if flag_dict[cid] == 1:
                    frame_result_data.set_camera_feat(cid, feat_dict[cid])
                    feat_list.append(feat_dict[cid])
            if not len(feat_list):
                feat_list = [feat_dict[cid] for cid in feat_dict.keys() if flag_dict[cid] == 1]
                if not len(feat_list):
                    feat_list = [feat_dict[cid] for cid in feat_dict.keys()]
            mean_feat = np.mean(np.concatenate(feat_list, axis=0), axis=0)
            mean_feat = mean_feat / np.linalg.norm(mean_feat, ord=2)
            frame_result_data.set_feat(mean_feat)
            global_trail_data[time_stamp] = frame_result_data
        global_result_dic[global_id] = global_trail_data
    return global_result_dic

def get_clustered_world_list(world_list):
    cost_matrix = np.zeros((len(world_list), len(world_list)), dtype=np.float32) + LARGE_COST
    for i in range(len(world_list) - 1):
        for j in range(i+1, len(world_list)):
            cost_matrix[i, j] = np.linalg.norm(world_list[i][:2] - world_list[j][:2], ord=2)
            cost_matrix[j, i] = cost_matrix[i, j]
    cluster_labels = AgglomerativeClustering(n_clusters=2, affinity='precomputed',
                                linkage='average').fit_predict(cost_matrix)
    labels = get_match(cluster_labels)
    res_list = []
    select_list = []
    for l in labels:
        if len(l) > len(select_list):
            select_list = l
    res_list = [world_list[i] for i in select_list]
    return res_list, select_list

def get_global_result(clustered_record_list, track_data_dic, camera_param_dic, sim_dict, len_th):

    global_result_dic = get_average_result(clustered_record_list, track_data_dic, camera_param_dic, sim_dict, len_th)

    return global_result_dic

def global_track_iml(track_data_dic, camera_param_dic, reid_th, len_th):
    """Global track iml."""
    track_list = list(track_data_dic.keys())
    # get sim between each track, including position and reid
    track_sim_list, sim_dict = get_tracks_sim(track_data_dic, camera_param_dic, track_list)

    # merge tracklet by sim
    if len(track_sim_list) == 0:
        return {}, []
    clustered_recordid_list = merge_by_sim(track_sim_list, track_data_dic, track_list, reid_th)

    # average world coordinate from multiple cameras, get map coordinate.
    global_result = get_global_result(clustered_recordid_list, track_data_dic, camera_param_dic, sim_dict, len_th)

    return global_result, clustered_recordid_list
