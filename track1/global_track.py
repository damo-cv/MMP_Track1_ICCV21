from io_data import parse_track_data, get_global_results
from utils.utils import parse_camera_param
from track_imp import global_track_iml

def global_track(track_data, camera_param_file, reid_th, len_th):
    """track_data: {camera_id: dets}"""
    #fid, tid, x, y, w, h, score, feat ###no category
    track_data_dict = {}
    for camera_id in track_data.keys():
        dets = track_data[camera_id]
        t_dict = parse_track_data(camera_id, dets)
        track_data_dict.update(t_dict)
    camera_param_dict = parse_camera_param(camera_param_file)
    global_result_dict, clustered_recordid_list = global_track_iml(track_data_dict, camera_param_dict, reid_th, len_th)
    camera_ids = [cid for cid in track_data.keys()]
    out_res = get_global_results(global_result_dict, camera_ids)
    return out_res
