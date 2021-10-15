import numpy as np
from collections import defaultdict
from utils.utils import parse_camera_param
import pandas as pd

def global2pixel(person_coords, camera_id, camera_param_dict):
    # det : X Y Z
    world_coord = person_coords / camera_param_dict['discretization_factor'] + camera_param_dict['min_volume']
    trans_coord = world_coord - camera_param_dict[camera_id]['Translation']
    uvw = np.linalg.inv(camera_param_dict[camera_id]['Rotation']) @ trans_coord.transpose(1, 0)
    uvw = uvw.transpose(1, 0)
    pixel_coords = uvw / camera_param_dict[camera_id]['FInv'] / uvw[:, 2:3] + camera_param_dict[camera_id]['C']
    return pixel_coords[:, :2]

def interp_track1to2(track2_res, track1_res, camera_id, camera_param_file):
    #track1_res: fid, tid, Y, X, Z, feat
    #track2_res: fid, tid, x, y, w, h, score, cat, feat 
    camera_param_dict = parse_camera_param(camera_param_file)
    data_for_projection = np.concatenate((track1_res[:, 3:4], track1_res[:, 2:3], track1_res[:, 4:5]), axis=1)
    pixel_coord = global2pixel(data_for_projection, camera_id, camera_param_dict)
    track1_pixel_res = np.hstack((track1_res[:, :2], pixel_coord))# fid, tid, x, y
    traj_df = pd.DataFrame(data=track2_res[:, :7], columns=['frame', 'trkid', 'x', 'y', 'w', 'h', 'score'])
    traj_df['cx'] = traj_df['x'] + 0.5 * traj_df['w']
    traj_df['cy'] = traj_df['y'] + 0.5 * traj_df['h']
    reixed_traj_df = traj_df.set_index('trkid')
    full_traj_dfs = []
    traj_start_ends = traj_df.groupby('trkid')['frame'].agg(['min', 'max'])
    for ped_id, (traj_start, traj_end) in traj_start_ends.iterrows():
        if ped_id != -1:
            data = np.hstack((track1_pixel_res[track1_pixel_res[:, 1] == ped_id, 0:1], track1_pixel_res[track1_pixel_res[:, 1] == ped_id, 2:4]))
            full_traj_df = pd.DataFrame(data=data, columns=['frame', 'mx', 'my'])
            partial_traj_df = reixed_traj_df.loc[[ped_id]].reset_index()
        
            #partial_traj_df['flag'] = 1
            full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'cx', 'cy', 'w', 'h', 'score']],
                                    how='left', on='frame')
            #full_traj_df['flag'].fillna(0, inplace=True)
            full_traj_df['cx'].fillna(full_traj_df['mx'], inplace=True)
            full_traj_df['cy'].fillna(full_traj_df['my'], inplace=True)
            full_traj_df = full_traj_df.sort_values(by='frame').interpolate()
            full_traj_df.fillna(method='ffill', axis=0, inplace=True)
            full_traj_df.fillna(method='bfill', axis=0, inplace=True) 
            full_traj_dfs.append(full_traj_df)
    
    traj_df = pd.concat(full_traj_dfs)
    traj_df['x'] = traj_df['cx'] - 0.5 * traj_df['w']
    traj_df['y'] = traj_df['cy'] - 0.5 * traj_df['h']
    res = traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score']].to_numpy()
    return res 

    
     
     
