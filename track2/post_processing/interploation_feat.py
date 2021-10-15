import numpy as np
import pandas as pd

def interpolate_traj(trks, mark_interpolation=True, drop_len=1, interp_len=100):
    '''
    trks: 2d np array of MOT format.
    '''
    print('performing interpolation.')
    feat_dim = trks[:, 8:].shape[1]
    #['image_id', 'tid','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'category']
    traj_df = pd.DataFrame(data=trks[:, :8], columns=['frame', 'trkid', 'x', 'y', 'w', 'h', 'score', 'category'])

    traj_df['cx'] = traj_df['x'] + 0.5 * traj_df['w']
    traj_df['cy'] = traj_df['y'] + 0.5 * traj_df['h']
    # Build sub_dfs with full trajectories across all missing frames
    reixed_traj_df = traj_df.set_index('trkid')
    full_traj_dfs = []
    traj_start_ends = traj_df.groupby('trkid')['frame'].agg(['min', 'max'])
    for ped_id, (traj_start, traj_end) in traj_start_ends.iterrows():
        if ped_id != -1:
            full_traj_df = pd.DataFrame(data=np.arange(traj_start, traj_end + 1), columns=['frame'])
            partial_traj_df = reixed_traj_df.loc[[ped_id]].reset_index()
            if mark_interpolation:
                partial_traj_df['flag'] = 1

                # Interpolate bb centers, heights and widths
                full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'cx', 'cy', 'h', 'w', 'score', 'category', 'flag']],
                                    how='left', on='frame')
                full_traj_df['flag'].fillna(0, inplace=True)
            else:
                full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'cx', 'cy', 'h', 'w', 'score', 'category']],
                                    how='left', on='frame')
            full_traj_df = full_traj_df.sort_values(by='frame').interpolate()
            full_traj_dfs.append(full_traj_df)

    traj_df = pd.concat(full_traj_dfs)
    # Recompute bb coords based on the interpolated centers, heights and widths
    traj_df['x'] = traj_df['cx'] - 0.5 * traj_df['w']
    traj_df['y'] = traj_df['cy'] - 0.5 * traj_df['h']
    if mark_interpolation:
        res = traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score', 'category', 'flag']].to_numpy()
        start_row = 0
        remove_ids = []
        feat_pad = np.zeros((res.shape[0], feat_dim), dtype=res.dtype)
        res = np.hstack((res, feat_pad))
        original_trks = trks.copy()
        trks_df = pd.DataFrame(data=original_trks, columns=['frame', 'trkid', 'x', 'y', 'w', 'h', 'score', 'category'] + [f'feat{i}' for i in range(original_trks[:, 8:].shape[1])])
        trks_df = trks_df.sort_values(by=['trkid', 'frame'])
        trks_np = trks_df.values
        res[res[:, 8] == 1, 9:] = trks_np[:, 8:]
        for i in range(res.shape[0]):
            if res[i, 8] == 1:
                cnt = i - start_row - 1
                if cnt > interp_len:
                    remove_ids.append(np.arange(start_row+1, i))
                else:
                    start_feat = res[start_row, 9:]
                    end_feat = res[i, 9:]
                    for idx in range(start_row+1, i):
                        feat1 = (i - idx + 1) / (cnt + 2) * start_feat
                        feat2 = (idx - start_row) / (cnt + 2) * end_feat
                        feat = feat1 + feat2
                        feat = feat / np.linalg.norm(feat, ord=2)
                        res[idx, 9:] = feat
                start_row = i
        if len(remove_ids):
            remove_ids = np.hstack(remove_ids)
        remain_ids = set(list(range(res.shape[0]))) - set(remove_ids)
        remain_ids = np.array(list(remain_ids)).astype(int)
        res = res[remain_ids, :]
        return res
    else:
        return traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score', 'category']].to_numpy()