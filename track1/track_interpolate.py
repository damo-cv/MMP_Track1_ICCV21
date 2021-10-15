import numpy as np
import pandas as pd

def interpolate_traj(trks, threshold, mark_interpolation=False, drop_len=1):
    trks = trks[np.argsort(trks[:, 1])]
    feat_dim = trks[:, 5:].shape[1]
    traj_df = pd.DataFrame(data=trks[:, :5], columns=['frame', 'trkid', 'y', 'x', 'z'])
    
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
                                    partial_traj_df[['trkid', 'frame', 'y', 'x', 'z', 'flag']],
                                    how='left', on='frame')
                full_traj_df['flag'].fillna(0, inplace=True)
            else:
                full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'y', 'x', 'z']],
                                    how='left', on='frame')
            full_traj_df = full_traj_df.sort_values(by='frame').interpolate()
            full_traj_dfs.append(full_traj_df)

    traj_df = pd.concat(full_traj_dfs)
    # Recompute bb coords based on the interpolated centers, heights and widths
    if mark_interpolation:
        res = traj_df[['frame', 'trkid', 'y', 'x', 'z', 'flag']].to_numpy()
        start_row = 0
        remove_ids = []
        feat_pad = np.zeros((res.shape[0], feat_dim), dtype=res.dtype)
        res = np.hstack((res, feat_pad))
        original_trks = trks.copy()
        trks_df = pd.DataFrame(data=original_trks, columns=['frame', 'trkid', 'y', 'x', 'z'] + [f'feat{i}' for i in range(original_trks[:, 5:].shape[1])])
        trks_df = trks_df.sort_values(by=['trkid', 'frame'])
        trks_np = trks_df.values
        res[res[:, 5] == 1, 6:] = trks_np[:, 5:]
        for i in range(res.shape[0]):
            if res[i, 5] == 1:
                cnt = i - start_row - 1
                if cnt > threshold:
                    remove_ids.append(np.arange(start_row+1, i))
                else:
                    start_feat = res[start_row, 6:]
                    end_feat = res[i, 6:]
                    for idx in range(start_row+1, i):
                        feat1 = (i - idx + 1) / (cnt + 2) * start_feat
                        feat2 = (idx - start_row) / (cnt + 2) * end_feat
                        feat = feat1 + feat2
                        feat = feat / np.linalg.norm(feat, ord=2)
                        res[idx, 6:] = feat
                start_row = i
        if len(remove_ids):
            remove_ids = np.hstack(remove_ids)
        remain_ids = set(list(range(res.shape[0]))) - set(remove_ids)
        remain_ids = np.array(list(remain_ids)).astype(int)
        res = res[remain_ids, :]
        return np.hstack((res[:, :5], res[:, 6:]))
    else:
        return traj_df[['frame', 'trkid', 'y', 'x', 'z']].to_numpy()
