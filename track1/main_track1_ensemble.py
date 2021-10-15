import argparse
import itertools
import json
import logging
import pickle
import sys
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from global_track import global_track

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def track_and_save(calibration_dir, detection_dir1, detection_dir2, output, reid_th, len_th):
    output_npy = output / (output.name + '.npy')
    output_txt = output / (output.name + '.txt')
    if os.path.exists(str(output_txt)) and os.path.exists(str(output_npy)):
        return
    det_files1 = detection_dir1.glob('*.npy')
    det_files2 = detection_dir2.glob('*.npy')
    track_data = {}
    for det_f1, det_f2 in zip(det_files1, det_files2):
        camera_id = int(det_f1.stem.split('_')[-1])
        camera_id2 = int(det_f2.stem.split('_')[-1])
        assert camera_id == camera_id2
        dets1 = np.load(det_f1)
        dets2 = np.load(det_f2)
        det1_ids = np.unique(dets1[:, 1])
        dets2[:, 1] += (np.max(det1_ids)+1)
        dets = np.vstack((dets1, dets2))
        dets = np.concatenate((dets[:, :7], dets[:, 8:]), axis=1)
        track_data[camera_id] = dets 
    scene_name = detection_dir1.name
    scene_name = '_'.join(scene_name.split('_')[:-1])
    calibration_file = calibration_dir / scene_name / 'calibrations.json'
    res = global_track(track_data, calibration_file, reid_th, len_th) # fid tid Y X Z feat
    
    output.mkdir(exist_ok=True, parents=True)

    np.save(output_npy, res)
   
    res_dfs = pd.DataFrame(res[:, : 5+512])
    res_dfs.to_csv(output_txt, header=False, index=False)

def track_and_save_star(args):
    track_and_save(*args)

def parse_det(base_dir, middle_file):
    videos = []
    det_base_dir = base_dir / middle_file
    for time_folder in det_base_dir.iterdir():
        for video in time_folder.iterdir():
            if not video.is_dir():
                continue
            videos.append(video)
    return videos

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split',
                        type=str,
                        required=True,
                        default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--base-dir',
                        type=Path,
                        required=True,
                        help='data base dir, for locate calibration file')
    parser.add_argument('--res-dir1',
                        type=Path,
                        required=True,
                        help='first track2 result directory with features')
    parser.add_argument('--res-dir2',
                        type=Path,
                        required=True,
                        help='second track2 result directory with features')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--reid_th', default=0.3, type=float)
    parser.add_argument('--len_th', default=400, type=int) 
    parser.add_argument('--workers', default=24, type=int)

    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(det):
        return args.output_dir / str(det.relative_to(args.res_dir1))
    
    dets1 = parse_det(args.res_dir1 / args.split, 'res')
    dets2 = parse_det(args.res_dir2 / args.split, 'res')
    tasks = []
    for det1, det2 in zip(dets1, dets2):
        output = get_output_path(det1)
        tasks.append((args.base_dir / args.split / 'calibrations', det1, det2, output, args.reid_th, args.len_th))

    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(track_and_save_star, tasks),
                 total=len(tasks),
                 desc='Tracking'))
    else:
        for task in tqdm(tasks):
            track_and_save(*task)
    print(f'Finished')
    
if __name__ == "__main__":
    main()
