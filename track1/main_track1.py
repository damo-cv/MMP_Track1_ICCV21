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

def track_and_save(calibration_dir, detection_dir, output, reid_th, len_th):
    output_npy = output / (output.name + '.npy')
    output_txt = output / (output.name + '.txt')
    if os.path.exists(str(output_txt)) and os.path.exists(str(output_npy)):
        return
    det_files = detection_dir.glob('*.npy')
    track_data = {}
    for det_f in det_files:
        camera_id = int(det_f.stem.split('_')[-1])
        dets = np.load(det_f)
        dets = np.concatenate((dets[:, :7], dets[:, 8:]), axis=1)
        track_data[camera_id] = dets 
    scene_name = detection_dir.name
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
    parser.add_argument('--res-dir',
                        type=Path,
                        required=True,
                        help='Track2 results directory with features')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--reid_th', default=0.3, type=float)
    parser.add_argument('--len_th', default=400, type=int) 
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    
    def get_output_path(det):
        return args.output_dir / str(det.relative_to(args.res_dir))
    
    dets = parse_det(args.res_dir / args.split, 'res')
    
    tasks = []
    for det in dets:
        output = get_output_path(det)
        tasks.append((args.base_dir / args.split / 'calibrations', det, output, args.reid_th, args.len_th))

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
