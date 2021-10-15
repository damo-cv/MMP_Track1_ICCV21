import argparse
import itertools
import json
import logging
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from interp_track1to2 import interp_track1to2

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def remove_1len_track(all_results):
    refined_results = []
    tids = np.unique(all_results[:, 1])
    for tid in tids:
        results = all_results[all_results[:, 1] == tid]
        if results.shape[0] <= 1:
            continue
        refined_results.append(results)
    refined_results = np.concatenate(refined_results, axis=0)
    return refined_results

def track_and_save(calibration_dir, track2_dir, track1_dir, output):
    camera_id = int(track2_dir.stem.split('_')[-1])
    track2_res = np.load(track2_dir)
    #dets: fid, cat, x, y, w, h, score, feat 
    scene_name = track2_dir.parent.name
    scene_name = '_'.join(scene_name.split('_')[:-1])
    calibration_file = calibration_dir / scene_name / 'calibrations.json'
    track1_res = np.load(track1_dir) #fid, tid, Y, X, Z
    res = interp_track1to2(track2_res, track1_res[:, :5], camera_id, calibration_file)
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, res)
    
def track_and_save_star(args):
    track_and_save(*args)

def parse_det(base_dir, middle_file):
    videos = []
    det_base_dir = base_dir / middle_file
    for time_folder in det_base_dir.iterdir():
        for video in time_folder.iterdir():
            if not video.is_dir():
                continue
            for det_file in video.glob('*.npy'):
                videos.append(det_file)
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
    parser.add_argument('--track2-dir',
                        type=Path,
                        required=True,
                        help='track2 res dir')
    parser.add_argument('--track1-dir',
                        type=Path,
                        required=True,
                        help='track1 res dir')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(det):
        return args.output_dir / str(det.relative_to(args.track2_dir))
    
    def get_track1_path(det):
        det_parent = det.relative_to(args.track2_dir).parent
        track1_path = str(det_parent / (det_parent.name + '.npy'))
        return args.track1_dir / track1_path
    
    ress = parse_det(args.track2_dir / args.split, 'res')
    
    tasks = []
    for res in ress:
        output = get_output_path(res)
        track1_res = get_track1_path(res)
        tasks.append((args.base_dir / args.split / 'calibrations', res, track1_res, output))

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
