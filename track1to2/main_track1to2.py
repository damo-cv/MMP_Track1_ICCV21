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
from track1to2_track import track1to2_track
from interp_track1to2 import interp_track1to2
# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def track_and_save(calibration_dir, detection_dir, track1_dir, output, min_confidence, cos_th, pos_th):
    camera_id = int(detection_dir.stem.split('_')[-1])
    dets = np.load(detection_dir)
    dets = dets[dets[:, 6] >= min_confidence]
    #dets: fid, cat, x, y, w, h, score, feat 
    scene_name = detection_dir.parent.name
    scene_name = '_'.join(scene_name.split('_')[:-1])
    calibration_file = calibration_dir / scene_name / 'calibrations.json'

    track1_res = np.load(track1_dir) #fid, tid, Y, X, Z
    #fid, tid, cat, x, y, w, h, score, feat
    track1_res = np.hstack((track1_res[:, :5], track1_res[:, 5+camera_id*512: 5+camera_id*512+512]))    
    res = track1to2_track(dets, track1_res, camera_id, calibration_file, cos_th, pos_th, cos_first=True)

    res = np.concatenate((res[:, 0:2], res[:, 3:8], res[:, 2:3], res[:, 8:]), axis=1)

    final_res = interp_track1to2(res, track1_res[:, :5], camera_id, calibration_file)

    output.parent.mkdir(exist_ok=True, parents=True)
    
    np.save(output, final_res)
    
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
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='detection directory with features')
    parser.add_argument('--track1-dir',
                        type=Path,
                        required=True,
                        help='track1 res dir')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--min_confidence',
                        default=0.0,
                        type=float,
                        help='Float or "none".') 
    parser.add_argument('--cos_th', default=0.3, type=float)
    parser.add_argument('--pos_th', default=10, type=float)
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(det):
        return args.output_dir / str(det.relative_to(args.detections_dir)).replace('det_video', 'res')
    
    def get_track1_path(det):
        det_parent = det.relative_to(args.detections_dir).parent
        track1_path = str(det_parent / (det_parent.name + '.npy')).replace('det_video', 'res')
        return args.track1_dir / track1_path
    
    dets = parse_det(args.detections_dir / args.split, 'det_video')
    
    tasks = []
    for det in dets:
        output = get_output_path(det)
        track1_res = get_track1_path(det)
        tasks.append((args.base_dir / args.split / 'calibrations', det, track1_res, output, args.min_confidence, args.cos_th, args.pos_th))

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
