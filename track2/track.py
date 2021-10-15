import argparse
import itertools
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import fair_app
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def track_and_save(det_path, output, sort_kwargs):
    if os.path.exists(str(output)):
        return
    detections = np.load(det_path)
   
    #['image_id', 'category','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'] features 512
    
    categories = np.unique(detections[:, 1].astype(int))

    id_gen = itertools.count(1)
    unique_track_ids = defaultdict(lambda: next(id_gen))

    all_results = []
    #th = 32 ** 2
    for category in categories:
        mask = detections[:, 1].astype(int) == category
        det = detections[mask]
        
        results = fair_app.run(det, **sort_kwargs) #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, feature
        if len(results) == 0:
            continue
        results = np.array(results).reshape(len(results), -1)
        track_ids = np.array([unique_track_ids[(x, category)] for x in results[:, 1]])
        category_res = np.ones((results.shape[0]), dtype=np.float32) * category
        results_save = np.hstack((results[:, 0:1], track_ids[:, np.newaxis], results[:, 2:7], category_res[:, np.newaxis]))
        results_save = np.hstack((results_save, results[:, 7:]))
        all_results.append(results_save)

    all_results = np.concatenate(all_results, axis=0) #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, all_results)


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
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='Detection directory with features')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    #deepsort args
    parser.add_argument('--min_confidence',
                        default=0.5,
                        type=float,
                        help='Float or "none".')
    parser.add_argument('--nms_max_overlap', type=float, default=-1)
    parser.add_argument('--max_cosine_distance', default=0.4, type=float)
    
    parser.add_argument('--workers', default=8, type=int)
    
    args = parser.parse_args()
    
    def get_output_path(det):
        return args.output_dir / str(det.relative_to(args.detections_dir)).replace('det_video', 'res')

    dets = parse_det(args.detections_dir / args.split, 'det_video')
    
    tasks = []
    for det in dets:
        output = get_output_path(det)
        tasks.append((det, output, {
                          'min_confidence': args.min_confidence,
                          'nms_max_overlap': args.nms_max_overlap,
                          'max_cosine_distance': args.max_cosine_distance
                      }))

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
