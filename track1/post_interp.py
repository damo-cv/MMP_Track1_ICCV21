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
from track_interpolate import interpolate_traj

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = ['FrameId', 'Id', 'Y', 'X']

def save(res, output, threshold):

    results = np.load(res)

    results = interpolate_traj(results, threshold, mark_interpolation=True)
    
    output.parent.mkdir(exist_ok=True, parents=True)
    
    res_dfs = pd.DataFrame(results[:, :5])
    res_dfs.to_csv(str(output).replace('npy', 'txt'), header=False, index=False)
    np.save(output, results)

def save_star(args):
    save(*args)
    
def parse_res(base_dir, middle_file):
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
                        default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument(
        '--res-dir',
        type=Path,
        required=True,
        help=('track1 results directory'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--threshold',
                        type=float, default=1000)
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    def get_output_path(res):
        return args.output_dir / res.relative_to(args.res_dir)

    ress = parse_res(args.res_dir / args.split, 'res')
    
    tasks = []
    for res in ress:
        output = get_output_path(res)
        tasks.append((res, output, args.threshold))

    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(save_star, tasks),
                 total=len(tasks),
                 desc='Tracking'))
    else:
        for task in tqdm(tasks):
            save(*task)
    print(f'Finished')
 
if __name__ == "__main__":
    main()
