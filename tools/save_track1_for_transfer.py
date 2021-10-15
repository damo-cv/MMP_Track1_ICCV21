import argparse
import itertools
import json
import logging
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from src import fair_app
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def save(res_path, output):
    output.parent.mkdir(exist_ok=True, parents=True)
    ress = np.loadtxt(res_path, delimiter=',')
    ress = ress[:, :5]
    res_dfs = pd.DataFrame(ress)
    res_dfs.to_csv(output, header=False, index=False)

def save_star(args):
    save(*args)

def parse_res(base_dir, middle_file):
    videos = []
    det_base_dir = base_dir / middle_file
    for time_folder in det_base_dir.iterdir():
        for video in time_folder.iterdir():
            if not video.is_dir():
                continue
            for det_file in video.glob('*.txt'):
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
    parser.add_argument('--res-dir',
                        type=Path,
                        required=True,
                        help='Results directory with features')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(res):
        return args.output_dir / res.relative_to(args.res_dir)

    ress = parse_res(args.res_dir / args.split, 'res')
    
    tasks = []
    for res in ress:
        output = get_output_path(res)
        tasks.append((res, output))

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
