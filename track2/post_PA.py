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
import torch
import os
from tqdm import tqdm
from post_processing.post_association_by_cluster import associate as associate_cluster
# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def save(res_path, output, threshold):
    if os.path.exists(str(output)):
        return
    output.parent.mkdir(exist_ok=True, parents=True)
    results = np.load(res_path)
    #['image_id', 'tid','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'category'] features 512
    loaded_trk_ids = np.unique(results[:, 1])
    file_name = res_path.stem
    time_name = str(res_path).split('/')[-3]
    split = str(res_path).split('/')[-5]
    seq_name = '_'.join([split, time_name, file_name]) 
    #association    
    results = associate_cluster(results, threshold, 8, identity=seq_name)

    trk_ids = np.unique(results[:, 1])
    print('after association, merging ', len(loaded_trk_ids) - len(trk_ids), ' tracks')
 
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
                        required=True,
                        default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--res-dir',
                        type=Path,
                        required=True,
                        help='Results directory of track2')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    ress = parse_res(args.res_dir / args.split, 'res')
        
    def get_output_path(res):
        return args.output_dir / str(res.relative_to(args.res_dir))    

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
