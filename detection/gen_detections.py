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
import torch
from torchvision.ops import nms
from tqdm import tqdm
import pandas as pd
# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def save_detections(det, output, score_threshold, nms_thresh):
    fids = np.unique(det[:, 0])
    res = []
    for fid in fids:
        sub_det = det[det[:, 0] == fid].copy()
        if score_threshold > 0:
            sub_det = sub_det[sub_det[:, 5] >= score_threshold]
        if nms_thresh >= 0:
            bboxes = sub_det[:, 1:5].copy()
            bboxes[:, 2:4] += bboxes[:, :2]
            nms_keep = nms(torch.from_numpy(bboxes),
                           torch.from_numpy(sub_det[:, 5]),
                           iou_threshold=nms_thresh).numpy()
            sub_det = sub_det[nms_keep]
        res.append(sub_det)
    res = np.vstack(res)
    cat_pad = np.zeros((res.shape[0], 1))
    res = np.hstack((res[:, 0:1], cat_pad, res[:, 1:])) #'image_id', 'category', 'bb_left','bb_top', 'bb_width', 'bb_height', 'conf'
    output.parent.mkdir(exist_ok=True,parents=True)
    det_dfs = pd.DataFrame(res)
    det_dfs.to_csv(str(output), header=False, index=False)

def save_detections_star(args):
    save_detections(*args)

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--detections-file',
                        type=Path,
                        default='results/comp4_det_test_320_False__person.txt',
                        help='results file in txt format')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))

    parser.add_argument('--score-threshold',
                        default=-1,
                        help='threshold to filter detections')

    parser.add_argument('--nms-thresh', type=float, default=-1, help='nms threshold')
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    
    args.score_threshold = (-float('inf') if args.score_threshold == 'none'
                            else float(args.score_threshold))

    def get_name(video_name):
        video_name_parse = video_name.split('/')
        camera_id = video_name_parse[-1].split('_')[-1]
        video_name_parse = video_name_parse[:-1]
        video_name_parse.append(video_name_parse[-1] + '_' + camera_id)
        return '/'.join(video_name_parse)

    def get_fid(video_name):
        video_name_parse = video_name.split('/')
        fid = int(video_name_parse[-1].split('_')[-2])
        return fid
    
    df = pd.read_csv(os.path.join(args.detections_file), delimiter=' ',header=None)
    df.columns = ['video_name', 'score', 'x1', 'y1', 'x2', 'y2']
    df['name'] = df['video_name'].apply(get_name)
    print('finish get name')
    df['fid'] = df['video_name'].apply(get_fid)
    print('finish get fid')
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']
    unique_names = np.unique(df['name'].values)

    tasks = []
    for name in unique_names:
        sub_df = df[df['name'] == name]
        res = sub_df[['fid', 'x1', 'y1', 'w', 'h', 'score']].values
        output = args.output_dir / (name + '.txt')
        output = Path(str(output).replace('images', 'det_video'))    
        tasks.append((res, output, args.score_threshold, args.nms_thresh))
    
    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(save_detections_star, tasks),
                 total=len(tasks),
                 desc='Save Detections'))
    else:
        for task in tqdm(tasks):
            save_detections(*task)
      
    print('Finished')

if __name__ == "__main__":
    main()
