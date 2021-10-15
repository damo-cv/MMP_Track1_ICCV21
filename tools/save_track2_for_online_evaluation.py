import argparse
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

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_FID_DICT = {'cafe_shop_1': 4855, 'cafe_shop_0': 4341, 'retail_0': 6556, 'retail_7': 2536, 
    'office_1': 4564, 'retail_6': 4891, 'office_0': 4119, 'retail_1': 5806, 'lobby_3': 7600,
    'lobby_1': 4693, 'industry_safety_2': 5709, 'industry_safety_3': 4429, 'industry_safety_4': 3408, 
    'lobby_0': 5523, 'cafe_shop_2': 4654, 'retail_4': 4464, 'office_2': 5127, 'retail_3': 2404, 
    'retail_2': 6079, 'retail_5': 4894, 'office_3': 4808, 'industry_safety_1': 4942, 'lobby_2': 10421, 
    'industry_safety_0': 5767}
CAMERA_NUM_DICT = {'cafe_shop_1': 4, 'cafe_shop_0': 4, 'retail_0': 6, 'retail_7': 6, 'office_1': 5, 
    'retail_6': 6, 'office_0': 5, 'retail_1': 6, 'lobby_1': 4, 'industry_safety_2': 4, 'industry_safety_3': 4, 
    'industry_safety_4': 4, 'lobby_0': 4, 'cafe_shop_2': 4, 'retail_4': 6, 'office_2': 5, 'retail_3': 6, 
    'retail_2': 6, 'retail_5': 6, 'office_3': 5, 'industry_safety_1': 4, 'lobby_2': 4, 'lobby_3': 4, 'industry_safety_0': 4}

def save(res_path, output):
    output.mkdir(exist_ok=True, parents=True)
    ress = np.load(res_path)
    #['image_id', 'tid','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'] features 512
    file_name = res_path.stem
    camera_id = int(file_name.split('_')[-1])
    fid2res = defaultdict(list)
    for res in ress:
        fid = int(res[0])
        fid2res[fid].append(res[1:6].tolist())
    scene_name = res_path.parent.name
    for fid in range(MAX_FID_DICT[scene_name] + 1):
        out_path = output / ('rgb_%05d_%d.json' % (fid, camera_id))
        out = {}
        if fid in fid2res:
            for fr in fid2res[fid]:
                tid = str(int(fr[0]))
                out[tid] = [fr[1], fr[2], fr[1]+fr[3], fr[2]+fr[4]] #xyxy
        with open(str(out_path), 'w') as f:
            json.dump(out, f)
        
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
        return args.output_dir / res.relative_to(args.res_dir).parent

    ress = parse_res(args.res_dir / 'test', 'res')
    
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
