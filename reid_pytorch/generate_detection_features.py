import argparse
import sys
import os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from parallel.fixed_gpu_pool import FixedGpuPool
from PIL import Image
from numpy import pad
from inference_reid import ReID_Inference
import torchvision.transforms as T
# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

DET_COL_NAMES = ('image_id', 'category', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df, pad_ = True, pad_mode = 'mean',  mean=[123.675,116.280,103.530], 
                 std=[57.0,57.0,57.0], output_size = (128, 384)):
        self.det_df = det_df
        self.output_size = output_size
        self.pad = pad_
        self.curr_image_path = None
        self.curr_img = None
        self.pad_mode = pad_mode
        mean = [k/255. for k in mean]
        std = [k/255. for k in std]
        trans_list = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        self.transform = T.Compose(trans_list)
    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]
        # Load this bounding box' frame img, in case we haven't done it yet
        if row['image_path'] != self.curr_image_path:
            self.curr_img = cv2.imread(row['image_path'])
            if self.curr_img is None:
                self.curr_img = cv2.imread(row['image_path'].replace('jpg', 'jpeg'))
            self.curr_image_path = row['image_path']

        frame_img = self.curr_img
        frame_height = frame_img.shape[0]
        frame_width = frame_img.shape[1]
        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['bb_top'])): int(max(0, row['bb_bot'])),
                   int(max(0, row['bb_left'])): int(max(0, row['bb_right']))]
        if self.pad:
            x_height_pad = np.abs(row['bb_top'] - max(row['bb_top'], 0)).astype(int)
            y_height_pad = np.abs(row['bb_bot'] - min(row['bb_bot'], frame_height)).astype(int)

            x_width_pad = np.abs(row['bb_left'] - max(row['bb_left'], 0)).astype(int)
            y_width_pad = np.abs(row['bb_right'] - min(row['bb_right'], frame_width)).astype(int)

            bb_img = pad(bb_img, ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)), mode=self.pad_mode)
        
        bb_img = cv2.resize(bb_img, self.output_size)
        
        bb_img = bb_img[:, :, ::-1]
        bb_img = Image.fromarray(bb_img)
        bb_img = self.transform(bb_img)
        
        return bb_img


def init_model(init_args, context):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(context['gpu'])

def infer(kwargs, context):
    image_path = kwargs['image_path']
    det_path = kwargs['det_path']
    output = kwargs['output']
    model_file = kwargs['model_file']
    import os
    if os.path.exists(str(output)):
        return
    model = ReID_Inference(model_file, '0')
    det_df = pd.read_csv(det_path, header=None)
    video_name = det_path.stem
    camera_id = int(str(video_name).split('_')[-1])
    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES
    
    det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
    det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values
    func = lambda x: str(image_path / ('rgb_%05d_%d.jpg' % (int(x), camera_id)))
    det_df['image_path'] = det_df['image_id'].apply(func)
    conds = (det_df['bb_width'] > 1) & (det_df['bb_height'] > 1)
    conds = conds & (det_df['bb_right'] > 1) & (det_df['bb_bot'] > 1)
    sample_img = cv2.imread(det_df['image_path'][0])
    conds = conds & (det_df['bb_left'] < sample_img.shape[1]) & (det_df['bb_top'] <  sample_img.shape[0])
    
    det_df = det_df[conds]

    bbox_dataset = BoundingBoxDataset(det_df)
    bbox_loader = DataLoader(bbox_dataset, batch_size=300, pin_memory=False, num_workers=16)
    #Feed all bboxes to the CNN to obtain node and reid embeddings
    
    print(f"Computing embeddings for {len(bbox_dataset)} detections")
    #start_time = time.time()
    features = []
    with torch.no_grad():
        for bboxes in bbox_loader:
            feature = model(bboxes.cuda())
            features.append(feature.cpu().numpy())
            #print(time.time() - start_time, image_path)
    
    features = np.concatenate(features, axis=0)
    data = np.hstack((det_df[['image_id', 'category','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']].values.astype(np.float32),features))

    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output), data)

def parse_det(base_dir, middle_file):
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
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='detections directory with .txt files')
    parser.add_argument(
        '--image-dir',
        type=Path,
        required=True,
        help=('image directory'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory')) 
    parser.add_argument('--model-file',
                        type=str,
                        default='resnet50_ibn_a_model_60.pth',
                        help='name of the reid model file')
    parser.add_argument('--gpus', default=[0], nargs='+', type=int)
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    dets = parse_det(args.detections_dir / args.split, 'det_video')    
    
    def get_output_path(det):
        output = args.output_dir / str(det.relative_to(args.detections_dir)).replace('.txt', '.npy')
        return output
    def get_image_path(det):
        return args.image_dir / str(det.relative_to(args.detections_dir).parent).replace('det_video', 'images')
   
    infer_tasks = []
    for det in dets:
        output = get_output_path(det)
        image_path = get_image_path(det)
        infer_tasks.append({'image_path': image_path, 'det_path': det, 'output': output,
                            'model_file': args.model_file})
    init_args = []
    if len(args.gpus) == 1:
        context = {'gpu': args.gpus[0]}
        init_model(init_args, context)
        for task in tqdm(infer_tasks,
                         mininterval=1,
                         desc='Running generation',
                         dynamic_ncols=True):
            infer(task, context)
    else:
        pool = FixedGpuPool(
            args.gpus, initializer=init_model, initargs=init_args)
        list(
            tqdm(pool.imap_unordered(infer, infer_tasks),
                 total=len(infer_tasks),
                 mininterval=10,
                 desc='Running generation',
                 dynamic_ncols=True))
    print(f'Finished')

if __name__ == "__main__":
    main()
