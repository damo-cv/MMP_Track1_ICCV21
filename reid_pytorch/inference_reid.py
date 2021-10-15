# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import sys

sys.path.append('.')
import torch
from torch.backends import cudnn
from modeling import build_model
import torchvision.transforms as T

class ReID_Inference:

    def __init__(self,model_dir,gpus):
    
        # useful setting
        model_name = 'resnet50_ibn_a'
        self.w = 128
        self.h = 384

        # cuda
        if gpus == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        cudnn.benchmark = True

        # model
        self.model = build_model(model_name)
        self.model.load_param(model_dir,'self')
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to('cuda')
        self.model.eval()

        # preprocessing
        mean = [k/255. for k in [123.675,116.280,103.530]]
        std = [k/255. for k in [57.0,57.0,57.0]]
        trans_list = [T.ToTensor(),T.Normalize(mean=mean, std=std)]
        self.transform = T.Compose(trans_list)

    def __call__(self, imgs):
        feat = self.model(imgs)
        return feat