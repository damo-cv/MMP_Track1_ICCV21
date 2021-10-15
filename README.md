# Tracking Code for the winner of track1 in MMP-Trakcing challenge

This repository contains our tracking code for the Multi-camera Multiple People Tracking (MMP-Tracking) Challenge at ICCV 2021 Workshop.

## 1. Environment setup

This tracking code has been tested on Python 3.7.6, Pytorch 1.5.1, CUDA 10.1, please install related libraries before running this code:

```bash
pip install -r requirements.txt
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. Detection

We provide the detection results on the test split in [yolox_baseline.zip](detection/yolox_baseline.zip) and [yolox_finetune.zip](detection/yolox_finetune.zip). They should be extracted first.

## 3. Tracking

Run the following script to reproduce our tracking results.
```bash
sh auto_run.sh
```

## 4. Acknowledgement

We use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to train our detection model. 
Some code from [deep_sort](https://github.com/nwojke/deep_sort), [FairMOT](https://github.com/ifzhang/FairMOT), and [tao](https://github.com/TAO-Dataset/tao).
