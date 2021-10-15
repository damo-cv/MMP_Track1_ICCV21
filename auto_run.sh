export PYTHONPATH=$PWD
split=test
MMP_BASE_DIR=/path/to/MMP/
#1. generate detection txt files from the output of the detection models
#we have two detection models, yolox and yolox_finetune

python detection/gen_detections.py --split $split \
    --detections-file ./detection/yolox_baseline/comp4_det_test_320_False__person.txt \
    --output-dir ./detection_results/yolox/ \
    --workers 24

python detection/gen_detections.py --split $split \
    --detections-file ./detection/yolox_finetune/comp4_det_test_320_False__person.txt \
    --output-dir ./detection_results/yolox_finetune/ \
    --workers 24

#2. generate features using the reid model

#Run the following code for multiple GPU inference
export PYTHONOPTIMIZE=1

python reid_pytorch/generate_detection_features.py --split $split \
    --detections-dir ./detection_results/yolox/ \
    --image-dir ${MMP_BASE_DIR} \
    --output-dir ./features/yolox_mmp_r50ibna/ \
    --model-file ./reid_pytorch/resnet50_ibn_a_model_60.pth \
    --gpus 0 1 2 3 4 5 6 7

python reid_pytorch/generate_detection_features.py --split $split \
    --detections-dir ./detection_results/yolox_finetune/ \
    --image-dir ${MMP_BASE_DIR} \
    --output-dir ./features/yolox_finetune_mmp_r50ibna/ \
    --model-file ./reid_pytorch/resnet50_ibn_a_model_60.pth \
    --gpus 0 1 2 3 4 5 6 7
#3. get track2 results using deepsort

python track2/track.py --split $split \
    --detections-dir ./features/yolox_mmp_r50ibna/ \
    --output-dir ./track2_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65/ \
    --min_confidence 0.5 \
    --nms_max_overlap 0.65 \
    --max_cosine_distance 0.4 \
    --workers 24

python track2/track.py --split $split \
    --detections-dir ./features/yolox_finetune_mmp_r50ibna/ \
    --output-dir ./track2_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65/ \
    --min_confidence 0.3 \
    --nms_max_overlap 0.65 \
    --max_cosine_distance 0.4 \
    --workers 24

#4. post associate the tracks. we use the mothd "Post Association" since CVPR20-MOTS challenge, 
#and also apply to ECCV20-TAO and CVPR21-RobMOTS. This method performs tracklet-level merging. 
#You are refered to part 2.3 from "https://arxiv.org/abs/2101.08040" for more details.

python track2/post_PA.py --split $split \
    --res-dir ./track2_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65/ \
    --output-dir ./track2_PA_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65_0.4clsPA/ \
    --threshold 0.4 \
    --workers 24

python track2/post_PA.py --split $split \
    --res-dir ./track2_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65/ \
    --output-dir ./track2_PA_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA/ \
    --threshold 0.5 \
    --workers 24

#5. Interplate absent targets. The detection model is not perfect, we can interplate absent targets in each track.

python track2/post_interp.py --split $split \
    --res-dir ./track2_PA_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65_0.4clsPA/ \
    --output-dir ./track2_interp_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65_0.4clsPA_itp50/ \
    --threshold 50 \
    --workers 24

python track2/post_interp.py --split $split \
    --res-dir ./track2_PA_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA/ \
    --output-dir ./track2_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp50/ \
    --threshold 50 \
    --workers 24

#6. We get the track1 results use above two track2 results

python track1/main_track1_ensemble.py --split $split \
    --base-dir ${MMP_BASE_DIR} \
    --res-dir1 ./track2_interp_results/yolox_mmp_r50ibna_th5_cos0.4_nms0.65_0.4clsPA_itp50/ \
    --res-dir2 ./track2_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp50/ \
    --output-dir ./track1_results/yolox_yolox_finetune_mmp_r50ibna_itp50_track1_ensemble_reid0.3_l400 \
    --reid_th 0.3 \
    --len_th 400

#7. We perform track1 post processing, including removing duplicate tracks by 3d nms, merging overlaping tracks and interplating tracks
#Note that the merge operation has nearly no effect on our final results, but we forget to remove to in our final submission.
#So we keep it in this code.

python track1/post_processing.py --split $split \
    --res-dir ./track1_results/yolox_yolox_finetune_mmp_r50ibna_itp50_track1_ensemble_reid0.3_l400 \
    --output-dir ./track1_interp_results/yolox_yolox_finetune_mmp_r50ibna_itp50_track1_ensemble_reid0.3_l400_30_ios0.5_mg30_itp1000 \
    --nms_dist_th 30 \
    --nms_ios_th 0.5 \
    --merge_threshold 30 \
    --interp_threshold 1000 \
    --workers 24

#This is our final track1 results. we can use tools/save_track1_for_online_evaluation.py to save it in original format.

#python tools/save_track1_for_online_evaluation.py \
#    --res-dir ./track1_interp_results/yolox_yolox_finetune_mmp_r50ibna_itp50_track1_ensemble_reid0.3_l400_30_ios0.5_mg30_itp1000 \
#    --output-dir ./track1_online/yolox_yolox_finetune_mmp_r50ibna_itp50_track1_ensemble_reid0.3_l400_30_ios0.5_mg30_itp1000 \
#    --workers 24

#For track2, we match the detections in each frame to the track1 results and interpolate absent targets using track1 results as reference.
#We only use single yolox_finetune model to obtain the track2 resutls.

#8. get track2 interpolate results of yolox_finetune. Actually, we do not interpolate track2 here, just keep the data format.

python track2/post_interp.py --split $split \
    --res-dir ./track2_PA_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA/ \
    --output-dir ./track2_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0/ \
    --threshold 0 \
    --workers 24

#9. get track1 results

python track1/main_track1.py --split $split \
    --base-dir ${MMP_BASE_DIR} \
    --res-dir ./track2_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0/ \
    --output-dir ./track1_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400 \
    --reid_th 0.3 \
    --len_th 400 \
    --workers 24

#10. post processing 

python track1/post_processing.py --split $split \
    --res-dir ./track1_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400 \
    --output-dir ./track1_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400_30_ios0.5_mg30_itp500 \
    --nms_dist_th 30 \
    --nms_ios_th 0.5 \
    --merge_threshold 30 \
    --interp_threshold 500 \
    --workers 24

#11. track1 to track2 matching

python track1to2/main_track1to2.py --split $split \
    --base-dir ${MMP_BASE_DIR} \
    --detections-dir ./features/yolox_finetune_mmp_r50ibna/ \
    --track1-dir ./track1_interp_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400_30_ios0.5_mg30_itp500 \
    --output-dir ./track1to2_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400_30_ios0.5_mg30_itp500_mc0.0_cos0.3_pos10 \
    --cos_th 0.3 \
    --pos_th 10 \
    --min_confidence 0.0 \
    --workers 24

#This is our final track2 results. we can use tools/save_track2_for_online_evaluation.py to save it in original format.

#python tools/save_track2_for_online_evaluation.py \
#    --res-dir ./track1to2_results/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400_30_ios0.5_mg30_itp500_mc0.0_cos0.3_pos10 \
#    --output-dir ./track1to2_online/yolox_finetune_mmp_r50ibna_th3_cos0.4_nms0.65_0.5clsPA_itp0_reid0.3_l400_30_ios0.5_mg30_itp500_mc0.0_cos0.3_pos10 \
#    --workers 24
