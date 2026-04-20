#!/bin/bash
cd ..
dataset="iu_xray"
annotation="dataset/annotation.json"
base_dir="dataset/images"

version="v1_shallow"
vision_encoder="conceptclip_l"
savepath="outputs/$dataset/$version/$vision_encoder"

cuda_devices="0"  

CUDA_VISIBLE_DEVICES=$cuda_devices python -u main_conceptclip_l.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 2 \
    --accumulate_grad_batches 8 \
    --val_batch_size 4 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 15 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    2>&1 |tee scripts/logs/${dataset}_${version}_${vision_encoder}.txt