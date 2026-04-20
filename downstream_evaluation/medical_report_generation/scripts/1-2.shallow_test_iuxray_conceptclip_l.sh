#!/bin/bash
cd ..
dataset="iu_xray"
annotation="dataset/annotation.json"
base_dir="dataset/images"
delta_file="models/ckpt/conceptclip_l_iu.pth"

version="v1_shallow"
vision_encoder="conceptclip_l"
savepath="outputs/$dataset/$version/$vision_encoder"

python -u main_conceptclip_l.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 8 \
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