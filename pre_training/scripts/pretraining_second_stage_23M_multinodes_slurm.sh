#!/bin/bash

#SBATCH -J pretraining #Slurm job name
#SBATCH -t 168:00:00
#SBATCH -p batch
#SBATCH --nodes=6 # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks-per-node=1 # This needs to match Trainer(devices=...)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --output=src/logs/srun_output/pretraining_second_stage_23M_multinodes_slurm.txt
#SBATCH --error=src/logs/srun_output/pretraining_second_stage_23M_multinodes_slurm.error.txt


source ~/.bashrc
source activate torch24
cd src

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

set -x

# GPUS=${GPUS:-16}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
# NODES=$((GPUS / GPUS_PER_NODE))


# PYTHONPATH="${PYTHONPATH}:$(pwd)"
# MASTER_PORT=29666
# TF_CPP_MIN_LOG_LEVEL=3


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12666
WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"


export OMP_NUM_THREADS=16
export WANDB_API_KEY='xxxx'
export TORCH_CUDNN_V8_API_ENABLED=1
srun torchrun --nnodes=6 --nproc_per_node 8 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --rdzv_backend=c10d --rdzv_id=$RANDOM --rdzv_endpoint=${head_node_ip}:36899 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data 'pretraining_sample_data/pretraining_meta_file_sample_min.jsonl' \
    --image-dir 'pretraining_sample_data/sample_images' \
    --pretrained 'PATH_TO_PRETRAINED_MODEL_IN_FIRST_STAGE' \
    --dataset-type pmc  \
    --resume latest \
    --lr "3e-4" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 2000 \
    --wd 0.2 \
    --batch-size 128 \
    --aug-cfg scale='(0.5, 1.0)' \
    --epochs=20 \
    --workers=16 \
    --model ConceptCLIP-Pretraining \
    --grad-checkpointing \
    --grad-clip-norm 1.0 \
    --csv-img-key 'image' \
    --csv-caption-key 'caption' \
    --meta-key 'umls_meta_info' \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 384 \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5 \
    --image-interpolation 'bicubic' \
    --image-resize-mode 'squash' \
    --log-every-n-steps 4 \
    --seed 0 \
    --log-dir ./logs/ \
    --concept-loss-weight 0.5 \
    --name 'pretraining_second_stage_23M' \
    --report-to "tensorboard wandb" \
    --wandb-project-name "medclip" \
    --siglipconcept
