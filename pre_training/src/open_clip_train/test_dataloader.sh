#!/bin/bash

#SBATCH -J mbook #Slurm job name
#SBATCH -t 10:00:00
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1 # This needs to match Trainer(devices=...)
#SBATCH --account=medimgfmod

#SBATCH --mail-user=ynieae@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --output=test_dataloader.log
#SBATCH --error=test_dataloader.error.log

source ~/.bashrc
source activate torch24


# Go to the job submission directory and run your application

cd /home/ynieae/open_clip_med_pretrain/src/open_clip_train

python test_dataloader.py
