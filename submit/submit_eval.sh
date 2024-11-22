#!/bin/bash

# Specify job queue and resources
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=4
#PBS -l walltime=04:00:00
#PBS -P personal-e1325997
#PBS -N EvaluateModelMultiGPU
#PBS -o ./Results/

# Load Python environment
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llama  # Activate your Python environment

export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs

# Change to the working directory
cd ${PBS_O_WORKDIR}

# Run the evaluation script
python3 eval_new_data.py --model_name resnet152 \
                   --model_path /home/users/nus/e1325997/scratch/ResNet_binary/resnet152_pixelation_e1_b512_lr00001/final_model.pth\
                   --real_dir /home/users/nus/e1325997/scratch/thumbnails128x128 \
                   --fake_dir /home/users/nus/e1325997/scratch/DFDB \
                   --batch_size 1024 \
                   --limit 10000

