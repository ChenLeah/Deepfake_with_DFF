#!/bin/bash

# Specify job queue and resources
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=4
#PBS -l walltime=05:00:00
#PBS -P personal-e1325997
#PBS -N ResNet_Training
#PBS -o ./Results/

# Load Python environment
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llama  # Activate the virtual environment

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training for ResNet50, ResNet101, and ResNet152 with specified parameters

cd ${PBS_O_WORKDIR}

# ResNet50
MODEL_NAME="resnet50"
BATCH_SIZE=1024
LR=0.000001
OUTPUT_DIR="/home/users/nus/e1325997/scratch/ResNet_binary/ResNet50_e5_b1024_lr000001"
mkdir -p $OUTPUT_DIR
python3 ResNet.py --model_name $MODEL_NAME --epochs 5 --batch_size $BATCH_SIZE --lr $LR --output_dir $OUTPUT_DIR

# # ResNet101
# MODEL_NAME="resnet101"
# BATCH_SIZE=512
# LR=0.000001
# OUTPUT_DIR="/home/users/nus/e1325997/scratch/ResNet_binary/ResNet101_e5_b512_lr000001"
# mkdir -p $OUTPUT_DIR
# python3 ResNet.py --model_name $MODEL_NAME --epochs 5 --batch_size $BATCH_SIZE --lr $LR --output_dir $OUTPUT_DIR

# # ResNet152
# MODEL_NAME="resnet152"
# BATCH_SIZE=256
# LR=0.000001
# OUTPUT_DIR="/home/users/nus/e1325997/scratch/ResNet_binary/ResNet152_e5_b256_lr000001"
# mkdir -p $OUTPUT_DIR
# python3 ResNet.py --model_name $MODEL_NAME --epochs 5 --batch_size $BATCH_SIZE --lr $LR --output_dir $OUTPUT_DIR
