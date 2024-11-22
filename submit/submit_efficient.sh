#!/bin/bash

# Specify job queue and resources
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=4
#PBS -l walltime=08:00:00
#PBS -P personal-e1325997
#PBS -N ResNet50
#PBS -o ./Results/

# Load Python environment if necessary
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llama # Activate the virtual environment

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the test script

cd ${PBS_O_WORKDIR}

mkdir /home/users/nus/e1325997/scratch/EfficientNet_final/EfficentNetB2_e5_b512_lr0001
python3 EfficientNet.py --model_name "efficientnet_b2" --epochs 5 --batch_size 512 --lr 0.0001 --output_dir "/home/users/nus/e1325997/scratch/EfficientNet_final/EfficentNetB2_e5_b512_lr0001"
