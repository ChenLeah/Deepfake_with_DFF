#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=4
#PBS -l walltime=05:00:00
#PBS -P personal-e1325997
#PBS -N ResNet50_Contrast_Training
#PBS -o ./Results/

# Load Python environment
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llama  # Activate the virtual environment

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Variables for this configuration
MODEL_NAME="resnet50" # Replace with resnet101 or resnet152 as needed
PERTURBATION="pixelation" # Replace with contrast pixelation or gaussian_noise as needed
EPOCHS=1
BATCH_SIZE=512
LR=0.00001
ROOT_DIR="images"
OUTPUT_DIR="/home/users/nus/e1325997/scratch/ResNet_binary/${MODEL_NAME}_${PERTURBATION}_contrast_e${EPOCHS}_b${BATCH_SIZE}_lr00001"
FINE_TUNED_PATH="/home/users/nus/e1325997/scratch/ResNet_binary/resnet50_contrast_e1_b1024_lr00001/final_model.pth"  # Update path if necessary
SCRIPT="adversarial_training.py"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

cd ${PBS_O_WORKDIR}
# Run the script
python3 $SCRIPT \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --root_dir $ROOT_DIR \
    --output_dir $OUTPUT_DIR \
    --perturbation $PERTURBATION \
    --fine_tuned_path $FINE_TUNED_PATH



