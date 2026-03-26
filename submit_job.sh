#!/bin/bash
# SBATCH --job-name=ssl_climate_4x_dinov3-sat
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hoang/SSL/ssl_training_pipeline/logs/lora8_ssl_climate_4x_dinov3-sat.%j.out
#SBATCH --error=/home/hoang/SSL/ssl_training_pipeline/logs/lora8_ssl_climate_4x_dinov3-sat.%j.err

CONFIG_FILE=$1

source ~/.bashrc
conda activate runai

# ── Load Environment Variables ────────────────────────────────────────────────
# Define the path to your .env file (assuming project root is /home/hoang/SSL/)
ENV_FILE="/home/hoang/SSL/ssl_training_pipeline/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    # set -a: automatically export all variables defined in the following source command
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

# Verify WANDB_API_KEY is loaded (optional debug)
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set!"
else
    echo "WANDB_API_KEY loaded successfully (length: ${#WANDB_API_KEY})"
fi
export OMP_NUM_THREADS=1

# Verify GPU allocation
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Auto-detect number of GPUs from Slurm
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)

echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

# Launch training
torchrun --nproc_per_node=$NUM_GPUS training.py --config $CONFIG_FILE 