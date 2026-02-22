#!/bin/bash
#SBATCH --job-name=sft_llada
#SBATCH --partition=h100
#SBATCH --account=aszalay1_ssci
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00

# environment setup
module purge
module load gcc/12.3.0
module load cuda/12.6.3
echo "Loaded Modules:"
module list
echo "GCC Version (Should be 12.3.0):"
g++ --version | head -n 1
echo "NVCC Version (Should be 12.6.3):"
nvcc --version
cd /scratch/aszalay1/tianze/L-dLLM
source .venv/bin/activate

# Ensure we use the allocated GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# run script
# Using the configuration for 1 node with 8 GPUs (matching the SBATCH request)
# Ensure the accelerate config matches the resources requested
ACCELERATE_CONFIG="accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml"
SCRIPT="train/sft_llada.py"
CONFIG="configs/sft_llada.yaml"

echo "Starting training with config: $CONFIG"

accelerate launch \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port 30002 \
    --config_file $ACCELERATE_CONFIG \
    $SCRIPT \
    config=$CONFIG
