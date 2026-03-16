#!/bin/bash
#SBATCH --job-name=sft_llada
#SBATCH --partition=h100
#SBATCH --account=aszalay1_ssci
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --mem=512G
#SBATCH --exclude=c001,h07

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export TORCH_EXTENSIONS_DIR="/scratch/aszalay1/tianze/L-dLLM/.cache/torch_extensions"
export TRITON_CACHE_DIR="/scratch/aszalay1/tianze/L-dLLM/.cache/triton"
export HF_HOME="/scratch/aszalay1/tianze/L-dLLM/.cache/huggingface"
export DEBUG_MODE="1"

mkdir -p $TORCH_EXTENSIONS_DIR
mkdir -p $TRITON_CACHE_DIR
mkdir -p $HF_HOME

# run script
# Using the configuration for 1 node with 4 GPUs (matching the SBATCH request)
# Ensure the accelerate config matches the resources requested
ACCELERATE_CONFIG="accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml"
SCRIPT="train/cpt_latent_llada_fullscale.py"
CONFIG="configs/cpt_latent_llada_fullscale.yaml"

echo "Starting training with config: $CONFIG"

accelerate launch \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port 30002 \
    --config_file $ACCELERATE_CONFIG \
    $SCRIPT \
    config=$CONFIG
