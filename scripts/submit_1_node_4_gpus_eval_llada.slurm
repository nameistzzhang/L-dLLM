#!/bin/bash
#SBATCH --job-name=sft_llada
#SBATCH --partition=a100
#SBATCH --account=aszalay1_ssci
#SBATCH --gres=gpu:4
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
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script
EVAL_CONFIG="configs/llada_eval.yaml"

echo "Starting evaluation with config: $EVAL_CONFIG"

python eval.py config=$EVAL_CONFIG