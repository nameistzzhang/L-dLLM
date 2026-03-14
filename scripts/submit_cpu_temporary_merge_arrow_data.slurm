#!/bin/bash
#SBATCH --job-name=download_fineweb
#SBATCH --partition=cpu
#SBATCH --account=aszalay1_ssci
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

cd data
python temporary_merge_arrow_data.py