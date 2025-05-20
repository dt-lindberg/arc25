#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --job-name=ModelTest
#SBATCH --output=slurm_output_%A.out

# Loading modules (CUDA and Anaconda are located in module 2024)
module load 2024
module load CUDA/12.6.0
module load Anaconda3/2024.06-1

# Set working directory
cd $HOME/arc25

# Activate conda environment
source activate lips_env

# Logging info
echo "Starting job at $(date)"

# Run the python script
srun python src/run_qwen.py --num-tasks 5 --output-file qwen_evaluation_results.json --verbose

echo "Finished job at $(date)"