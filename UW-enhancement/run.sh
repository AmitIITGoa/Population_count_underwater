#!/bin/bash
# ============================================
# SLURM Job Submission Script — High Priority (6 GPUs)
# ============================================

#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --time=03:00:00               # Maximum run time (hh:mm:ss)
#SBATCH --job-name=main_high_gpu      # Job name
#SBATCH --partition=gpu               # Partition name
#SBATCH --gres=gpu:1                 # Request 6 GPUs
#SBATCH --account=amit               # Account name
#SBATCH --output=amit.out            # Standard output log file
#SBATCH --error=amit.err             # Standard error log file

# ============================================
# High Priority Scheduling
# ============================================
# NOTE: Priority can depend on your cluster’s configuration.
# Some clusters support QoS (Quality of Service) for priority jobs.

#SBATCH --qos=high                    # Request high priority (if supported)
# export CUDA_VISIBLE_DEVICES=5

# ============================================
# User Configurations
# ============================================

# Path to your main Python script
SCRIPT_PATH="custom_dataset.py"

# (Optional) Load/Activate your environment
# source ~/.bashrc
# conda activate myenv

# ============================================
# Run the Training / Main Script
# ============================================

echo "============================================"
echo " Starting job: $SLURM_JOB_NAME"
echo " Node: $(hostname)"
echo " GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo " Start time: $(date)"
echo "============================================"

# Run the Python program using SLURM's srun
srun python3 "$SCRIPT_PATH"

echo "============================================"
echo " Job finished on $(date)"
echo "============================================"
