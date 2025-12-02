#!/bin/bash

#SBATCH --job-name=dp_turn
#SBATCH --output=logs/%j_dp_turn_output.log
#SBATCH --error=logs/%j_dp_turn_error.log
#SBATCH --partition=mlgpu_short
#SBATCH --time=5:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

# Initialize conda (try common locations)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    eval "$(conda shell.bash hook)"
fi

conda activate aienv
python main.py