#!/bin/sh
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
load_var=False
python model_flowers.py $load_var