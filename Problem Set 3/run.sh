#!/bin/sh
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
load_var=0
python model_adult.py $load_var
