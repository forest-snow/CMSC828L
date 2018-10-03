#!/bin/sh
#SBATCH --qos=dque
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=02:00:00

python model.py
