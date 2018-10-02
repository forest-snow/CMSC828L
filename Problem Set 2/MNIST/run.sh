#!/bin/sh
#SBATCH --qos=shallow
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=02:00:00

python mnist.py
