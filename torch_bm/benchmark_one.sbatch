#!/bin/bash

#SBATCH --job-name=torch_bm
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --nodelist=nv-ai-01
#SBATCH --output=/home/its.aau.dk/cc11gz/tests/outputs/%x-%j.out
#SBATCH --error=/home/its.aau.dk/cc11gz/tests/outputs/%x-%j.err

srun singularity exec --nv pytorch_23.10.sif python3 benchmark_script.py
