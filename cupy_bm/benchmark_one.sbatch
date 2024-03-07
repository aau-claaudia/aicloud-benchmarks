#!/bin/bash

#SBATCH --job-name=cupy_bm
#SBATCH --gres=gpu:1
#SBATCH --reservation=root_51
#SBATCH --nodelist=nv-ai-04
#SBATCH --partition=batch
#SBATCH --time=0:10:00
#SBATCH --output=/home/its.aau.dk/cc11gz/tests/outputs/%x-%j.out
#SBATCH --error=/home/its.aau.dk/cc11gz/tests/outputs/%x-%j.err

singularity exec --nv /home/its.aau.dk/cc11gz/containers/cupy.sif python3 /home/its.aau.dk/cc11gz/sysadm/tests/scripts/cupy_benchmark.py
