#!/bin/bash

# -------
job_name="cupy_bm"
start_time=$(date '+%Y-%m-%dT%H:%M:%S')
container="cupy.sif"
test_script="benchmark_script.py"
output_directory="$HOME/sysadm/tests/outputs/$job_name-$start_time"

mkdir -p $output_directory

# -------
# CHOOSE LIST OF TARGET NODES (by commenting out)

# Run on all nodes
# target_nodes=$(sinfo -aN | awk '/batch/{print $1}')

# Run on all available nodes
# target_nodes=$(sinfo -aN | awk '/(idle|mix)/ && /batch/{print $1}')

# Run on first available node
# target_nodes=$(sinfo -aN | awk '/(idle|mix)/ && /batch/{print $1}' | head -n 1)

# Run on specified nodes
target_nodes=nv-ai-02

# -------
for node in $target_nodes; do
  srun  -J $job_name \
        -t 15:00 \
        -G 1 \
        -w $node \
        -p batch \
        -o $output_directory/$node \
        singularity exec --nv $container python3 $test_script &
done
