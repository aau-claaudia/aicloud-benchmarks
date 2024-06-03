#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --
from datetime import datetime
time_start = datetime.now()

# --
import torch

cpu_iterations = 1
gpu_iterations = 10

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available - proceeding ...")
    
    #
    matrix_size = 32 * 512
    x = torch.randn(matrix_size, matrix_size)
    y = torch.randn(matrix_size, matrix_size)
    
    # CPU Benchmark
    print("")
    print("-- CPU Benchmarks -", cpu_iterations, "iterations")
    for i in range(cpu_iterations):
        start = datetime.now()
        cpu_result = torch.mm(x, y)
        print("Using device:", cpu_result.device, "- took:", datetime.now() - start)
    
    # GPU Benchmark
    print("-- GPU Benchmarks -", gpu_iterations, "iterations")
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    for i in range(gpu_iterations):
        start = datetime.now()
        gpu_result = torch.mm(x_gpu, y_gpu)
        print("Using device:", gpu_result.device, "- took:", datetime.now() - start)
else:
    print("CUDA is not available - cancelling test ...")

# -- 
time_stop = datetime.now()

print("")
print('Started at:\t', time_start.strftime("%Y-%m-%d %H:%M:%S"))
print('Finished at:\t', time_stop.strftime("%Y-%m-%d %H:%M:%S"))
print('Duration:\t', time_stop - time_start)

# --
import subprocess
print("--")
print("Nodename:", subprocess.run("printf `uname -n`", shell = True, capture_output=True, text=True).stdout)
print("Nvidia driver version:", subprocess.run("printf `nvidia-smi --query-gpu=driver_version --format=noheader,csv`", shell = True, capture_output=True, text=True).stdout)
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
