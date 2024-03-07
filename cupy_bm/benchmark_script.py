# --
from datetime import datetime
time_start = datetime.now()

# --
import cupy as cp
import numpy as np

cpu_iterations = 1
gpu_iterations = 10

array_size = 256 * 32

if cp.cuda.is_available():
    device = "placeholder ... get device function" # placeholder 
    print("CUDA is available - proceeding ...")
    print("")

    # CPU Benchmark
    print("--")
    print("CPU Benchmarks (using NumPy) -", cpu_iterations, "iterations")
    numpy_x = np.random.rand(array_size, array_size) # does this work?
    numpy_y = np.random.rand(array_size, array_size) # does this work?

    for i in range(cpu_iterations):
        start = datetime.now()
        numpy_result = np.dot(numpy_x, numpy_y)                # does this work?
        print("Using NumPy - took:", datetime.now() - start)

    # GPU Benchmark
    print("--")
    print("GPU Benchmarks (using CuPy) -", gpu_iterations, "iterations")
    cupy_x = cp.random.rand(array_size, array_size) # does this work?
    cupy_y = cp.random.rand(array_size, array_size) # does this work?

    for i in range(gpu_iterations):
        start = datetime.now()
        cupy_result = cp.dot(cupy_x, cupy_y)                # does this work?
        print("Using device:", cupy_result.device, "- took:", datetime.now() - start)

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
print("CUDA version:", cp.cuda.runtime.runtimeGetVersion())
print("CuPy version:", cp.__version__)
