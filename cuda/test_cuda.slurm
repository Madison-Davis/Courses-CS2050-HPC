#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --output=cuda_test_%j.out
#SBATCH --error=cuda_test_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

set -e

# Load proper Spack environment
source ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu
echo "nvcc: $(which nvcc)"
nvcc --version

# Compile cuda
echo "Compiling CUDA test..."
nvcc -O3 -std=c++17 \
    -DTEST_MODE \
    -I../additional \
    -I../serial \
    test_cuda.cpp run_cuda.cu ../serial/run_serial.cpp \
    -o test_cuda

# Run tests
export CUDA_VISIBLE_DEVICES=0
./test_cuda