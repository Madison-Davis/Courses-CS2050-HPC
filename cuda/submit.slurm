#!/bin/bash
#SBATCH --job-name=submit_cuda
#SBATCH --output=cuda_output_%j.txt
#SBATCH --error=cuda_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu

# Variables
NVCC=nvcc
CXXFLAGS="-O3 -std=c++17"
CONFIG_PATH="../additional/Config.h"
TARGET_CUDA=run_cuda
SRC=run_cuda.cu
BODY_COUNTS=(128 256 512 1024 2048 4096 5000 8192 16384 32768 65536 131072)
DEFAULT_NUM_BODIES=5000     # restored at the end

# Helper function
set_num_bodies() {
    local N=$1
    sed -i "s/static constexpr int NUM_BODIES = [0-9]*/static constexpr int NUM_BODIES = $N/" $CONFIG_PATH
    echo "  [Config.h] NUM_BODIES set to $N"
}

# Make sure config.h exists
if [[ ! -f $CONFIG_PATH ]]; then
    echo "Error: Config.h not found!"
    exit 1
fi

# Load proper Spack environment
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# Showcase NVIDIA-SMI
echo "NVIDIA-SMI snapshot before run:"
nvidia-smi || true
export CUDA_VISIBLE_DEVICES=0

# Body-count scaling (vary N to find where simulation sits on the GPU scaling curve)
echo "Running body-count scaling to produce timing csv files"
for N in "${BODY_COUNTS[@]}"; do
    TIMING_FILE="cuda_timing_scale_${N}.csv"
    echo "Program,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Recompiling with NUM_BODIES=$N ..."
    set_num_bodies $N
    $NVCC $CXXFLAGS -I../additional -o $TARGET_CUDA $SRC
    echo "  Running: N=$N => $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" srun --cpu-bind=cores ./$TARGET_CUDA > /dev/null; } 2>&1 )
    echo "cuda,$N,$ELAPSED_TIME" >> $TIMING_FILE
done

# Restore Config.h to default
set_num_bodies $DEFAULT_NUM_BODIES
echo "All jobs completed. Timing CSVs saved per body count."


"""
# If you wish to see the L2 makeup via CUDA deviceQuery, run a sample function really quick...
cat << 'EOF' > /tmp/device_query.cu
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("L2 cache size: %d MB\n", prop.l2CacheSize / (1024 * 1024));
    printf("CUDA cores: %d SMs x %d = %d\n",
           prop.multiProcessorCount,
           128,  // L4 has 128 CUDA cores per SM
           prop.multiProcessorCount * 128);
    printf("Global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    return 0;
}
EOF
$NVCC /tmp/device_query.cu -o /tmp/device_query && /tmp/device_query
"""