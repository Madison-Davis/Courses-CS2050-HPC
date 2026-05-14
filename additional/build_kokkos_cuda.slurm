#!/bin/bash
#SBATCH --job-name=kokkos_build_cuda
#SBATCH --output=build_kokkos_cuda_output_%j.txt
#SBATCH --error=build_kokkos_cuda_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

# Load Spack environment with CUDA
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# Navigate to proper directory
cd "$SLURM_SUBMIT_DIR"
cd kokkos
rm -rf build-cuda
mkdir -p build-cuda
cd build-cuda

# Compile with CUDA Enabled
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-install-cuda \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_VOLTA70=ON \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_COMPILER=$(pwd)/../bin/nvcc_wrapper

make -j16
make install

echo "CUDA Kokkos build complete."