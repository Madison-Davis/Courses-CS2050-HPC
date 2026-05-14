#!/bin/bash
#SBATCH --job-name=submit_kokkos_cuda
#SBATCH --output=kokkos_cuda_output_%j.txt
#SBATCH --error=kokkos_cuda_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -e

CONFIG_DIR="$SLURM_SUBMIT_DIR"
SRC="$CONFIG_DIR/run_kokkos.cpp"
TARGET="$CONFIG_DIR/run_kokkos_cuda"

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

echo "Running Kokkos CUDA benchmark..."
cd "$SLURM_SUBMIT_DIR"

[[ -f "$SRC" ]] || { echo "Missing $SRC"; exit 1; }

echo "Compiling..."
"$CONFIG_DIR/kokkos/bin/nvcc_wrapper" -O3 -std=c++20 \
  --expt-extended-lambda \
  -I"$CONFIG_DIR" \
  -I"$HOME/kokkos-install-cuda/include" \
  "$SRC" \
  -L"$HOME/kokkos-install-cuda/lib" \
  -L"$HOME/kokkos-install-cuda/lib64" \
  -lkokkoscore \
  -lkokkoscontainers \
  -lcuda \
  -lcudart \
  -ldl \
  -o "$TARGET"

echo "Running benchmark..."
export CUDA_VISIBLE_DEVICES=0
srun "$TARGET"