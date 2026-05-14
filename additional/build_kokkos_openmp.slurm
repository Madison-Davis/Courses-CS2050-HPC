#!/bin/bash
#SBATCH --job-name=kokkos_build_openmp
#SBATCH --output=build_kokkos_openmp_output_%j.txt
#SBATCH --error=build_kokkos_openmp_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=general

set -e

# Navigate to proper directory
cd "$SLURM_SUBMIT_DIR"
cd kokkos
rm -rf build-openmp
mkdir -p build-openmp
cd build-openmp

# Compile with OpenMP Enabled
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-install-openmp \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DCMAKE_CXX_STANDARD=20

make -j16
make install
echo "OpenMP Kokkos build complete."