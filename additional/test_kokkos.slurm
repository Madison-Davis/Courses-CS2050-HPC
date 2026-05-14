#!/bin/bash
#SBATCH --job-name=kokkos_test
#SBATCH --output=kokkos_test_%j.out
#SBATCH --error=kokkos_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=general

# Variables
CXX=g++
CXXFLAGS="-O3 -std=c++20 -march=native -ffast-math -Wall"
OPENMPFLAGS="-fopenmp"
KOKKOS_INSTALL="$HOME/kokkos-install"
KOKKOS_INC="-I${KOKKOS_INSTALL}/include"
KOKKOS_LIB="-L${KOKKOS_INSTALL}/lib -L${KOKKOS_INSTALL}/lib64 -lkokkoscore -lkokkoscontainers"
INCLUDE_PATH_SERIAL="-I../serial"
TARGET=test_kokkos
SRC=test_kokkos.cpp

# Load environment
source ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# OpenMP binding
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KOKKOS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile
echo "Compiling test..."
$CXX $CXXFLAGS $OPENMPFLAGS \
    $KOKKOS_INC \
    $INCLUDE_PATH_SERIAL \
    -I. \
    -DTEST_MODE \
    $SRC ../serial/run_serial.cpp run_kokkos.cpp \
    $KOKKOS_LIB \
    -o $TARGET

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
echo "Compilation successful"

# Run tests
echo "Running Kokkos test with ${SLURM_CPUS_PER_TASK} threads..."
srun --cpu-bind=cores ./$TARGET

echo ""
echo "All tests completed"