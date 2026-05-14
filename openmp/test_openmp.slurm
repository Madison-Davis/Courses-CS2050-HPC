#!/bin/bash
#SBATCH --job-name=omp_test
#SBATCH --output=omp_test_%j.out
#SBATCH --error=omp_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=general

# Variables
CXX=g++
CXXFLAGS="-O3 -std=c++17 -march=native -ffast-math -Wall"
OPENMPFLAGS="-fopenmp"
INCLUDE_PATH_ADDITIONAL="-I../additional"
INCLUDE_PATH_SERIAL="-I../serial"
TARGET=test_openmp
SRC=test_openmp.cpp

# Compile tests
echo "Compiling test..."
$CXX $CXXFLAGS $OPENMPFLAGS \
    $INCLUDE_PATH_ADDITIONAL $INCLUDE_PATH_SERIAL \
    -DTEST_MODE \
    $SRC ../serial/run_serial.cpp run_openmp.cpp \
    -o $TARGET
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
echo "Compilation successful"

# Run tests
echo "Running with 1 thread (baseline)"
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./$TARGET

echo "Running with ${SLURM_CPUS_PER_TASK} threads"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./$TARGET

echo ""
echo "All tests completed"