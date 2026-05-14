#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --output=mpi_test_%j.out
#SBATCH --error=mpi_test_%j.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=general

# Variables
CXX=mpicxx
CXXFLAGS="-O3 -std=c++17 -march=native -ffast-math -Wall"
INCLUDE_PATH_ADDITIONAL="-I../additional"
INCLUDE_PATH_SERIAL="-I../serial"
TARGET=test_mpi
SRC=test_mpi.cpp

# Compile tests
echo "Compiling test..."
$CXX $CXXFLAGS \
    $INCLUDE_PATH_ADDITIONAL $INCLUDE_PATH_SERIAL \
    -DTEST_MODE \
    $SRC ../serial/run_serial.cpp run_mpi.cpp \
    -o $TARGET
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
echo "Compilation successful"

# Run tests
echo "Running with 1 process (baseline)"
mpirun --bind-to none -np 1 ./$TARGET

echo "Running with ${SLURM_NTASKS} processes"
mpirun --bind-to none -np $SLURM_NTASKS ./$TARGET

echo ""
echo "All tests completed"