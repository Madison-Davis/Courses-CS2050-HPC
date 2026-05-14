#!/bin/bash
#SBATCH --job-name=submit_openmp
#SBATCH --output=omp_output_%j.txt
#SBATCH --error=omp_output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=general

# Variables
CXX=g++
CXXFLAGS="-O3 -std=c++17 -march=native -Wall"
OPENMPFLAGS="-fopenmp"
CONFIG_PATH="../additional/Config.h"
TARGET_OMP=run_openmp
SRCS_OMP=run_openmp.cpp
THREADS_LIST=(1 2 4 8 16)
BODIES_PER_THREAD=1000      # weak scaling: total N = BODIES_PER_THREAD * num_threads
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

# Compile OpenMP
echo "Compiling OpenMP version"
$CXX $CXXFLAGS $OPENMPFLAGS -I../additional -o $TARGET_OMP $SRCS_OMP

# Run OpenMP with 1 thread to produce positions csv file
echo "Running OpenMP version (1 thread) to generate positions.csv"
export OMP_NUM_THREADS=1
./$TARGET_OMP

# Strong scaling
echo "Running Strong Scaling (N=5000) to produce timing csv files"
set_num_bodies 5000
$CXX $CXXFLAGS $OPENMPFLAGS -I../additional -o $TARGET_OMP $SRCS_OMP
for THREADS in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS=$THREADS
    TIMING_FILE="openmp_timing_strongscale_${THREADS}.csv"
    echo "Program,Threads,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $THREADS thread(s), N=5000 -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" ./$TARGET_OMP > /dev/null; } 2>&1 )
    echo "openmp,$THREADS,5000,$ELAPSED_TIME" >> $TIMING_FILE
done

# Weak scaling
echo "Running Weak Scaling ($BODIES_PER_THREAD bodies/thread) to produce timing csv files"
for THREADS in "${THREADS_LIST[@]}"; do
    N=$(( BODIES_PER_THREAD * THREADS ))
    export OMP_NUM_THREADS=$THREADS
    TIMING_FILE="openmp_timing_weakscale_${THREADS}.csv"
    echo "  Recompiling with NUM_BODIES=$N ..."
    set_num_bodies $N
    $CXX $CXXFLAGS $OPENMPFLAGS -I../additional -o $TARGET_OMP $SRCS_OMP
    echo "Program,Threads,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $THREADS thread(s), N=$N -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" ./$TARGET_OMP > /dev/null; } 2>&1 )
    echo "openmp,$THREADS,$N,$ELAPSED_TIME" >> $TIMING_FILE
done

# Restore Config.h to default
set_num_bodies $DEFAULT_NUM_BODIES
echo "All jobs completed. Timing CSVs saved per thread count"