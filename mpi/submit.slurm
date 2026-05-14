#!/bin/bash
#SBATCH --job-name=submit_mpi
#SBATCH --output=mpi_output_%j.txt
#SBATCH --error=mpi_output_%j.txt
#SBATCH --nodes=8                   # as per directions, run on > 1 node
#SBATCH --ntasks=16                 # total MPI ranks
#SBATCH --cpus-per-task=1           # 1 core per rank
#SBATCH --time=01:00:00
#SBATCH --partition=general

# Fix home directory for dynamic nodes
export HOME=$(getent passwd $USER | cut -d: -f6)

# Variables
MPICXX=mpic++
CXXFLAGS="-O3 -std=c++17 -march=native -Wall"
CONFIG_PATH="../additional/Config.h"
TARGET_MPI=run_mpi
SRCS_MPI=run_mpi.cpp
PROCS_LIST=(1 2 4 8 16)
BODIES_PER_PROC=1000      # weak scaling: total N = BODIES_PER_PROC * num_procs
DEFAULT_NUM_BODIES=5000   # restored at the end

# Helper function
set_num_bodies() {
    local N=$1
    sed -i "s/static constexpr int NUM_BODIES = [0-9]*/static constexpr int NUM_BODIES = $N/" $CONFIG_PATH
    echo "  [Config.h] NUM_BODIES set to $N"
}

# Make sure Config.h exists
if [[ ! -f $CONFIG_PATH ]]; then
    echo "Error: Config.h not found!"
    exit 1
fi

# Compile MPI version
echo "Compiling MPI version"
$MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI

# Run with 1 process to produce positions csv file
echo "Running MPI version (1 process) to generate positions.csv"
srun --export=ALL -n 1 ./$TARGET_MPI

# Strong scaling
echo "Running Strong Scaling (N=5000) to produce timing csv files"
set_num_bodies 5000
$MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI
for PROCS in "${PROCS_LIST[@]}"; do
    TIMING_FILE="mpi_timing_strongscale_${PROCS}.csv"
    echo "Program,Processes,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $PROCS process(es), N=5000 -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" srun --export=ALL -n $PROCS ./$TARGET_MPI > /dev/null; } 2>&1 | tail -1 )
    echo "mpi,$PROCS,5000,$ELAPSED_TIME" >> $TIMING_FILE
done

# Weak scaling
echo "Running Weak Scaling ($BODIES_PER_PROC bodies/process) to produce timing csv files"
for PROCS in "${PROCS_LIST[@]}"; do
    N=$(( BODIES_PER_PROC * PROCS ))
    TIMING_FILE="mpi_timing_weakscale_${PROCS}.csv"
    echo "  Recompiling with NUM_BODIES=$N ..."
    set_num_bodies $N
    $MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI
    echo "Program,Processes,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $PROCS process(es), N=$N -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" srun --export=ALL -n $PROCS ./$TARGET_MPI > /dev/null; } 2>&1 | tail -1 )
    echo "mpi,$PROCS,$N,$ELAPSED_TIME" >> $TIMING_FILE
done

# Restore Config.h to default
set_num_bodies $DEFAULT_NUM_BODIES
echo "All MPI runs completed. Timing CSVs generated."