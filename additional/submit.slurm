#!/bin/bash
#SBATCH --job-name=submit_kokkos_openmp
#SBATCH --output=kokkos_openmp_%j.txt
#SBATCH --error=kokkos_openmp_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=general

set -e

# Variables
CONFIG_DIR="$SLURM_SUBMIT_DIR"
SRC="$CONFIG_DIR/run_kokkos.cpp"
TARGET="$CONFIG_DIR/run_kokkos"
CONFIG_PATH="$CONFIG_DIR/Config.h"
THREADS_LIST=(1 2 4 8 16)
FIXED_BODIES=5000

# Safety checks
[[ -f "$SRC" ]] || { echo "Missing source"; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Missing Config.h"; exit 1; }

# Helper functions
set_num_bodies() {
    local N=$1
    sed -i "s/static constexpr int NUM_BODIES = [0-9]*/static constexpr int NUM_BODIES = $N/" $CONFIG_PATH
    echo "  NUM_BODIES = $N"
}

compile_kokkos() {
    g++ -O3 -std=c++20 -fopenmp \
      -I"$CONFIG_DIR" \
      -I"$HOME/kokkos-install/include" \
      "$SRC" \
      -L"$HOME/kokkos-install/lib" \
      -L"$HOME/kokkos-install/lib64" \
      -lkokkoscore \
      -lkokkoscontainers \
      -o "$TARGET"
}
cd "$SLURM_SUBMIT_DIR"

# Compile Kokkos OpenMP
echo "Compiling..."
set_num_bodies $FIXED_BODIES
compile_kokkos

# Run Kokkos OpenMP
echo "Running strong scaling..."
for THREADS in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS=$THREADS
    export KOKKOS_NUM_THREADS=$THREADS
    TIMING_FILE="kokkos_openmp_${THREADS}.csv"
    echo "Program,Threads,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $THREADS threads, N=$FIXED_BODIES"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" \
    srun --cpu-bind=cores "$TARGET" --kokkos-threads=$THREADS > /dev/null; } 2>&1 )
    echo "kokkos,$THREADS,$FIXED_BODIES, Elapsed Time: $ELAPSED_TIME" >> $TIMING_FILE
done
echo "Done."