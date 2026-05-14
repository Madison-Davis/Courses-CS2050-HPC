#!/bin/bash
#SBATCH --job-name=openmp_profile
#SBATCH --output=openmp_profile_output_%j.out
#SBATCH --error=openmp_profile_error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=general

cd "$SLURM_SUBMIT_DIR"
CONFIG_PATH="../additional/Config.h"

# Make sure config.h exists
if [[ ! -f $CONFIG_PATH ]]; then
    echo "Error: Config.h not found!"
    exit 1
fi

# Compile
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo ">>> Compiling on $(hostname)..."
g++ -O2 -fopenmp -I../additional -g -fno-omit-frame-pointer -std=c++17 -o run_openmp run_openmp.cpp
echo "    Done."

# Profile
echo ">>> Running VTune..."
vtune -collect hotspots \
      -knob sampling-mode=sw \
      -result-dir vtune_hotspots_openmp \
      -- ./run_openmp

# Report
echo ">>> Hotspot summary:"
vtune -report hotspots \
      -result-dir vtune_hotspots_openmp \
      -format text \
      -column="Function,CPU Time" \
      2>/dev/null | head -30