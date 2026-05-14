#!/bin/bash
#SBATCH --job-name=serial_profile
#SBATCH --output=serial_profile_output_%j.out
#SBATCH --error=serial_profile_error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
echo ">>> Compiling on $(hostname)..."
g++ -O2 -I../additional -g -fno-omit-frame-pointer -std=c++17 -o run_serial run_serial.cpp
echo "    Done."

# Profile
echo ">>> Running VTune..."
vtune -collect hotspots \
      -knob sampling-mode=sw \
      -result-dir vtune_hotspots_serial \
      -- ./run_serial

# Report
echo ">>> Hotspot summary:"
vtune -report hotspots \
      -result-dir vtune_hotspots_serial \
      -format text \
      -column="Function,CPU Time" \
      2>/dev/null | head -30