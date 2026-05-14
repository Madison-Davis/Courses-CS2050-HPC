#!/bin/bash
#SBATCH --job-name=submit_serial
#SBATCH --output=serial_output_%j.txt
#SBATCH --error=serial_output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=general

# Variables
CXX=g++
CXXFLAGS="-O3 -std=c++17 -march=native -Wall"
CONFIG_PATH="../additional/Config.h"
TARGET_SERIAL=run_serial
SRCS_SERIAL=run_serial.cpp

# Make sure config.h exists
if [[ ! -f $CONFIG_PATH ]]; then
    echo "Error: Config.h not found!"
    exit 1
fi

# Compile serial
echo "Compiling serial version"
$CXX $CXXFLAGS -I../additional -o $TARGET_SERIAL $SRCS_SERIAL

# Run serial to produce position and timing csv files
echo "Running serial version to generate position and timing csv files"
./$TARGET_SERIAL