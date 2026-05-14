#!/bin/bash
#SBATCH --job-name=submit_mpi
#SBATCH --output=mpi_output_%j.txt
#SBATCH --error=mpi_output_%j.txt
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=general

# ─── Variables ────────────────────────────────────────────────────────────────
MPICXX=mpic++
CXXFLAGS="-O3 -std=c++17 -march=native -Wall"
CONFIG_PATH="../additional/Config.h"
TARGET_MPI=run_mpi
SRCS_MPI=run_mpi.cpp

PROCS_LIST=(1 2 4 8 16)
BODIES_PER_PROC=1000      # weak scaling: total N = BODIES_PER_PROC * num_procs
DEFAULT_NUM_BODIES=5000   # restored at the end

# ─── Helper: update NUM_BODIES in Config.h ────────────────────────────────────
set_num_bodies() {
    local N=$1
    sed -i "s/static constexpr int NUM_BODIES = [0-9]*/static constexpr int NUM_BODIES = $N/" $CONFIG_PATH
    echo "  [Config.h] NUM_BODIES set to $N"
}

# ─── Sanity check ─────────────────────────────────────────────────────────────
if [[ ! -f $CONFIG_PATH ]]; then
    echo "Error: Config.h not found at $CONFIG_PATH"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Positions run  (1 process, N=5000)
# ══════════════════════════════════════════════════════════════════════════════
echo "=== Positions run (N=5000, 1 process) ==="
set_num_bodies 5000
$MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI
mpirun --bind-to none -np 1 ./$TARGET_MPI
echo "  positions.csv written"

# ══════════════════════════════════════════════════════════════════════════════
# 2.  STRONG SCALING  — fixed N=5000, procs = 1 2 4 8 16
#     Output: mpi_timing_strongscale_<procs>.csv
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Strong Scaling (N=5000) ==="
set_num_bodies 5000
$MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI

for PROCS in "${PROCS_LIST[@]}"; do
    TIMING_FILE="mpi_timing_strongscale_${PROCS}.csv"
    echo "Program,Processes,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $PROCS process(es), N=5000 -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" mpirun --bind-to none -np $PROCS ./$TARGET_MPI > /dev/null; } 2>&1 )
    echo "mpi,$PROCS,5000,$ELAPSED_TIME" >> $TIMING_FILE
done

# ══════════════════════════════════════════════════════════════════════════════
# 3.  WEAK SCALING  — N scales with procs (1000 bodies per process)
#     1 proc->N=1000, 2->2000, 4->4000, 8->8000, 16->16000
#     Output: mpi_timing_weakscale_<procs>.csv
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Weak Scaling ($BODIES_PER_PROC bodies/process) ==="

for PROCS in "${PROCS_LIST[@]}"; do
    N=$(( BODIES_PER_PROC * PROCS ))
    TIMING_FILE="mpi_timing_weakscale_${PROCS}.csv"

    echo "  Recompiling with NUM_BODIES=$N ..."
    set_num_bodies $N
    $MPICXX $CXXFLAGS -I../additional -o $TARGET_MPI $SRCS_MPI

    echo "Program,Processes,N_Bodies,Time(s)" > $TIMING_FILE
    echo "  Running: $PROCS process(es), N=$N -> $TIMING_FILE"
    ELAPSED_TIME=$( { /usr/bin/time -f "%e" mpirun --bind-to none -np $PROCS ./$TARGET_MPI > /dev/null; } 2>&1 )
    echo "mpi,$PROCS,$N,$ELAPSED_TIME" >> $TIMING_FILE
done

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Restore Config.h to default
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Restoring Config.h to NUM_BODIES=$DEFAULT_NUM_BODIES ==="
set_num_bodies $DEFAULT_NUM_BODIES
echo "Done. All timing CSVs written."
