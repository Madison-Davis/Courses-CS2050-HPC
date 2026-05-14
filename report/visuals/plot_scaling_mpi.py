# plot_scaling_mpi.py

"""
Strong & Weak Scaling plots for MPI N-body simulation
Expected input files (relative to this script's location):
  ../../mpi/saved_run/mpi_timing_strongscale_<P>.csv
  ../../mpi/saved_run/mpi_timing_weakscale_<P>.csv

Each CSV has one header row and one data row:
  Program,Processes,N_Bodies,Time(s)
  mpi,<P>,<N>,<time>

Outputs:
  mpi_strong_scaling.png
  mpi_weak_scaling.png
"""


# Imports
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Path Variables
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "mpi", "saved_run")
OUT_STRONG  = os.path.join(SCRIPT_DIR, "mpi_strong_scaling.png")
OUT_WEAK    = os.path.join(SCRIPT_DIR, "mpi_weak_scaling.png")
PROCS_LIST  = [1, 2, 4, 8, 16]


# Plot Style Variables
MEASURED_COLOR = "#DD8452"      # orange to distinguish from OpenMP blue
IDEAL_COLOR    = "#cccccc"
FONT_TITLE     = dict(fontsize=14, fontweight="bold")
FONT_LABEL     = dict(fontsize=12)
FONT_TICK      = dict(labelsize=10)
FIG_SIZE       = (6.5, 5)
DPI            = 150


# Helper Function: Load the Timing Files
def load_timing(prefix):
    """Return dict {processes: time_seconds} for the given file prefix."""
    data = {}
    for p in PROCS_LIST:
        path = os.path.join(DATA_DIR, f"{prefix}_{p}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping P={p}")
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[int(row["Processes"])] = float(row["Time(s)"])
    return data


# Load the Data
print("Loading timing data...")
strong_data = load_timing("mpi_timing_strongscale")
weak_data   = load_timing("mpi_timing_weakscale")
if not strong_data:
    raise FileNotFoundError(f"No strong-scaling CSVs found in {DATA_DIR}")
if not weak_data:
    raise FileNotFoundError(f"No weak-scaling CSVs found in {DATA_DIR}")


# Derived Quantities
# Strong scaling: speedup = T(1)/T(p), efficiency = T(1)/(p*T(p))
strong_procs      = sorted(strong_data)
t1_strong         = strong_data[1]
strong_speedup    = [t1_strong / strong_data[p] for p in strong_procs]
ideal_speedup     = [float(p) for p in strong_procs]
strong_efficiency = [t1_strong / (p * strong_data[p]) for p in strong_procs]
# Weak scaling: efficiency = T(1)/T(p)
# S(p) = p*T(s)/T(p), efficiency = S(p)/p = T(1)/T(p)
weak_procs      = sorted(weak_data)
t1_weak         = weak_data[1]
weak_efficiency = [t1_weak / weak_data[p] for p in weak_procs]


# Plot Figure 1: STRONG SCALING
fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Strong Scaling: N-body Simulation (N = 5000 bodies, fixed)",
             **FONT_TITLE, y=1.01)
# Left panel: Speedup (Amdahl y = x, T(1)/T(p))
ax_sp.plot(strong_procs, ideal_speedup,
           color=IDEAL_COLOR, linewidth=1.5, linestyle="--",
           label="Ideal (y = p)", zorder=1)
ax_sp.plot(strong_procs, strong_speedup,
           color=MEASURED_COLOR, linewidth=2, linestyle="-",
           marker="o", markersize=7, markerfacecolor="white",
           markeredgewidth=2, label="Measured", zorder=2)
for p, s in zip(strong_procs, strong_speedup):
    ax_sp.annotate(f"{s:.2f}x",
                   xy=(p, s), xytext=(4, 6), textcoords="offset points",
                   fontsize=9, color=MEASURED_COLOR)
ax_sp.set_xscale("log", base=2)
ax_sp.set_yscale("log", base=2)
ax_sp.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax_sp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax_sp.set_xticks(strong_procs)
ax_sp.set_yticks(ideal_speedup)
ax_sp.set_xlabel("MPI Processes", **FONT_LABEL)
ax_sp.set_ylabel("Speedup  T(1) / T(p)", **FONT_LABEL)
ax_sp.set_title("Speedup", fontsize=12)
ax_sp.tick_params(**FONT_TICK)
ax_sp.legend(fontsize=10, framealpha=0.9)
ax_sp.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
# Right panel: Efficiency T(1)/(p*T(p))
ax_eff.axhline(100, color=IDEAL_COLOR, linewidth=1.5, linestyle="--",
               label="Ideal (100%)", zorder=1)
ax_eff.plot(strong_procs, [e * 100 for e in strong_efficiency],
            color=MEASURED_COLOR, linewidth=2, linestyle="-",
            marker="o", markersize=7, markerfacecolor="white",
            markeredgewidth=2, label="Measured", zorder=2)
for p, e in zip(strong_procs, strong_efficiency):
    ax_eff.annotate(f"{e*100:.1f}%",
                    xy=(p, e * 100), xytext=(4, 6), textcoords="offset points",
                    fontsize=9, color=MEASURED_COLOR)
ax_eff.set_xscale("log", base=2)
ax_eff.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax_eff.set_xticks(strong_procs)
ax_eff.set_ylim(0, 120)
ax_eff.set_xlabel("MPI Processes", **FONT_LABEL)
ax_eff.set_ylabel("Efficiency  T(1) / (p · T(p))  (%)", **FONT_LABEL)
ax_eff.set_title("Parallel Efficiency", fontsize=12)
ax_eff.tick_params(**FONT_TICK)
ax_eff.legend(fontsize=10, framealpha=0.9)
ax_eff.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
fig.tight_layout()
fig.savefig(OUT_STRONG, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {OUT_STRONG}")
plt.close(fig)


# Plot Figure 2: WEAK SCALING
fig, ax = plt.subplots(figsize=FIG_SIZE)
# Ideal line
ax.axhline(100, color=IDEAL_COLOR, linewidth=1.5, linestyle="--",
           label="Ideal (100%)", zorder=1)
# Measured efficiency
ax.plot(weak_procs, [e * 100 for e in weak_efficiency],
        color=MEASURED_COLOR, linewidth=2, linestyle="-",
        marker="o", markersize=7, markerfacecolor="white",
        markeredgewidth=2, label="Measured efficiency", zorder=2)
# Annotate each point
for p, e in zip(weak_procs, weak_efficiency):
    ax.annotate(f"{e*100:.1f}%",
                xy=(p, e * 100), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=MEASURED_COLOR)
# Secondary x-axis showing N
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(weak_procs)
ax_top.set_xticklabels([f"N={p*1000:,}" for p in weak_procs], fontsize=8)
ax_top.set_xlabel("Problem size", fontsize=10)
# Set x-axis, y-axis, title, and legend
ax.set_xlabel("MPI Processes", **FONT_LABEL)
ax.set_ylabel("Parallel Efficiency  (%)", **FONT_LABEL)
ax.set_xticks(weak_procs)
ax.set_xticklabels(weak_procs)
ax.set_ylim(0, 120)
ax.tick_params(**FONT_TICK)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_title("Weak Scaling: N-body Simulation (1000 bodies per process)", **FONT_TITLE)
# Save figure
fig.tight_layout()
fig.savefig(OUT_WEAK, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {OUT_WEAK}")
plt.close(fig)

