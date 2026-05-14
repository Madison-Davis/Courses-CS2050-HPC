# plot_scaling_openmp.py 

"""
Strong & Weak Scaling plots for OpenMP N-body simulation
Expected input files (relative to this script's location):
  ../openmp/saved_run/openmp_timing_strongscale_<T>.csv
  ../openmp/saved_run/openmp_timing_weakscale_<T>.csv

Each CSV has one header row and one data row:
  Program,Threads,N_Bodies,Time(s)
  openmp,<T>,<N>,<time>

Outputs (written next to this script):
  openmp_strong_scaling.png
  openmp_weak_scaling.png
"""


# Imports
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Path Variables
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "..", "openmp", "saved_run")
OUT_STRONG   = os.path.join(SCRIPT_DIR, "openmp_strong_scaling.png")
OUT_WEAK     = os.path.join(SCRIPT_DIR, "openmp_weak_scaling.png")
THREADS_LIST = [1, 2, 4, 8, 16]


# Plot Style Variables
MEASURED_COLOR = "#4C72B0"
IDEAL_COLOR    = "#cccccc"
FONT_TITLE     = dict(fontsize=14, fontweight="bold")
FONT_LABEL     = dict(fontsize=12)
FONT_TICK      = dict(labelsize=10)
FIG_SIZE       = (6.5, 5)
DPI            = 150


# Helper Function: Load the Timing Files
def load_timing(prefix):
    """Return dict {threads: time_seconds} for the given file prefix."""
    data = {}
    for t in THREADS_LIST:
        path = os.path.join(DATA_DIR, f"{prefix}_{t}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping T={t}")
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[int(row["Threads"])] = float(row["Time(s)"])
    return data


# Load the Data
print("Loading timing data...")
strong_data = load_timing("openmp_timing_strongscale")
weak_data   = load_timing("openmp_timing_weakscale")
if not strong_data:
    raise FileNotFoundError(f"No strong-scaling CSVs found in {DATA_DIR}")
if not weak_data:
    raise FileNotFoundError(f"No weak-scaling CSVs found in {DATA_DIR}")


# Derived Quantities
# Strong scaling: efficiency = T(1) / (p * T(p))
# S(p) = T(s)/T(p), efficiency = S(p)/p = T(1)/(p*T(p))
strong_threads    = sorted(strong_data)
t1_strong         = strong_data[1]
strong_speedup    = [t1_strong / strong_data[t] for t in strong_threads]
ideal_speedup     = [float(t) for t in strong_threads]
strong_efficiency = [t1_strong / (t * strong_data[t]) for t in strong_threads]
# Weak scaling: efficiency = T(1) / T(p)
# S(p) = p*T(s)/T(p), efficiency = S(p)/p = T(1)/T(p)
weak_threads    = sorted(weak_data)
t1_weak         = weak_data[1]
weak_efficiency = [t1_weak / weak_data[t] for t in weak_threads]


# Plot Figure 1: STRONG SCALING
fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Strong Scaling: N-body Simulation (N = 5000 bodies, fixed)",
             **FONT_TITLE, y=1.01)
# Left panel: Speedup (Amdahl y = x, T(1)/(T(p))
ax_sp.plot(strong_threads, ideal_speedup,
           color=IDEAL_COLOR, linewidth=1.5, linestyle="--",
           label="Ideal (y = p)", zorder=1)
ax_sp.plot(strong_threads, strong_speedup,
           color=MEASURED_COLOR, linewidth=2, linestyle="-",
           marker="o", markersize=7, markerfacecolor="white",
           markeredgewidth=2, label="Measured", zorder=2)
for t, s in zip(strong_threads, strong_speedup):
    ax_sp.annotate(f"{s:.2f}x",
                   xy=(t, s), xytext=(4, 6), textcoords="offset points",
                   fontsize=9, color=MEASURED_COLOR)
ax_sp.set_xscale("log", base=2)
ax_sp.set_yscale("log", base=2)
ax_sp.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax_sp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax_sp.set_xticks(strong_threads)
ax_sp.set_yticks(ideal_speedup)
ax_sp.set_xlabel("OpenMP Threads", **FONT_LABEL)
ax_sp.set_ylabel("Speedup  T(1) / T(p)", **FONT_LABEL)
ax_sp.set_title("Speedup", fontsize=12)
ax_sp.tick_params(**FONT_TICK)
ax_sp.legend(fontsize=10, framealpha=0.9)
ax_sp.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
# Right panel: Efficiency T(1)/(p*T(p))
ax_eff.axhline(100, color=IDEAL_COLOR, linewidth=1.5, linestyle="--",
               label="Ideal (100%)", zorder=1)
ax_eff.plot(strong_threads, [e * 100 for e in strong_efficiency],
            color=MEASURED_COLOR, linewidth=2, linestyle="-",
            marker="o", markersize=7, markerfacecolor="white",
            markeredgewidth=2, label="Measured", zorder=2)
for t, e in zip(strong_threads, strong_efficiency):
    ax_eff.annotate(f"{e*100:.1f}%",
                    xy=(t, e * 100), xytext=(4, 6), textcoords="offset points",
                    fontsize=9, color=MEASURED_COLOR)
ax_eff.set_xscale("log", base=2)
ax_eff.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax_eff.set_xticks(strong_threads)
ax_eff.set_ylim(0, 120)
ax_eff.set_xlabel("OpenMP Threads", **FONT_LABEL)
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
ax.plot(weak_threads, [e * 100 for e in weak_efficiency],
        color=MEASURED_COLOR, linewidth=2, linestyle="-",
        marker="o", markersize=7, markerfacecolor="white",
        markeredgewidth=2, label="Measured efficiency", zorder=2)
# Annotate each point
for t, e in zip(weak_threads, weak_efficiency):
    ax.annotate(f"{e*100:.1f}%",
                xy=(t, e * 100), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=MEASURED_COLOR)
# Secondary x-axis showing N
ax3 = ax.twiny()
ax3.set_xlim(ax.get_xlim())
ax3.set_xticks(weak_threads)
ax3.set_xticklabels([f"N={t*1000:,}" for t in weak_threads], fontsize=8)
ax3.set_xlabel("Problem size", fontsize=10)
# Set x-axis, y-axis, title, and legend
ax.set_xlabel("OpenMP Threads", **FONT_LABEL)
ax.set_ylabel("Parallel Efficiency  (%)", **FONT_LABEL)
ax.set_xticks(weak_threads)
ax.set_xticklabels(weak_threads)
ax.set_ylim(0, 120)
ax.tick_params(**FONT_TICK)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_title("Weak Scaling: N-body Simulation (1000 bodies per thread)", **FONT_TITLE)
# Save figure
fig.tight_layout()
fig.savefig(OUT_WEAK, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {OUT_WEAK}")
plt.close(fig)