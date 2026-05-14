# plot_scaling_cuda.py

"""
Scaling plots for CUDA N-body simulation
Expected input files (relative to this script's location):
  ../../mpi/saved_run/cuda_timing_scale_<N>.csv

Each CSV has one header row and one data row:
  Program,N_Bodies,Time(s)
  cuda,<N>,<time>

Output:
  cuda_scaling.png
"""


# Imports
import glob
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Load all per-run CSVs
CSV_DIR = os.path.join(os.path.dirname(__file__), "../../cuda/saved_run")
pattern = os.path.join(CSV_DIR, "cuda_timing_scale_*.csv")
records = []
for path in glob.glob(pattern):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        records.append((int(row["N_Bodies"]), float(row["Time(s)"])))
records.sort()
ns = np.array([r[0] for r in records])
ts = np.array([r[1] for r in records])
print(f"{'N':>10}  {'Time(s)':>10}")
for n, t in zip(ns, ts):
    print(f"{n:>10}  {t:>10.2f}")

# Per-region power-law regression
# Power law equation: y = ax^b
# For our case, y is time and x is N bodies, so we want to regress t = a N^b
# On a log-log scale, t = a N^b becomes log(t) = log(a) + b*log(N),
# so a linear fit in log-log space gives the exponent b directly.
    # Interpreting b:
    #   b ~= 0    flat:                 GPU underutilised, extra bodies cost almost nothing
    #   b ~= 0.5  sub-linear:           GPU filling up, parallelism still absorbing some work
    #   b ~= 1    linear:               fully saturated, runtime grows proportionally with N
    #   b > 1     super-linear: O(N^2)  kernel work dominates, doubling N multiplies time by 2^b
def fit_slope(n_vals, t_vals):
    log_n = np.log(n_vals.astype(float))
    log_t = np.log(t_vals.astype(float))
    b, log_a = np.polyfit(log_n, log_t, 1)
    return np.exp(log_a), b  # t ~= a * N^b

# Based on a cursory reading of the data, these regions can be updated for new run scales
regions = [
    ("flat",             ns <= 1024,                   "green"),
    ("sub-linear",      (ns >= 1024) & (ns <= 8192),   "orange"),
    ("super-linear",     ns >= 8192,                   "red"),
]

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ns, ts, marker="o", color="#378ADD", linewidth=2, zorder=5, label="measured time (s)")

# For each region (flat, linear, super-linear)...
for label, mask, color in regions:
    n_reg = ns[mask]
    t_reg = ts[mask]
    # Shade region
    ax.axvspan(n_reg.min(), n_reg.max(), alpha=0.07, color=color)
    # Fit and draw regression line over the region
    a, b = fit_slope(n_reg, t_reg)
    n_smooth = np.linspace(n_reg.min(), n_reg.max(), 200)
    t_fit = a * n_smooth ** b
    ax.plot(n_smooth, t_fit, linestyle="--", color=color, linewidth=1.8,
            label=f"{label}: $N^{{{b:.2f}}}$")
    # Annotate slope at the geometric midpoint of the region
    mid_n = np.sqrt(n_reg.min() * n_reg.max())
    mid_t = a * mid_n ** b
    ax.annotate(f"$N^{{{b:.2f}}}$",
                xy=(mid_n, mid_t),
                xytext=(0, 22), textcoords="offset points",
                ha="center", fontsize=10, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

# Set axes, titesl, and legends
ax.set_xscale("log", base=2)
ax.set_xticks(ns)
ax.set_xticklabels([str(n) if n < 1024 else f"{n//1024}k" for n in ns], rotation=45, ha="right")
ax.set_xlabel("Number of bodies (N)")
ax.set_ylabel("Wall time (s)")
ax.set_title("CUDA N-body scaling: wall time vs body count")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

# Save the plot and show it
out_path = os.path.join(os.path.dirname(__file__), "cuda_scaling.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()

