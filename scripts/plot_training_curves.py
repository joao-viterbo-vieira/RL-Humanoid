#!/usr/bin/env python3
"""Publication-quality S1--S4 training-curve figures for the paper.

Each figure plots `rollout/ep_rew_mean` vs. `time/total_timesteps` from the
corresponding run's `progress.csv` in `outputs_best/`. Styling targets an IEEE
two-column paper rather than a TensorBoard screenshot:

  * serif (Times-like) typography matching the paper body text;
  * proper axis labels with units, no TensorBoard "metric" title (the LaTeX
    \\caption already names the metric);
  * a validated, colorblind-safe categorical palette (one hue per task);
  * a faded raw trace under an EMA-smoothed line, so both variance and trend
    are visible and honest;
  * vector PDF output (crisp at any print size) plus PNG for quick preview.

Usage:
    python scripts/plot_training_curves.py [--smoothing 0.6] [--outdir figures]
                                           [--zero-baseline]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Root of the repository (this file lives in <root>/scripts/).
ROOT = Path(__file__).resolve().parents[1]

# One validated, colorblind-safe hue per task (see scripts/validate_palette.js in
# the dataviz skill: passes lightness band, chroma floor, CVD separation, and
# 3:1 contrast on a light surface). Assigned in fixed S1->S4 order.
RUNS = [
    {
        "key": "S1",
        "label": "S1 – Flat locomotion",
        "run": "outputs_best/2025-10-28/17-47-25",  # env=humanoid, 50M
        "color": "#0F62A8",  # blue
        "stem": "S1_Baseline",
    },
    {
        "key": "S2",
        "label": "S2 – Stair climbing",
        "run": "outputs_best/2025-12-06/17-36-50",  # env=humanoid_stairs_easy, 30M
        "color": "#C64600",  # vermillion
        "stem": "S2_Stairs",
    },
    {
        "key": "S3",
        "label": "S3 – Circuit (flat)",
        "run": "outputs_best/2025-12-23/14-37-50",  # env=humanoid_circuit_flat, 80M
        "color": "#0E8F6E",  # teal-green
        "stem": "S3_Circuit",
    },
    {
        "key": "S4",
        "label": "S4 – Circuit + stairs",
        "run": "outputs_best/2025-12-24/16-29-37",  # env=humanoid_circuit_easy, 80M
        "color": "#9B4F9E",  # purple
        "stem": "S4_Circuit_Stairs",
    },
]

X_COL = "time/total_timesteps"
Y_COL = "rollout/ep_rew_mean"

X_LABEL = "Environment steps"
Y_LABEL = "Mean episode reward"

# Recessive, print-friendly ink/lines.
GRID = "#DCDCDC"
SPINE = "#BFBFBF"
INK = "#222222"


def set_style():
    """Paper-oriented Matplotlib style: serif type, thin recessive furniture."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": 12,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,   # embed TrueType so glyphs are editable/searchable
        "ps.fonttype": 42,
    })


def ema_smooth(values, weight):
    """TensorBoard-style exponential moving average with debiasing."""
    values = np.asarray(values, dtype=float)
    smoothed = np.empty_like(values)
    last = 0.0
    debias = 0.0
    for i, v in enumerate(values):
        last = last * weight + (1.0 - weight) * v
        debias = debias * weight + (1.0 - weight)
        smoothed[i] = last / debias
    return smoothed


def millions_formatter(x, _pos):
    """Format an x tick (env steps) as 0, 10M, 20M, ..."""
    if x == 0:
        return "0"
    return f"{x / 1e6:g}M"


def load_run(run_dir):
    df = pd.read_csv(ROOT / run_dir / "progress.csv")
    mask = df[X_COL].notna() & df[Y_COL].notna()
    x = df.loc[mask, X_COL].to_numpy()
    y = df.loc[mask, Y_COL].to_numpy()
    order = np.argsort(x)
    return x[order], y[order]


def draw_curve(ax, cfg, x, y, y_smooth, zero_baseline=True,
               title=None, xlabel=True, ylabel=True):
    """Draw one publication-style reward curve onto the given axes."""
    color = cfg["color"]
    # Faded raw trace behind the smoothed line: shows episode-to-episode variance.
    ax.plot(x, y, color=color, alpha=0.22, linewidth=0.8, zorder=1)
    # Solid EMA-smoothed line on top: the trend.
    ax.plot(x, y_smooth, color=color, alpha=1.0, linewidth=1.6, zorder=2)

    # Recessive grid + axes; drop top/right spines.
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    # Detailed y-axis: ~5 intervals at round steps, thousands-separated labels
    # matching how reward values are written in the paper prose.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.tick_params(length=0)

    # Both axes anchored at 0: x from training start, y from a zero reward
    # baseline so the curve rises from the bottom-left origin. Pass --autoscale-y
    # to instead zoom the y-axis to the reward range.
    ax.set_xlim(left=0, right=float(np.max(x)))
    ymin = min(float(np.min(y)), float(np.min(y_smooth)))
    ymax = max(float(np.max(y)), float(np.max(y_smooth)))
    if zero_baseline:
        ax.set_ylim(bottom=0, top=ymax * 1.05)
    else:
        pad = (ymax - ymin) * 0.06
        ax.set_ylim(bottom=ymin - pad, top=ymax + pad)

    if xlabel:
        ax.set_xlabel(X_LABEL)
    if ylabel:
        ax.set_ylabel(Y_LABEL)
    if title:
        ax.set_title(title, fontsize=12, color=INK, pad=5)


def save(fig, outdir, stem):
    """Write both a vector PDF (for the paper) and a PNG (for preview)."""
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p)
    return paths


def plot_run(cfg, smoothing, outdir, zero_baseline=True):
    x, y = load_run(cfg["run"])
    y_smooth = ema_smooth(y, smoothing)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    draw_curve(ax, cfg, x, y, y_smooth, zero_baseline=zero_baseline)
    fig.tight_layout(pad=0.4)
    paths = save(fig, outdir, cfg["stem"])
    plt.close(fig)

    print(f"{cfg['key']}: {cfg['stem']}.(pdf|png)  "
          f"steps=[0, {x.max()/1e6:g}M]  peak_raw={y.max():.0f}  "
          f"final_smoothed={y_smooth[-1]:.0f}")
    return paths


def plot_combined(smoothing, outdir, zero_baseline=True, stem="combined_training_curves"):
    """All four tasks as one 2x2 panel (compact comparison figure)."""
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.6))
    for i, (cfg, ax) in enumerate(zip(RUNS, axes.flat)):
        x, y = load_run(cfg["run"])
        y_smooth = ema_smooth(y, smoothing)
        row, col = divmod(i, 2)
        draw_curve(
            ax, cfg, x, y, y_smooth, zero_baseline=zero_baseline,
            title=cfg["label"],
            xlabel=(row == 1),   # x label only on the bottom row
            ylabel=False,        # shared y label via fig.supylabel below
        )
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.9, rect=(0.025, 0, 1, 1))
    fig.supylabel(Y_LABEL, fontsize=12, color=INK, x=0.012)
    save(fig, outdir, stem)
    plt.close(fig)
    print(f"combined: {stem}.(pdf|png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoothing", type=float, default=0.6,
                    help="TensorBoard EMA smoothing weight in [0,1) (default 0.6)")
    ap.add_argument("--outdir", type=str, default="figures",
                    help="Output directory for the figures (default: figures/)")
    ap.add_argument("--autoscale-y", action="store_true",
                    help="Zoom the y-axis to the reward range instead of "
                         "anchoring it at 0 (default: 0 baseline on both axes)")
    args = ap.parse_args()

    zero_baseline = not args.autoscale_y
    set_style()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for cfg in RUNS:
        plot_run(cfg, args.smoothing, outdir, zero_baseline)
    plot_combined(args.smoothing, outdir, zero_baseline)

    print(f"\nWrote {len(RUNS)} individual figures + 1 combined panel "
          f"(PDF + PNG) to {outdir}")


if __name__ == "__main__":
    main()
