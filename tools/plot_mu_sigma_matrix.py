"""High-resolution PDF matrix plot of the a*mu + b*(sigma*eps) rFID sweep
(results/mu_sigma_sweep/combined.csv).

Confusion-matrix layout: x = a (mu multiplier), y = b (sigma*eps multiplier), and each
cell holds that (a, b) pair's rFID -- one 8x8 matrix per checkpoint. Each panel carries
its OWN colorbar because the ranges are not comparable: fsq_E spans 0.586-8.60 while the
other three stay inside 0.40-0.51, so a shared scale would flatten three panels into a
single flat block. Cell text is the actual number, so color is never the only encoding.

Usage:
    python tools/plot_mu_sigma_matrix.py \
        --csv results/mu_sigma_sweep/combined.csv \
        --out results/mu_sigma_sweep/matrix.pdf
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

LABELS = {
    "checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305": "DualVAE — FSQ 0.1",
    "checkpoints/dualvae/VQ_0.1/dualvae_20260805-134849_09061b": "DualVAE — VQ 0.1",
    "checkpoints/dualvae/fsq_E/dualvae_20260813-204926_f85dfe": "DualVAE — FSQ-E",
    "checkpoints/vae/vae_20260810-165000_c9374d": "Plain VAE",
}
ORDER = list(LABELS.keys())
MULT = [1.000, 0.975, 0.950, 0.900, 0.850, 0.800, 0.750, 0.700]


def tick(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/mu_sigma_sweep/combined.csv")
    p.add_argument("--out", default="results/mu_sigma_sweep/matrix.pdf")
    args = p.parse_args()

    by_ckpt = defaultdict(dict)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_ckpt[row["checkpoint_dir"]][(float(row["a"]), float(row["b"]))] = float(row["rfid"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#4a5568",
        "axes.labelcolor": "#1a202c",
        "text.color": "#1a202c",
        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    fig.suptitle(
        "rFID over the (a, b) grid  —  z = z_vq + a·μ + b·(σε)",
        fontsize=17, fontweight="bold", y=0.975,
    )
    fig.text(0.5, 0.945,
             "full ImageNet-1k val split, N = 50,000  ·  x = a (μ multiplier)  ·  y = b (σε multiplier)  "
             "·  cell = rFID, lower is better  ·  each panel has its own color scale",
             ha="center", fontsize=11, color="#4a5568")

    for ax, key in zip(axes.flat, ORDER):
        cells = by_ckpt[key]
        # rows = b (top row is b=1.000), cols = a (left col is a=1.000)
        M = np.array([[cells[(a, b)] for a in MULT] for b in MULT])

        im = ax.imshow(M, cmap="Blues", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("rFID", fontsize=9)

        ax.set_xticks(range(len(MULT)), [tick(v) for v in MULT])
        ax.set_yticks(range(len(MULT)), [tick(v) for v in MULT])
        ax.set_xlabel("a  (μ multiplier)", fontsize=11)
        ax.set_ylabel("b  (σε multiplier)", fontsize=11)
        ax.set_title(LABELS[key], fontsize=13, fontweight="bold", loc="left", pad=10)

        vmin, vmax = M.min(), M.max()
        best = np.unravel_index(np.argmin(M), M.shape)   # (row=b, col=a)

        for i in range(len(MULT)):
            for j in range(len(MULT)):
                v = M[i, j]
                t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if t > 0.55 else "#1a202c")

        # best cell (lowest rFID) and the untouched baseline a = b = 1.0
        ax.add_patch(Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                               fill=False, edgecolor="#b8551f", linewidth=2.5, zorder=5))
        # Inset by a hair: at exactly (-0.5, -0.5) the top and left edges land on the axes
        # spine and only half the box is visible.
        ax.add_patch(Rectangle((-0.44, -0.44), 0.88, 0.88,
                               fill=False, edgecolor="#4a5568", linewidth=1.4,
                               linestyle=(0, (3, 2)), zorder=5))
        ax.text(best[1], best[0] + 0.38, "best", ha="center", va="center",
                fontsize=7, color="#b8551f", fontweight="bold", zorder=6)

    fig.text(0.5, 0.045,
             "solid orange = best cell in the grid          dashed grey = untouched baseline (a = b = 1.0)",
             ha="center", fontsize=10, color="#4a5568")

    fig.subplots_adjust(hspace=0.30, wspace=0.22, top=0.90, bottom=0.09, left=0.07, right=0.95)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
