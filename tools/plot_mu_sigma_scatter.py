"""High-resolution PDF scatter plot of the a*mu + b*(sigma*eps) rFID sweep
(results/mu_sigma_sweep/combined.csv). One panel per checkpoint, x=a, y=rFID,
point color=b -- each panel autoscales its own y-axis since fsq_E's rFID spans
up to 8.6 while the other three stay under 0.51.

Usage:
    python tools/plot_mu_sigma_scatter.py \
        --csv results/mu_sigma_sweep/combined.csv \
        --out results/mu_sigma_sweep/scatter.pdf
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

LABELS = {
    "checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305": "DualVAE — FSQ 0.1",
    "checkpoints/dualvae/VQ_0.1/dualvae_20260805-134849_09061b": "DualVAE — VQ 0.1",
    "checkpoints/dualvae/fsq_E/dualvae_20260813-204926_f85dfe": "DualVAE — FSQ-E",
    "checkpoints/vae/vae_20260810-165000_c9374d": "Plain VAE",
}
ORDER = list(LABELS.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/mu_sigma_sweep/combined.csv")
    p.add_argument("--out", default="results/mu_sigma_sweep/scatter.pdf")
    args = p.parse_args()

    by_ckpt = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_ckpt[row["checkpoint_dir"]].append(
                (float(row["a"]), float(row["b"]), float(row["rfid"]))
            )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#4a5568",
        "axes.labelcolor": "#1a202c",
        "text.color": "#1a202c",
        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle(
        "rFID vs. inference-time latent multipliers  —  z = z_vq + a·μ + b·(σε)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(0.5, 0.965,
              "full ImageNet-1k val split, N = 50,000  ·  color = b (σε multiplier)  ·  each panel scaled to its own rFID range",
              ha="center", fontsize=10, color="#4a5568")

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.7, vmax=1.0)

    for ax, key in zip(axes.flat, ORDER):
        pts = by_ckpt[key]
        a_vals = [p_[0] for p_ in pts]
        b_vals = [p_[1] for p_ in pts]
        rfid_vals = [p_[2] for p_ in pts]

        sc = ax.scatter(a_vals, rfid_vals, c=b_vals, cmap=cmap, norm=norm,
                         s=55, edgecolors="white", linewidths=0.5, zorder=3)

        best_i = min(range(len(pts)), key=lambda i: rfid_vals[i])
        ax.scatter([a_vals[best_i]], [rfid_vals[best_i]], s=180,
                   facecolors="none", edgecolors="#b8551f", linewidths=1.8, zorder=4)
        ax.annotate(f"best {rfid_vals[best_i]:.3f}\n(a={a_vals[best_i]:.2f}, b={b_vals[best_i]:.2f})",
                    (a_vals[best_i], rfid_vals[best_i]),
                    textcoords="offset points", xytext=(10, 10), fontsize=8.5, color="#7c3b13")

        ax.set_title(LABELS[key], fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel("a  (μ multiplier)")
        ax.set_ylabel("rFID")
        ax.invert_xaxis()  # 1.0 -> 0.7 left to right, matching the sweep's own ordering
        ticks = sorted(set(a_vals), reverse=True)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.3f}".rstrip("0").rstrip(".") for t in ticks],
                            rotation=45, ha="right")

    # Lay the panels out FIRST, then drop the colorbar into a reserved strip on the right --
    # a colorbar created with ax=axes fixes its position from the axes' pre-adjust geometry,
    # so a later subplots_adjust leaves it sitting on top of the panels.
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.90, bottom=0.07, left=0.08, right=0.87)
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.60])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="b  (σε multiplier)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
