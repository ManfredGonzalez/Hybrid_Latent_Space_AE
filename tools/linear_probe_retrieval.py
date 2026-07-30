#!/usr/bin/env python
"""
Downstream usefulness of the DUALVAE latent: linear probing + retrieval, frozen.

The question this answers: does the better-organized (Gaussian-mixture) latent help on
downstream tasks, or does it just look prettier? We FREEZE each trained model, pull a single
global feature vector per image, and measure two standard frozen-feature protocols on
Imagenette's 10 classes:

Everything is evaluated on the VALIDATION SET ONLY. The linear probe uses a stratified split
*within* val (fit on one part, score on the held-out part); retrieval is leave-one-out over val.

  * LINEAR PROBE -- fit a logistic-regression classifier on a stratified subset of the val
    features, report top-1 accuracy on the held-out val features. Tests linear separability.
  * RETRIEVAL mAP -- for each val image, rank all other val images by cosine similarity and
    score how well same-class images come first (mean Average Precision + precision@10).

Three feature variants per model (latent grid is C=8 channels x 32x32 locations):
  * z_sum   : mean-pooled full latent (z_vq + z_cont)          -> 8-d
  * z_vq    : mean-pooled discrete branch                       -> 8-d
  * z_cont  : mean-pooled continuous branch                     -> 8-d
  * bagcodes: normalized histogram of the 256 codebook entries  -> 256-d (a learned
              bag-of-visual-words descriptor; the natural retrieval feature for a GMM latent)

An optional standalone VAE (--vae-dir) is included as a baseline (mean-pooled 4-d latent only).
Labels come from noisy_imagenette.csv (column noisy_labels_0), which maps every flat filename
to its class regardless of naming (nXXXX_* or ILSVRC2012_val_*).

Outputs: two grouped bar charts (probe accuracy, retrieval mAP), a JSON report, and printed
tables --- written to --output-dir.

Example
-------
    python tools/linear_probe_retrieval.py \
        --checkpoint-dirs \
            checkpoints/dualvae/dualvae_20260724-121508_392b5b \
            checkpoints/dualvae/dualvae_20260724-133147_b03f09 \
            checkpoints/dualvae/dualvae_20260725-015430_5878cc \
            checkpoints/dualvae/dualvae_20260725-134828_ce6208 \
        --run-labels A B C D \
        --vae-dir "checkpoints/vae/VAE_betaKL@0.001@Downsample_8" \
        --output-dir reports/downstream
"""
import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.vae import VAE
from tools.dualvae_latent_analysis import (
    load_config, find_weights, build_dualvae, build_transform, VALID_EXTS,
)

FEATURES = ["z_sum", "z_vq", "z_cont", "bagcodes"]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def load_label_map(csv_path):
    """basename -> class label, from noisy_imagenette.csv (noisy_labels_0 = clean label)."""
    m = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            base = os.path.basename(row["path"])
            m[base] = row["noisy_labels_0"]
    return m


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_dualvae(model, paths, transform, device, batch_size, num_embeddings):
    """Per-image features: mean-pooled z_sum/z_vq/z_cont (C-d each) and a 256-d code histogram."""
    model.eval()
    ds = model.downsample_factor
    C = model.latent_channels
    feats = {k: [] for k in FEATURES}
    for start in range(0, len(paths), batch_size):
        imgs = torch.stack([transform(Image.open(p).convert("RGB"))
                            for p in paths[start:start + batch_size]]).to(device)
        b, _, hh, ww = imgs.shape
        z_e = model.encoder(imgs)
        z_e_vq = model.bottle_neck_VQ(z_e)
        z_vq, _, code_idx, _, _ = model.vq_layer(z_e_vq)     # code_idx: (b*h*w,) depth-1
        if model.residual_continuous:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
        else:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e)
        torch.manual_seed(0)
        noise = torch.randn((b, C, hh // ds, ww // ds), device=device)
        z_cont, _, _ = model.forward_vanilla_z(z_e_vanilla, noise)
        z_sum = z_vq + z_cont

        feats["z_sum"].append(z_sum.mean(dim=(2, 3)).cpu().numpy())
        feats["z_vq"].append(z_vq.mean(dim=(2, 3)).cpu().numpy())
        feats["z_cont"].append(z_cont.mean(dim=(2, 3)).cpu().numpy())

        # Bag-of-codes: per-image normalized histogram over the codebook.
        n_loc = (hh // ds) * (ww // ds)
        codes = code_idx.view(b, n_loc)
        hist = torch.zeros(b, num_embeddings, device=device)
        hist.scatter_add_(1, codes, torch.ones_like(codes, dtype=hist.dtype))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1)
        feats["bagcodes"].append(hist.cpu().numpy())

    return {k: np.concatenate(v, 0) for k, v in feats.items()}


@torch.no_grad()
def extract_vae(model, paths, transform, device, batch_size):
    """Baseline: mean-pooled VAE posterior mean (4-d)."""
    model.eval()
    out = []
    for start in range(0, len(paths), batch_size):
        imgs = torch.stack([transform(Image.open(p).convert("RGB"))
                            for p in paths[start:start + batch_size]]).to(device)
        b, _, hh, ww = imgs.shape
        torch.manual_seed(0)
        noise = torch.randn((b, model.latent_channels,
                             hh // model.downsample_factor, ww // model.downsample_factor),
                            device=device)
        _, mean, _ = model.encoder(imgs, noise)
        out.append(mean.mean(dim=(2, 3)).cpu().numpy())
    return {"z_sum": np.concatenate(out, 0)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def fit_probe(X, y, tr_idx, te_idx, seed):
    """Fit a logistic-regression probe on X[tr_idx], report top-1 accuracy on X[te_idx]."""
    scaler = StandardScaler().fit(X[tr_idx])
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed, n_jobs=-1)
    clf.fit(scaler.transform(X[tr_idx]), y[tr_idx])
    return float((clf.predict(scaler.transform(X[te_idx])) == y[te_idx]).mean())


def retrieval_map(X, y, k=10):
    """Leave-one-out cosine retrieval over X. Returns (mAP, precision@k)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)              # exclude self
    n = len(y)
    order = np.argsort(-S, axis=1)            # ranked gallery per query
    rel = (y[order] == y[:, None])            # (n, n-1 effective) relevance
    # Average precision per query.
    csum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, n + 1)[None, :]
    precision_at = csum / ranks
    total_rel = rel.sum(axis=1)
    ap = (precision_at * rel).sum(axis=1) / np.clip(total_rel, 1, None)
    p_at_k = rel[:, :k].mean(axis=1)
    valid = total_rel > 0
    return float(ap[valid].mean()), float(p_at_k[valid].mean())


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def grouped_bars(results, run_labels, metric_key, title, ylabel, baseline, out_path):
    feats = FEATURES
    x = np.arange(len(feats))
    w = 0.8 / len(run_labels)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, rl in enumerate(run_labels):
        vals = [results.get(rl, {}).get(f, {}).get(metric_key, np.nan) for f in feats]
        ax.bar(x + i * w, vals, w, label=rl, color=cmap(i), edgecolor="black", linewidth=0.3)
        for xx, v in zip(x + i * w, vals):
            if not np.isnan(v):
                ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    if baseline is not None:
        ax.axhline(baseline, color="crimson", ls="--", lw=1.2,
                   label=f"chance ({baseline:.2f})")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(feats)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def collect_split(images_dir, label_map):
    files, labels = [], []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(VALID_EXTS) and f in label_map:
            files.append(os.path.join(images_dir, f))
            labels.append(label_map[f])
    return files, np.array(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dirs", nargs="+", required=True)
    ap.add_argument("--run-labels", nargs="+", default=None,
                    help="Short names for the checkpoints (default: A, B, C, ...).")
    ap.add_argument("--vae-dir", default=None, help="Optional standalone-VAE baseline.")
    ap.add_argument("--dataset-dir",
                    default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320")
    ap.add_argument("--csv", default=None,
                    help="noisy_imagenette.csv (default: <dataset-dir>/noisy_imagenette.csv).")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--probe-test-frac", type=float, default=0.4,
                    help="Fraction of the val set held out to score the linear probe.")
    ap.add_argument("--output-dir", default="reports/downstream")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device: {device}")

    csv_path = args.csv or os.path.join(args.dataset_dir, "noisy_imagenette.csv")
    label_map = load_label_map(csv_path)
    classes = sorted(set(label_map.values()))
    print(f"[info] {len(classes)} classes from {csv_path}")

    # VALIDATION SET ONLY. The linear probe uses a stratified split *within* val
    # (fit on one part, score on the held-out part); retrieval is leave-one-out over all val.
    va_files, yva = collect_split(os.path.join(args.dataset_dir, "val"), label_map)
    rng = np.random.default_rng(args.seed)
    if args.max_val and args.max_val < len(va_files):
        idx = rng.permutation(len(va_files))[:args.max_val]
        va_files = [va_files[i] for i in idx]; yva = yva[idx]
    print(f"[info] val={len(va_files)} images (val-only evaluation)")

    # One shared stratified probe split, reused for every model/feature (fair comparison).
    all_idx = np.arange(len(va_files))
    tr_idx, te_idx = train_test_split(all_idx, test_size=args.probe_test_frac,
                                      stratify=yva, random_state=args.seed)
    print(f"[info] probe split: fit={len(tr_idx)}  test={len(te_idx)} (stratified, val-internal)")

    # Baselines for context (computed on the probe-test labels / full-val labels).
    _, counts_te = np.unique(yva[te_idx], return_counts=True)
    chance_acc = float(counts_te.max() / counts_te.sum())        # majority-class accuracy
    _, counts_all = np.unique(yva, return_counts=True)
    priors = counts_all / counts_all.sum()
    chance_map = float((priors ** 2).sum() / priors.sum())       # ~random-retrieval mAP

    run_labels = args.run_labels or [chr(ord("A") + i) for i in range(len(args.checkpoint_dirs))]
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    def eval_features(fva, tag, feats):
        results[tag] = {}
        for feat in feats:
            acc = fit_probe(fva[feat], yva, tr_idx, te_idx, args.seed)   # val-internal split
            mp, pk = retrieval_map(fva[feat], yva)                       # leave-one-out over val
            results[tag][feat] = {"probe_acc": acc, "retrieval_map": mp, "precision_at_10": pk}
            print(f"  [{tag}/{feat:<8}] probe_acc={acc:.4f}  mAP={mp:.4f}  P@10={pk:.4f}")

    for ck, rl in zip(args.checkpoint_dirs, run_labels):
        cfg, _ = load_config(ck)
        w = find_weights(ck)
        print(f"\n[==] Run {rl}: {ck}\n[info] weights: {w}")
        model = build_dualvae(cfg, device)
        sd = torch.load(w, map_location=device)
        model.load_state_dict(sd if "encoder.0.weight" in sd else sd["model_state_dict"])
        transform = build_transform(cfg.get("resize_img", 256), cfg.get("dataset_name", "imagenette"))
        nemb = cfg.get("num_embeddings", 256)
        fva = extract_dualvae(model, va_files, transform, device, args.batch_size, nemb)
        eval_features(fva, rl, FEATURES)

    if args.vae_dir:
        cfg, _ = (load_config(args.vae_dir) if os.path.exists(os.path.join(args.vae_dir, "config_used.yaml")) else ({}, None))
        w = find_weights(args.vae_dir)
        print(f"\n[==] VAE baseline: {args.vae_dir}\n[info] weights: {w}")
        vae = VAE(downsample_factor=cfg.get("downsample_factor", 8),
                  latent_channels=cfg.get("latent_channels", 4)).to(device)
        sd = torch.load(w, map_location=device)
        vae.load_state_dict(sd if "encoder.0.weight" in sd else sd["model_state_dict"])
        transform = build_transform(256, "imagenette")
        fva = extract_vae(vae, va_files, transform, device, args.batch_size)
        eval_features(fva, "VAE", ["z_sum"])
        run_labels = list(run_labels) + ["VAE"]

    report = {"classes": classes, "n_val": len(va_files),
              "probe_fit": len(tr_idx), "probe_test": len(te_idx),
              "chance_accuracy": chance_acc, "chance_retrieval_map": chance_map,
              "results": results}
    rp = os.path.join(args.output_dir, "downstream_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    grouped_bars(results, run_labels, "probe_acc",
                 "Linear-probe top-1 accuracy (Imagenette 10-way, frozen features)",
                 "val accuracy", chance_acc,
                 os.path.join(args.output_dir, "downstream_probe_acc.pdf"))
    grouped_bars(results, run_labels, "retrieval_map",
                 "Retrieval mAP (val, leave-one-out cosine)",
                 "mAP", chance_map,
                 os.path.join(args.output_dir, "downstream_retrieval_map.pdf"))

    _print_tables(results, run_labels, chance_acc, chance_map)
    print(f"\n[saved] {rp}\n[done]")


def _print_tables(results, run_labels, chance_acc, chance_map):
    for metric, name, base in [("probe_acc", "LINEAR PROBE top-1 accuracy", chance_acc),
                               ("retrieval_map", "RETRIEVAL mAP", chance_map)]:
        print("\n" + "=" * 66)
        print(f"{name}  (chance = {base:.3f})")
        print(f"{'run':<6}" + "".join(f"{f:>11}" for f in FEATURES))
        for rl in run_labels:
            row = f"{rl:<6}"
            for f in FEATURES:
                v = results.get(rl, {}).get(f, {}).get(metric)
                row += f"{v:>11.4f}" if v is not None else f"{'-':>11}"
            print(row)
        print("=" * 66)


if __name__ == "__main__":
    main()
