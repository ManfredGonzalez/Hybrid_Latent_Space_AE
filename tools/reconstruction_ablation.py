#!/usr/bin/env python
"""
Reconstruction-quality branch ablation for a trained DUALVAE.

Runs the val set through the model three ways and measures reconstruction quality, so
you can see how much each branch actually contributes to the image:

    full         (ablation_mode=-1) : both branches active
    vq_only      (ablation_mode= 0) : continuous branch zeroed  (codes only)
    vanilla_only (ablation_mode= 1) : VQ branch zeroed          (continuous only)

The gap between `full` and `vq_only` is exactly the reconstruction value added by the
continuous branch (a branch that had collapsed would show no gap). Metrics: MSE, PSNR,
SSIM, LPIPS -- all computed in denormalized [0,1] pixel space.

Outputs: a metrics bar chart (PDF), a qualitative grid original|full|vq_only|vanilla_only
(PDF), and a JSON report.

Example
-------
    python tools/reconstruction_ablation.py \
        --checkpoint-dir checkpoints/dualvae/dualvae_20260724-133147_b03f09 \
        --num-images 500
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips as lpips_lib

from tools.dualvae_latent_analysis import (
    load_config, find_weights, build_dualvae, build_transform, VALID_EXTS,
)

MODES = {"full": -1, "vq_only": 0, "vanilla_only": 1}
MODE_COLORS = {"full": "#3b7dd8", "vq_only": "#c0567a", "vanilla_only": "#5aa469"}


def sample_image_paths(images_dir, num_images, seed):
    files = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir)
                   if f.lower().endswith(VALID_EXTS))
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}.")
    rng = random.Random(seed)
    if num_images and num_images < len(files):
        files = rng.sample(files, num_images)
    return files


def denorm(t, dataset_name):
    if str(dataset_name).lower() == "imagenette":
        return (t * 0.5 + 0.5).clamp(0, 1)
    if str(dataset_name).lower() == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=t.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=t.device).view(1, 3, 1, 1)
        return (t * std + mean).clamp(0, 1)
    return t.clamp(0, 1)


@torch.no_grad()
def evaluate(model, paths, transform, device, batch_size, dataset_name,
             compute_fid=True, kid_subset=100):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    # Reconstruction-FID / KID between real val images and the FULL-model reconstructions
    # (the meaningful perceptual metric; degenerate ablation modes are skipped for FID).
    fid = kid = None
    if compute_fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        fid = FrechetInceptionDistance(normalize=True).to(device)
        kid = KernelInceptionDistance(subset_size=kid_subset, normalize=True).to(device)

    acc = {m: {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0} for m in MODES}
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        real = denorm(imgs, dataset_name)
        for m, code in MODES.items():
            torch.manual_seed(1234)  # same reparam noise across modes for a fair comparison
            recon = model(imgs, ablation_mode=code)[0]
            fake = denorm(recon, dataset_name)
            bs = imgs.size(0)
            acc[m]["mse"] += F.mse_loss(fake, real, reduction="mean").item() * bs
            acc[m]["psnr"] += psnr(fake, real).item() * bs
            acc[m]["ssim"] += ssim(fake, real).item() * bs
            acc[m]["lpips"] += lpips_fn(fake * 2 - 1, real * 2 - 1).mean().item() * bs
            acc[m]["n"] += bs
            if fid is not None and m == "full":
                fid.update(real.clamp(0, 1), real=True)
                fid.update(fake.clamp(0, 1), real=False)
                kid.update(real.clamp(0, 1), real=True)
                kid.update(fake.clamp(0, 1), real=False)

    metrics = {}
    for m in MODES:
        n = acc[m]["n"]
        metrics[m] = {k: acc[m][k] / n for k in ("mse", "psnr", "ssim", "lpips")}
    if fid is not None:
        metrics["full"]["fid"] = float(fid.compute().item())
        kid_mean, _ = kid.compute()
        metrics["full"]["kid"] = float(kid_mean.item())
    return metrics


@torch.no_grad()
def qualitative_grid(model, paths, transform, device, dataset_name, out_path, tag, n_show=5):
    paths = paths[:n_show]
    cols = ["original"] + list(MODES.keys())
    fig, axes = plt.subplots(len(paths), len(cols), figsize=(len(cols) * 2.3, len(paths) * 2.35))
    if len(paths) == 1:
        axes = axes[None, :]
    for i, p in enumerate(paths):
        img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        real = denorm(img, dataset_name)[0].cpu().permute(1, 2, 0).numpy()
        panels = [real]
        for m, code in MODES.items():
            torch.manual_seed(1234)
            rec = denorm(model(img, ablation_mode=code)[0], dataset_name)[0].cpu()
            panels.append(rec.permute(1, 2, 0).numpy())
        for j, panel in enumerate(panels):
            axes[i, j].imshow(np.clip(panel, 0, 1))
            axes[i, j].axis("off")
            if i == 0:
                axes[0, j].set_title(cols[j], fontsize=10, fontweight="bold")
    fig.suptitle(f"Branch ablation reconstructions  |  {tag}", fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def metrics_bars(metrics, out_path, tag):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, better in zip(axes, ["psnr", "ssim", "lpips"],
                               ["higher better", "higher better", "lower better"]):
        modes = list(MODES.keys())
        vals = [metrics[m][key] for m in modes]
        ax.bar(modes, vals, color=[MODE_COLORS[m] for m in modes],
               edgecolor="black", linewidth=0.4)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{key.upper()}  ({better})", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"Reconstruction quality by branch ablation  |  {tag}",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-dir",
                   default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320")
    p.add_argument("--split", default="val")
    p.add_argument("--num-images", type=int, default=500,
                   help="How many val images to score (subset for speed; use -1 for all).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-fid", action="store_true", help="Skip reconstruction-FID/KID.")
    p.add_argument("--kid-subset", type=int, default=100)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device: {device}")

    cfg, _ = load_config(args.checkpoint_dir)
    weights = find_weights(args.checkpoint_dir)
    print(f"[info] model=dualvae  weights={weights}")
    model = build_dualvae(cfg, device)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    images_dir = os.path.join(args.dataset_dir, args.split)
    num = None if args.num_images == -1 else args.num_images
    paths = sample_image_paths(images_dir, num, args.seed)
    print(f"[info] scoring {len(paths)} images from {images_dir}")

    transform = build_transform(cfg.get("resize_img", 256),
                                cfg.get("dataset_name", "imagenette"))
    dataset_name = cfg.get("dataset_name", "imagenette")

    metrics = evaluate(model, paths, transform, device, args.batch_size, dataset_name,
                       compute_fid=not args.skip_fid, kid_subset=args.kid_subset)

    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    metrics_bars(metrics, os.path.join(out_dir, f"ablation_metrics_{tag}.pdf"), tag)
    qualitative_grid(model, paths, transform, device, dataset_name,
                     os.path.join(out_dir, f"ablation_grid_{tag}.pdf"), tag)

    # Contribution of the continuous branch = full - vq_only.
    contrib = {k: metrics["full"][k] - metrics["vq_only"][k]
               for k in ("psnr", "ssim", "lpips")}
    report = {"checkpoint_dir": args.checkpoint_dir, "n_images": len(paths),
              "metrics": metrics, "continuous_contribution_full_minus_vqonly": contrib}
    rp = os.path.join(out_dir, f"ablation_report_{tag}.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'mode':<14}{'MSE':>10}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
    for m in MODES:
        d = metrics[m]
        print(f"{m:<14}{d['mse']:>10.5f}{d['psnr']:>9.3f}{d['ssim']:>9.4f}{d['lpips']:>9.4f}")
    print("-" * 60)
    if "fid" in metrics["full"]:
        print(f"FULL rFID = {metrics['full']['fid']:.3f}   KID = {metrics['full']['kid']:.4f}")
    print(f"continuous branch adds: +{contrib['psnr']:.3f} PSNR, "
          f"+{contrib['ssim']:.4f} SSIM, {contrib['lpips']:+.4f} LPIPS")
    print("=" * 60)
    print(f"[saved] {rp}\n[done]")


if __name__ == "__main__":
    main()
