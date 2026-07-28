#!/usr/bin/env python
"""
DINO-style latent-feature visualization for a trained DUALVAE.

For a handful of validation images this renders, per branch, the spatial latent feature
map as an RGB image -- exactly the trick DINO/DINOv2 papers use to *show what features
look like on the image*: run PCA over the per-location feature vectors, map the top-3
principal components to R/G/B, and upsample back to image size. Locations with similar
features get similar colors, so you can see which parts of the image each branch encodes.

We fit ONE PCA per branch over all the shown images' locations, so colors are consistent
across the row (same feature -> same color). The first PC is also shown on its own as a
heatmap (DINO's usual "foreground" component).

Branches visualized (each an (C, H/8, W/8) feature map):
    - z_vq        : discrete VQ code map
    - z_cont      : continuous branch
    - z_sum       : z_vq + z_cont (the decoder's actual input, pre-attention)

Output: one PDF per run with a grid of
    original | reconstruction | z_vq RGB | z_cont RGB | z_sum RGB | z_sum PC1
for each sampled image.

Example
-------
    python tools/feature_pca_images.py \
        --checkpoint-dir checkpoints/dualvae/dualvae_20260724-133147_b03f09 \
        --num-images 5
"""
import os
import sys
import glob
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

from sklearn.decomposition import PCA

# Reuse the loaders from the analysis script (DUALVAE-only, final_epoch.pt preferred).
from tools.dualvae_latent_analysis import (
    load_config, find_weights, build_dualvae, build_transform, VALID_EXTS,
)

BRANCHES = ["z_vq", "z_cont", "z_sum"]
BRANCH_TITLES = {"z_vq": "VQ branch", "z_cont": "Continuous branch",
                 "z_sum": "Summed (z_vq + z_cont)"}


def sample_image_paths(images_dir, num_images, seed):
    files = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir)
                   if f.lower().endswith(VALID_EXTS))
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}.")
    rng = random.Random(seed)
    return rng.sample(files, min(num_images, len(files)))


def denormalize(img, dataset_name):
    """Undo the eval normalization to get a [0,1] image for display."""
    if str(dataset_name).lower() == "imagenette":
        return (img * 0.5 + 0.5).clamp(0, 1)
    if str(dataset_name).lower() == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        return (img.cpu() * std + mean).clamp(0, 1)
    return img.clamp(0, 1)


@torch.no_grad()
def encode_maps(model, paths, transform, device, seed):
    """Returns per-image dict with the input, reconstruction, and each branch's
    (C, H, W) feature map (numpy)."""
    torch.manual_seed(seed)
    model.eval()
    ds = model.downsample_factor
    C = model.latent_channels
    out = []
    for p in paths:
        img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        b, _, hh, ww = img.shape

        z_e = model.encoder(img)
        z_e_vq = model.bottle_neck_VQ(z_e)
        z_vq, _, _, _, _ = model.vq_layer(z_e_vq)
        if getattr(model, "wavelet_detail", False):
            _, hf = model.dwt(img)
            z_e_vanilla = model.wavelet_meanvar(model.detail_encoder(hf))
        elif model.residual_continuous:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
        else:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e)
        noise = torch.randn((b, C, hh // ds, ww // ds), device=device)
        z_cont, _, _ = model.forward_vanilla_z(z_e_vanilla, noise)
        z_sum = z_vq + z_cont

        # Reconstruction through the real forward (full model, both branches).
        recon = model(img, ablation_mode=-1)[0]

        out.append({
            "path": p,
            "input": img[0].cpu(),
            "recon": recon[0].cpu(),
            "z_vq": z_vq[0].cpu().numpy(),
            "z_cont": z_cont[0].cpu().numpy(),
            "z_sum": z_sum[0].cpu().numpy(),
        })
    return out


def fit_branch_pca(records, branch, seed):
    """Fit PCA(3) over the pooled per-location features of the given branch across all
    shown images, so the RGB coloring is consistent image-to-image."""
    pooled = []
    for r in records:
        C = r[branch].shape[0]
        pooled.append(r[branch].reshape(C, -1).T)   # (H*W, C)
    pooled = np.concatenate(pooled, 0)
    pca = PCA(n_components=3, random_state=seed).fit(pooled)
    # Global 1st/99th percentiles for stable normalization across images.
    proj = pca.transform(pooled)
    lo = np.percentile(proj, 1, axis=0)
    hi = np.percentile(proj, 99, axis=0)
    return pca, lo, hi


def feature_rgb(feat, pca, lo, hi, out_hw):
    """(C,H,W) feature map -> (out_h, out_w, 3) RGB via the branch PCA, upsampled."""
    C, H, W = feat.shape
    proj = pca.transform(feat.reshape(C, -1).T)          # (H*W, 3)
    proj = (proj - lo) / (hi - lo + 1e-8)
    proj = np.clip(proj, 0, 1).reshape(H, W, 3)
    t = torch.from_numpy(proj).permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(t, size=out_hw, mode="nearest")[0].permute(1, 2, 0).numpy()
    return up


def feature_pc1(feat, pca, lo, hi, out_hw):
    """First principal component as a single-channel heatmap (DINO 'foreground')."""
    C, H, W = feat.shape
    p1 = pca.transform(feat.reshape(C, -1).T)[:, 0]
    p1 = (p1 - lo[0]) / (hi[0] - lo[0] + 1e-8)
    p1 = np.clip(p1, 0, 1).reshape(H, W)
    t = torch.from_numpy(p1)[None, None].float()
    up = F.interpolate(t, size=out_hw, mode="nearest")[0, 0].numpy()
    return up


def make_figure(records, pcas, dataset_name, out_path, tag):
    n = len(records)
    cols = ["input", "recon", "z_vq", "z_cont", "z_sum", "z_sum_pc1"]
    col_titles = ["original", "reconstruction", "VQ features",
                  "continuous features", "summed features", "summed PC1"]
    fig, axes = plt.subplots(n, len(cols), figsize=(len(cols) * 2.3, n * 2.35))
    if n == 1:
        axes = axes[None, :]
    out_hw = tuple(records[0]["input"].shape[1:])

    for i, r in enumerate(records):
        inp = denormalize(r["input"], dataset_name).permute(1, 2, 0).numpy()
        rec = denormalize(r["recon"], dataset_name).permute(1, 2, 0).numpy()
        panels = [inp, rec]
        for b in BRANCHES:
            pca, lo, hi = pcas[b]
            panels.append(feature_rgb(r[b], pca, lo, hi, out_hw))
        pca, lo, hi = pcas["z_sum"]
        pc1 = feature_pc1(r["z_sum"], pca, lo, hi, out_hw)

        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel)
            axes[i, j].axis("off")
        axes[i, len(panels)].imshow(pc1, cmap="viridis")
        axes[i, len(panels)].axis("off")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[0, j].set_title(t, fontsize=10, fontweight="bold")

    fig.suptitle(f"DUALVAE latent features projected onto the image (PCA→RGB)  |  {tag}",
                 fontsize=11, y=1.005)
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
    p.add_argument("--num-images", type=int, default=5)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device: {device}")

    cfg, cfg_path = load_config(args.checkpoint_dir)
    weights = find_weights(args.checkpoint_dir)
    print(f"[info] model=dualvae  weights={weights}")
    model = build_dualvae(cfg, device)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    images_dir = os.path.join(args.dataset_dir, args.split)
    paths = sample_image_paths(images_dir, args.num_images, args.seed)
    print(f"[info] {len(paths)} images sampled from {images_dir}")

    transform = build_transform(cfg.get("resize_img", 256),
                                cfg.get("dataset_name", "imagenette"))
    records = encode_maps(model, paths, transform, device, args.seed)
    pcas = {b: fit_branch_pca(records, b, args.seed) for b in BRANCHES}

    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    make_figure(records, pcas, cfg.get("dataset_name", "imagenette"),
                os.path.join(out_dir, f"feature_pca_images_{tag}.pdf"), tag)
    print("[done]")


if __name__ == "__main__":
    main()
