"""Measures the latent energy budget that explains each checkpoint's optimal mu-multiplier.

For each model, over a sample of ImageNet val, reports:
  E|mu|^2        energy the mean branch carries
  E[sigma^2]     energy the noise branch carries   -> predicts whether b matters at all
  noise_frac     E[sigma^2] / (E|mu|^2 + E[sigma^2])
  offset_ratio   ||E[mu]|| / sqrt(E|mu|^2)  -- how much of mu is a SYSTEMATIC (per-channel)
                 offset vs. zero-mean dither. An EMA codebook's residual is zero-mean by
                 construction (see DUALVAE._component_prior); a fixed FSQ grid's is not,
                 because its codes are grid corners rather than cluster centroids.
  delta_share    E|Delta| / E|z_vq + Delta|  -- how much of the decoder's input the
                 continuous branch supplies at all.

Usage:
    python tools/latent_budget_diag.py --dataset-path /data/.../imagenet_full_256.h5 --limit 512
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mu_sigma_sweep import (load_config, find_weights, build_model,
                                  dualvae_branch_terms, vae_branch_terms)
from data.datasets import HDF5ImageDataset, build_image_transform

CKPTS = [
    ("DualVAE — VQ 0.1",  "checkpoints/dualvae/VQ_0.1/dualvae_20260805-134849_09061b"),
    ("DualVAE — FSQ 0.1", "checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305"),
    ("DualVAE — FSQ-E",   "checkpoints/dualvae/fsq_E/dualvae_20260813-204926_f85dfe"),
    ("Plain VAE",         "checkpoints/vae/vae_20260810-165000_c9374d"),
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default="/data/mgonzalez/datasets_HVAE/imagenet_full_256.h5")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    transform = build_image_transform("imagenet", 256)
    ds = HDF5ImageDataset(args.dataset_path, split="val", transform=transform, labeled=False)
    ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"{'model':<20}{'E|mu|^2':>10}{'E[sig^2]':>10}{'noise_frac':>12}"
          f"{'offset_ratio':>14}{'delta_share':>13}")
    print("-" * 79)

    for name, ckpt in CKPTS:
        cfg, _ = load_config(ckpt)
        model, model_name = build_model(cfg, args.device)
        sd = torch.load(find_weights(ckpt), map_location=args.device)
        model.load_state_dict(sd if "encoder.0.weight" in sd else sd.get("model_state_dict", sd))
        model.eval()
        is_dual = model_name == "dualvae"

        mu_sq = sig_sq = 0.0
        mu_sum = None           # running sum of mu, for the systematic-offset test
        d_norm = z_norm = 0.0
        n = 0

        for batch in loader:
            imgs = batch["image"].to(args.device)
            if is_dual:
                z_vq, mean, stdev = dualvae_branch_terms(model, imgs)
            else:
                mean, stdev = vae_branch_terms(model, imgs)
                z_vq = torch.zeros_like(mean)

            b = imgs.size(0)
            mu_sq += (mean ** 2).mean().item() * b
            sig_sq += (stdev ** 2).mean().item() * b
            # per-channel mean of mu, accumulated over batch + spatial dims
            s = mean.mean(dim=(0, 2, 3)) * b
            mu_sum = s if mu_sum is None else mu_sum + s

            delta = mean + stdev * torch.randn_like(mean)
            d_norm += delta.norm(dim=1).mean().item() * b
            z_norm += (z_vq + delta).norm(dim=1).mean().item() * b
            n += b

        mu_sq /= n; sig_sq /= n; d_norm /= n; z_norm /= n
        mu_bar = (mu_sum / n)                       # (C,) per-channel systematic offset
        offset_ratio = (mu_bar.norm() / (mu_sq * mu_bar.numel()) ** 0.5).item()
        noise_frac = sig_sq / (mu_sq + sig_sq)

        print(f"{name:<20}{mu_sq:>10.4f}{sig_sq:>10.5f}{noise_frac:>12.4f}"
              f"{offset_ratio:>14.4f}{d_norm / z_norm:>13.4f}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
