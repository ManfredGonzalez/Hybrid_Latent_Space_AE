#!/usr/bin/env python
"""
Disentangling what the VQ codes actually contribute to a DUALVAE reconstruction.

The plain branch ablation (zeroing a branch) is confounded: the decoder was trained on the
SUM z = z_vq + z_cont, so zeroing z_vq both removes information AND pushes the decoder input
off the manifold it was trained on. This script runs four experiments to separate those.

EXPERIMENT 1 -- Replace-with-mean (decompose the code contribution).
    Compare, keeping the continuous branch fixed:
        full        : z_vq = real codes
        code_mean   : z_vq = dataset-mean code vector at EVERY location (on-manifold DC,
                      but no per-image code information)
        code_zero   : z_vq = 0                       (= vanilla_only)
    Then:
        location-specific code INFO   = quality(full) - quality(code_mean)
        DC / off-manifold effect      = quality(code_mean) - quality(code_zero)
    This splits the full-minus-vanilla_only gap into "real info" vs "artifact".

EXPERIMENT 2 -- Random-code swap.
        code_random : z_vq = real code vectors randomly permuted across locations
                      (identical code distribution/magnitude, wrong image).
    quality(full) - quality(code_random) large  => codes carry real per-image info;
    small => the codes act as a generic magnitude-matched backbone.

EXPERIMENT 3 -- Frequency decomposition.
    Radially-averaged power spectra of the pixel-space contributions
        Delta_codes = recon(full) - recon(code_mean)   (what per-image codes add)
        Delta_cont  = recon(full) - recon(cont_zero)    (what the continuous branch adds)
    Tests the "codes = coarse/low-frequency, continuous = fine/high-frequency" hypothesis.

EXPERIMENT 4 -- Standalone-VAE baseline.
    A matched continuous-only VAE (trained from scratch, --vae-dir) reconstructs the same
    images. This is the correct baseline for the claim "the codes improve a VAE" -- the
    dual model's own vanilla_only is NOT, because it offloaded coarse structure to the codes
    during training.

Outputs: a metrics bar chart, a frequency-spectrum plot, a qualitative grid, and a JSON
report -- all written to the dualvae checkpoint dir.

Example
-------
    python tools/branch_contribution.py \
        --dualvae-dir checkpoints/dualvae/dualvae_20260724-121508_392b5b \
        --vae-dir "checkpoints/vae/VAE_betaKL@0.001@Downsample_8" \
        --num-images 300
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips as lpips_lib

from models.vae import VAE
from tools.dualvae_latent_analysis import (
    load_config, find_weights, build_dualvae, build_transform, VALID_EXTS,
)

# Interventions on z_vq (continuous branch kept), plus the two single-branch references.
MODES = ["full", "code_mean", "code_random", "code_zero", "cont_zero"]
MODE_LABEL = {
    "full": "full (both branches)",
    "code_mean": "code=mean (no per-image code)",
    "code_random": "code=random (shuffled)",
    "code_zero": "code=0 (vanilla_only)",
    "cont_zero": "cont=0 (vq_only)",
}
MODE_COLORS = {"full": "#3b7dd8", "code_mean": "#7b6cd8", "code_random": "#d88a3b",
               "code_zero": "#5aa469", "cont_zero": "#c0567a", "vae_baseline": "#888888"}


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
def _branches(model, imgs, device):
    """Return z_vq, z_cont for a batch (continuous branch sampled with a fixed seed)."""
    ds = model.downsample_factor
    C = model.latent_channels
    b, _, hh, ww = imgs.shape
    z_e = model.encoder(imgs)
    z_e_vq = model.bottle_neck_VQ(z_e)
    z_vq, _, _, _, _ = model.vq_layer(z_e_vq)
    if getattr(model, "wavelet_detail", False):
        _, hf = model.dwt(imgs)
        z_e_vanilla = model.wavelet_meanvar(model.detail_encoder(hf))
    elif model.residual_continuous:
        z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
    else:
        z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e)
    torch.manual_seed(1234)
    noise = torch.randn((b, C, hh // ds, ww // ds), device=device)
    z_cont, _, _ = model.forward_vanilla_z(z_e_vanilla, noise)
    return z_vq, z_cont


@torch.no_grad()
def _decode(model, z_vq_use, z_cont_use):
    z = model.attention(z_vq_use + z_cont_use)
    return model.decoder(z)


@torch.no_grad()
def compute_mean_code(model, paths, transform, device, batch_size):
    """Dataset-mean z_vq vector (C,), averaged over all locations/images."""
    tot = None
    n = 0
    for start in range(0, len(paths), batch_size):
        imgs = torch.stack([transform(Image.open(p).convert("RGB"))
                            for p in paths[start:start + batch_size]]).to(device)
        z_vq, _ = _branches(model, imgs, device)
        s = z_vq.permute(0, 2, 3, 1).reshape(-1, z_vq.shape[1]).sum(0)
        tot = s if tot is None else tot + s
        n += z_vq.shape[0] * z_vq.shape[2] * z_vq.shape[3]
    return (tot / n)  # (C,)


@torch.no_grad()
def intervene_zvq(z_vq, mode, mean_code, seed):
    if mode == "full":
        return z_vq
    if mode == "code_zero" or mode == "cont_zero":
        return z_vq * 0 if mode == "code_zero" else z_vq
    if mode == "code_mean":
        return mean_code.view(1, -1, 1, 1).expand_as(z_vq).clone()
    if mode == "code_random":
        b, c, h, w = z_vq.shape
        flat = z_vq.permute(0, 2, 3, 1).reshape(-1, c)
        g = torch.Generator(device=flat.device).manual_seed(seed)
        perm = torch.randperm(flat.shape[0], generator=g, device=flat.device)
        return flat[perm].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise ValueError(mode)


@torch.no_grad()
def reconstruct(model, imgs, mode, mean_code, device, seed):
    z_vq, z_cont = _branches(model, imgs, device)
    z_vq_use = intervene_zvq(z_vq, mode, mean_code, seed)
    z_cont_use = z_cont * 0 if mode == "cont_zero" else z_cont
    return _decode(model, z_vq_use, z_cont_use)


class Metrics:
    def __init__(self, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    @torch.no_grad()
    def add(self, acc, fake, real):
        bs = real.size(0)
        acc["mse"] += F.mse_loss(fake, real, reduction="mean").item() * bs
        acc["psnr"] += self.psnr(fake, real).item() * bs
        acc["ssim"] += self.ssim(fake, real).item() * bs
        acc["lpips"] += self.lpips(fake * 2 - 1, real * 2 - 1).mean().item() * bs
        acc["n"] += bs


@torch.no_grad()
def radial_power_spectrum(diff_gray):
    """diff_gray: (B,H,W) real. Returns (freqs, mean radial power)."""
    B, H, W = diff_gray.shape
    f = torch.fft.fftshift(torch.fft.fft2(diff_gray), dim=(-2, -1))
    power = (f.abs() ** 2).mean(0).cpu().numpy()          # (H,W)
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    rmax = min(cy, cx)
    radial = np.array([power[r == i].mean() for i in range(rmax)])
    freqs = np.arange(rmax) / (2.0 * rmax)                 # cycles / pixel (0..0.5)
    return freqs, radial


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dualvae-dir", required=True)
    ap.add_argument("--vae-dir", default=None, help="Standalone VAE checkpoint dir (Exp 4).")
    ap.add_argument("--dataset-dir",
                    default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device: {device}")

    cfg, _ = load_config(args.dualvae_dir)
    w = find_weights(args.dualvae_dir)
    print(f"[info] dualvae weights: {w}")
    model = build_dualvae(cfg, device)
    sd = torch.load(w, map_location=device)
    model.load_state_dict(sd if "encoder.0.weight" in sd else sd["model_state_dict"])
    model.eval()
    dataset_name = cfg.get("dataset_name", "imagenette")
    transform = build_transform(cfg.get("resize_img", 256), dataset_name)

    images_dir = os.path.join(args.dataset_dir, args.split)
    paths = sample_image_paths(images_dir, args.num_images, args.seed)
    print(f"[info] scoring {len(paths)} images from {images_dir}")

    mean_code = compute_mean_code(model, paths, transform, device, args.batch_size)
    print(f"[info] ||mean code|| = {mean_code.norm().item():.3f}")

    M = Metrics(device)
    acc = {m: {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0} for m in MODES}
    # For the frequency experiment, accumulate the contribution difference images.
    spec = {"Delta_codes": [], "Delta_cont": []}

    vae = None
    if args.vae_dir:
        # Read the VAE's OWN config_used.yaml -- `cfg` here is the DUALVAE's, and its
        # downsample_factor/latent_channels only coincide with the baseline's in a
        # matched run. Falls back to `cfg` when the baseline dir has no config copy.
        try:
            vae_cfg, _ = load_config(args.vae_dir)
        except FileNotFoundError:
            vae_cfg = cfg
        vae = VAE(downsample_factor=vae_cfg.get("downsample_factor", 8),
                  latent_channels=vae_cfg.get("latent_channels", 4)).to(device)
        vw = find_weights(args.vae_dir)
        vsd = torch.load(vw, map_location=device)
        vae.load_state_dict(vsd if "encoder.0.weight" in vsd else vsd["model_state_dict"])
        vae.eval()
        acc["vae_baseline"] = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
        print(f"[info] VAE baseline weights: {vw}")

    for start in range(0, len(paths), args.batch_size):
      with torch.no_grad():
        imgs = torch.stack([transform(Image.open(p).convert("RGB"))
                            for p in paths[start:start + args.batch_size]]).to(device)
        real = denorm(imgs, dataset_name)
        recons = {}
        for m in MODES:
            fake = denorm(reconstruct(model, imgs, m, mean_code, device, args.seed), dataset_name)
            recons[m] = fake
            M.add(acc[m], fake, real)
        if vae is not None:
            torch.manual_seed(1234)
            vfake = denorm(vae(imgs)[0], dataset_name)
            M.add(acc["vae_baseline"], vfake, real)

        # Frequency contributions (grayscale differences).
        def gray(t):
            return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2])
        spec["Delta_codes"].append((gray(recons["full"]) - gray(recons["code_mean"])).cpu())
        spec["Delta_cont"].append((gray(recons["full"]) - gray(recons["cont_zero"])).cpu())

    metrics = {m: {k: acc[m][k] / acc[m]["n"] for k in ("mse", "psnr", "ssim", "lpips")}
               for m in acc}

    # ---- decomposition (Experiment 1 & 2) ----
    dec = {
        "location_specific_code_info": {
            k: metrics["full"][k] - metrics["code_mean"][k] for k in ("psnr", "ssim", "lpips")},
        "dc_offmanifold_effect": {
            k: metrics["code_mean"][k] - metrics["code_zero"][k] for k in ("psnr", "ssim", "lpips")},
        "random_code_degradation_vs_full": {
            k: metrics["full"][k] - metrics["code_random"][k] for k in ("psnr", "ssim", "lpips")},
    }

    out_dir = args.output_dir or args.dualvae_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.dualvae_dir))

    _plot_bars(metrics, out_dir, tag)
    fA, rA = radial_power_spectrum(torch.cat(spec["Delta_codes"], 0))
    fB, rB = radial_power_spectrum(torch.cat(spec["Delta_cont"], 0))
    _plot_spectrum(fA, rA, fB, rB, out_dir, tag)
    _qual_grid(model, paths[:5], transform, device, dataset_name, mean_code, args.seed,
               out_dir, tag, vae)

    report = {"dualvae_dir": args.dualvae_dir, "vae_dir": args.vae_dir,
              "n_images": len(paths), "mean_code_norm": float(mean_code.norm().item()),
              "metrics": metrics, "decomposition": dec}
    rp = os.path.join(out_dir, f"branch_contribution_{tag}.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    _print(metrics, dec, tag)
    print(f"[saved] {rp}\n[done]")


def _plot_bars(metrics, out_dir, tag):
    order = [m for m in MODES if m in metrics] + (["vae_baseline"] if "vae_baseline" in metrics else [])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, better in zip(axes, ["psnr", "ssim", "lpips"],
                               ["higher better", "higher better", "lower better"]):
        vals = [metrics[m][key] for m in order]
        ax.bar(range(len(order)), vals, color=[MODE_COLORS.get(m, "#555") for m in order],
               edgecolor="black", linewidth=0.4)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{key.upper()} ({better})", fontsize=11, fontweight="bold")
    fig.suptitle(f"Code-intervention reconstruction quality  |  {tag}", y=1.02, fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, f"branch_contribution_bars_{tag}.pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {p}")


def _plot_spectrum(fA, rA, fB, rB, out_dir, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(fA[1:], rA[1:], color="#c0567a", lw=2, label=r"$\Delta$codes (full $-$ code=mean)")
    ax.loglog(fB[1:], rB[1:], color="#3b7dd8", lw=2, label=r"$\Delta$cont (full $-$ cont=0)")
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("radial power")
    ax.set_title("Exp 3: which frequencies each branch contributes\n"
                 "left = coarse/low-freq, right = fine/high-freq",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(out_dir, f"branch_contribution_spectrum_{tag}.pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {p}")


@torch.no_grad()
def _qual_grid(model, paths, transform, device, dataset_name, mean_code, seed, out_dir, tag, vae):
    cols = ["original"] + MODES + (["vae_baseline"] if vae is not None else [])
    fig, axes = plt.subplots(len(paths), len(cols), figsize=(len(cols) * 2.0, len(paths) * 2.1))
    if len(paths) == 1:
        axes = axes[None, :]
    for i, p in enumerate(paths):
        imgs = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        real = denorm(imgs, dataset_name)[0].cpu().permute(1, 2, 0).numpy()
        panels = [real]
        for m in MODES:
            fake = denorm(reconstruct(model, imgs, m, mean_code, device, seed), dataset_name)
            panels.append(fake[0].cpu().permute(1, 2, 0).numpy())
        if vae is not None:
            torch.manual_seed(1234)
            panels.append(denorm(vae(imgs)[0], dataset_name)[0].cpu().permute(1, 2, 0).numpy())
        for j, panel in enumerate(panels):
            axes[i, j].imshow(np.clip(panel, 0, 1)); axes[i, j].axis("off")
            if i == 0:
                axes[0, j].set_title(cols[j], fontsize=8, fontweight="bold")
    fig.suptitle(f"Code-intervention reconstructions  |  {tag}", y=1.01, fontsize=10)
    fig.tight_layout()
    pth = os.path.join(out_dir, f"branch_contribution_grid_{tag}.pdf")
    fig.savefig(pth, format="pdf", bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {pth}")


def _print(metrics, dec, tag):
    print("\n" + "=" * 66)
    print(f"BRANCH CONTRIBUTION  |  {tag}")
    print(f"{'mode':<16}{'PSNR':>8}{'SSIM':>9}{'LPIPS':>9}")
    for m in [x for x in MODES if x in metrics] + (["vae_baseline"] if "vae_baseline" in metrics else []):
        d = metrics[m]
        print(f"{m:<16}{d['psnr']:>8.3f}{d['ssim']:>9.4f}{d['lpips']:>9.4f}")
    print("-" * 66)
    li, dc, rc = dec["location_specific_code_info"], dec["dc_offmanifold_effect"], dec["random_code_degradation_vs_full"]
    print("Decomposition of the full - vanilla_only gap:")
    print(f"  location-specific code INFO (full - code_mean): +{li['ssim']:.4f} SSIM / +{li['psnr']:.2f} PSNR")
    print(f"  DC / off-manifold effect   (code_mean - code_zero): +{dc['ssim']:.4f} SSIM / +{dc['psnr']:.2f} PSNR")
    print(f"  random-code degradation    (full - code_random): +{rc['ssim']:.4f} SSIM / +{rc['psnr']:.2f} PSNR")
    print("=" * 66)


if __name__ == "__main__":
    main()
