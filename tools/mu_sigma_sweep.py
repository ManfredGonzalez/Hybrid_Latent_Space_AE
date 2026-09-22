"""Inference-time sweep of z = a*mu + b*(sigma*eps) over the FULL ImageNet val split.

Motivation: DUALVAE.forward() combines z = z_vq + Delta, Delta = mu + sigma*eps (see
forward_vanilla_z in models/dual_vae.py). This script asks how rFID responds to damping
the continuous branch at inference time: z_vq + a*mu + b*sigma*eps, for every (a, b) in
an 8x8 grid of multipliers. The plain VAE baseline has no z_vq term, so its whole latent
is scaled instead: z = 0.18215 * (a*mu + b*sigma*eps) (0.18215 is VAE_Encoder's own
post-reparam scale, applied here so a=b=1.0 reproduces the checkpoint's normal forward
exactly).

Per-batch cost amortization: mu, sigma and z_vq do NOT depend on (a, b), so they are
computed ONCE per batch. A single eps ~ N(0, I) is also sampled ONCE per batch and reused
for every (a, b) cell in the grid, so the 64 rFID numbers differ only by the deterministic
scaling -- not by re-sampled reparameterization noise. Only the decoder forward + Inception
feature extraction is repeated per (a, b) cell (unavoidable: each produces a different
reconstruction). All 64 FrechetInceptionDistance accumulators share ONE InceptionV3
instance (`feature=` accepts a module), so the 64x cost is only the (small, constant)
accumulator buffers, not 64 copies of Inception.

Usage:
    python tools/mu_sigma_sweep.py --checkpoint-dir checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305 \
        --dataset-path /data/mgonzalez/datasets_HVAE/imagenet_full_256.h5 \
        --output-csv results/mu_sigma_sweep/fsq_0.1/rfid_grid.csv
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_vae import DUALVAE
from models.vae import VAE
from data.datasets import HDF5ImageDataset, build_image_transform

MULTIPLIERS = [1.000, 0.975, 0.950, 0.900, 0.850, 0.800, 0.750, 0.700]


def load_config(checkpoint_dir):
    """checkpoint_dir/config_used.yaml -> dict. Deliberately NOT imported from
    tools/dualvae_latent_analysis.py: that module pulls in matplotlib/sklearn/diptest
    purely for its plotting helpers, which this inference-only sweep doesn't need."""
    import glob
    import yaml

    candidates = glob.glob(os.path.join(checkpoint_dir, "config_used.yaml")) or \
        glob.glob(os.path.join(checkpoint_dir, "*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No config_used.yaml found in {checkpoint_dir}.")
    with open(candidates[0], "r") as f:
        return yaml.safe_load(f), candidates[0]


def find_weights(checkpoint_dir):
    """Prefer final_epoch.pt (the final-epoch weights, per LATENT_FLOW.md's
    resolve_ae_checkpoint convention) over best.pt, so every checkpoint in the sweep is
    evaluated at the SAME point (end of training), not whichever epoch happened to be
    best on train loss."""
    for name in ("final_epoch.pt", "final_epoch.pth", "best.pt", "best.pth"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Neither final_epoch.pt nor best.pt found in {checkpoint_dir}.")


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def build_model(cfg, device):
    name = str(cfg.get("model", "dualvae")).lower()
    if name == "vae":
        model = VAE(
            downsample_factor=cfg.get("downsample_factor", 8),
            latent_channels=cfg.get("latent_channels", 4),
        )
    elif name == "dualvae":
        model = DUALVAE(
            commitment_cost=cfg.get("commitment_cost", 0.25),
            latent_channels=cfg.get("latent_channels", 8),
            num_embeddings=cfg.get("num_embeddings", 256),
            downsample_factor=cfg.get("downsample_factor", 8),
            l2_normalize_codes=cfg.get("l2_normalize_codes", False),
            cont_dropout_p=cfg.get("cont_dropout_p", 0.0),
            use_ema_codebook=cfg.get("use_ema_codebook", False),
            ema_decay=cfg.get("ema_decay", 0.99),
            ema_eps=cfg.get("ema_eps", 1e-5),
            ema_dead_threshold=cfg.get("ema_dead_threshold", 1.0),
            rq_depth=cfg.get("rq_depth", 1),
            residual_continuous=cfg.get("residual_continuous", False),
            component_prior=cfg.get("component_prior", False),
            sigma2_floor=cfg.get("sigma2_floor", 1e-3),
            sigma2_ceil=cfg.get("sigma2_ceil", 10.0),
            wavelet_detail=cfg.get("wavelet_detail", False),
            wavelet_band_channels=cfg.get("wavelet_band_channels", None),
            hierarchical_semantic=cfg.get("hierarchical_semantic", False),
            coarse_factor=cfg.get("coarse_factor", 4),
            coarse_num_embeddings=cfg.get("coarse_num_embeddings", 64),
            quantizer=cfg.get("quantizer", "vq"),
            fsq_levels=cfg.get("fsq_levels", None),
        )
    else:
        raise ValueError(f"Unsupported model type {name!r} for this sweep.")
    return model.to(device), name


@torch.no_grad()
def dualvae_branch_terms(model, images):
    """Replicates DUALVAE.forward()'s encode path (ablation_mode=-1; eval mode makes
    cont_dropout a no-op) up to the point mu/sigma/z_vq are known, without sampling eps
    or decoding, so the caller can reuse it across an entire (a, b) grid."""
    if getattr(model, "hierarchical_semantic", False) or getattr(model, "wavelet_detail", False):
        raise NotImplementedError(
            "hierarchical_semantic / wavelet_detail models aren't wired into this sweep "
            "(none of the checkpoints this script was built for use them)."
        )
    z_e = model.encoder(images)
    z_e_vq = model.bottle_neck_VQ(z_e)
    z_vq, _, _, _, _ = model.vq_layer(z_e_vq)

    if model.residual_continuous:
        r = model.vq_layer.pre_quant(z_e_vq) - z_vq.detach()
        z_e_vanilla = model.vanilla_VAE_bottle_neck(r)
    else:
        z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e)

    mean, log_variance = torch.chunk(z_e_vanilla, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)
    stdev = log_variance.exp().sqrt()
    return z_vq, mean, stdev


@torch.no_grad()
def dualvae_decode(model, z_vq, mean, stdev, eps, a, b):
    delta = a * mean + b * stdev * eps
    z_combined = model.attention(z_vq + delta)
    return model.decoder(z_combined)


@torch.no_grad()
def vae_branch_terms(model, images):
    b, _, h, w = images.shape
    zeros_noise = torch.zeros(
        b, model.latent_channels, h // model.downsample_factor, w // model.downsample_factor,
        device=images.device,
    )
    _, mean, log_variance = model.encoder(images, zeros_noise)  # encoder clamps log_variance itself
    stdev = log_variance.exp().sqrt()
    return mean, stdev


@torch.no_grad()
def vae_decode(model, mean, stdev, eps, a, b):
    z = (a * mean + b * stdev * eps) * 0.18215  # match VAE_Encoder's own post-reparam scale
    return model.decoder(z)


def build_fid_grid(device):
    from torchmetrics.image.fid import FrechetInceptionDistance

    # normalize=False everywhere: this script converts [0,1] float images to uint8 itself
    # (see `to_uint8` / the main loop) and feeds that straight to every metric. Necessary
    # because passing `feature=<Module>` makes torchmetrics treat it as a fully custom
    # feature extractor (`used_custom_model=True`) and SKIP its own normalize->uint8
    # conversion (fid.py's update(): `(imgs*255).byte() if normalize and not
    # used_custom_model`) -- but the shared module is really the standard NoTrainInceptionV3,
    # which still requires uint8 input. Doing the conversion ourselves sidesteps that gap.
    grid = {}
    base = None
    for a in MULTIPLIERS:
        for b in MULTIPLIERS:
            if base is None:
                base = FrechetInceptionDistance(normalize=False).to(device)
                # The feature=<Module> constructor path also probes the module with a dummy
                # image to infer num_features, guessing a dtype (float vs uint8) from
                # `normalize` that doesn't match NoTrainInceptionV3's actual (always-uint8)
                # forward -- it crashes on the very sharing this loop exists to do. Setting
                # num_features up front makes later constructors skip that probe (torchmetrics
                # reads it straight off the module instead).
                base.inception.num_features = 2048
                grid[(a, b)] = base
            else:
                grid[(a, b)] = FrechetInceptionDistance(feature=base.inception, normalize=False).to(device)
    return grid


def to_uint8(img_01):
    return (img_01 * 255).clamp(0, 255).byte()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-path", required=True, help="Path to imagenet_full_256.h5")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Only use the first N images (smoke testing).")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="Run encoder/decoder under autocast(fp16).")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg, cfg_path = load_config(args.checkpoint_dir)
    model, model_name = build_model(cfg, device)
    weights_path = find_weights(args.checkpoint_dir)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd if "encoder.0.weight" in sd else sd.get("model_state_dict", sd))
    model.eval()
    is_dual = model_name == "dualvae"

    print(f"[cfg] {cfg_path}")
    print(f"[weights] {weights_path}")
    print(f"[model] {model_name}  quantizer={cfg.get('quantizer', 'vq') if is_dual else 'n/a'}  device={device}")

    transform = build_image_transform("imagenet", cfg.get("resize_img", 256))
    dataset = HDF5ImageDataset(args.dataset_path, split=args.split, transform=transform, labeled=False)
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[data] {args.dataset_path} split={args.split} n={len(dataset)} batch_size={args.batch_size}")

    fid_grid = build_fid_grid(device)
    print(f"[fid] {len(fid_grid)} accumulators, 1 shared InceptionV3")

    amp_dtype = torch.float16 if args.amp else None
    t0 = time.time()
    n_seen = 0
    for batch in tqdm(loader, desc="sweep", unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        real = to_uint8(denorm(images).float())

        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                             dtype=amp_dtype, enabled=args.amp):
            if is_dual:
                z_vq, mean, stdev = dualvae_branch_terms(model, images)
            else:
                mean, stdev = vae_branch_terms(model, images)
            eps = torch.randn_like(mean)

            for (a, b), fid in fid_grid.items():
                if is_dual:
                    recon = dualvae_decode(model, z_vq, mean, stdev, eps, a, b)
                else:
                    recon = vae_decode(model, mean, stdev, eps, a, b)
                fake = to_uint8(denorm(recon).float())
                fid.update(real, real=True)
                fid.update(fake, real=False)
        n_seen += images.size(0)

    elapsed = time.time() - t0
    print(f"[done] {n_seen} images in {elapsed/60:.1f} min ({elapsed/max(n_seen,1)*1000:.1f} ms/image)")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint_dir", "model", "quantizer", "a", "b", "rfid", "n_images"])
        for (a, b), fid in fid_grid.items():
            rfid = fid.compute().item()
            writer.writerow([args.checkpoint_dir, model_name,
                              cfg.get("quantizer", "vq") if is_dual else "n/a",
                              a, b, rfid, n_seen])
            print(f"  a={a:.3f} b={b:.3f}  rFID={rfid:.4f}")
    print(f"[csv] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
