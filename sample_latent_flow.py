"""Generate class-conditional images from a trained latent flow model, and score them.

The training checkpoints are self-contained (weights + EMA + latent statistics + the
autoencoder they were trained against), so a run is reproduced from its checkpoint path
alone. Use this for the final DUALVAE-vs-VAE comparison at a fixed sampling budget, and for
the guidance sweep -- FID depends strongly on the guidance scale, so comparing two models at
a single arbitrary scale can easily rank them backwards.

Examples
--------
# grid of 8 samples per class, EMA weights, CFG 2.0
python sample_latent_flow.py --checkpoint ./checkpoints/flow_dualvae/<run>/best.pt \
    --n_per_class 8 --guidance_scale 2.0 --out ./samples/flow_dualvae

# generative FID/KID against the validation images
python sample_latent_flow.py --checkpoint ./checkpoints/flow_dualvae/<run>/best.pt \
    --fid --n_samples 5000

# same noise in both latent spaces, for a side-by-side figure
python sample_latent_flow.py --checkpoint <dualvae_run>/best.pt --noise_seed 7 --out A
python sample_latent_flow.py --checkpoint <vae_run>/best.pt     --noise_seed 7 --out B
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.datasets import get_labeled_datasets
from losses.flow_matching import sample_flow
from models.flow_unet import LatentFlowUNet
from tools.latent_ae import LatentStats, load_frozen_ae
from tools.normalization import denormalize
from tools.utils import create_directory, select_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Sample from a trained latent flow model.")
    p.add_argument("--checkpoint", required=True, type=str, help="Flow checkpoint (best.pt / final_epoch.pt).")
    p.add_argument("--ae_checkpoint", default=None, type=str,
                   help="Override the autoencoder recorded in the checkpoint (e.g. if paths moved).")
    p.add_argument("--out", default="./samples/latent_flow", type=str)
    p.add_argument("--n_per_class", default=8, type=int, help="Samples per class for the grid.")
    p.add_argument("--classes", default=None, type=str,
                   help="Comma-separated class indices to generate (default: all, subject to "
                        "--max_grid_classes).")
    p.add_argument("--max_grid_classes", default=16, type=int,
                   help="When --classes is not given and the model has more classes than this, "
                        "the grid uses a fixed seeded subset instead of every class. 0 = no cap "
                        "(at ImageNet's 1000 classes that is 1000*n_per_class solver runs).")
    p.add_argument("--guidance_scale", default=None, type=float, help="CFG weight (default: the training config's).")
    p.add_argument("--steps", default=None, type=int, help="ODE steps (default: the training config's).")
    p.add_argument("--solver", default=None, choices=["euler", "heun"])
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--noise_seed", default=1234, type=int,
                   help="Seed for the initial noise -- same seed + same class ids = the same "
                        "trajectory start in every model, which is what makes side-by-side "
                        "figures comparable.")
    p.add_argument("--raw_weights", action="store_true", help="Sample from the raw weights instead of the EMA.")
    p.add_argument("--save_individual", action="store_true", help="Also write one PNG per sample.")
    p.add_argument("--fid", action="store_true", help="Compute generative FID/KID against the val images.")
    p.add_argument("--n_samples", default=1000, type=int, help="Generated samples used for FID.")
    p.add_argument("--n_real", default=0, type=int, help="Cap on real val images for FID (0 = all).")
    p.add_argument("--device", default="cuda", type=str)
    return p.parse_args()


def load_flow_checkpoint(path, device, ae_checkpoint_override=None):
    # Our own checkpoint: carries a config dict alongside the tensors, so weights_only=False.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ckpt["args"])
    c, h, w = ckpt["latent_shape"]

    model = LatentFlowUNet(
        in_channels=c,
        latent_size=h,
        base_channels=getattr(cfg, "unet_base_channels", 128),
        channel_mults=tuple(getattr(cfg, "unet_channel_mults", (1, 2, 2, 2))),
        num_res_blocks=getattr(cfg, "unet_num_res_blocks", 2),
        attention_resolutions=tuple(getattr(cfg, "unet_attention_resolutions", (16, 8))),
        num_classes=ckpt["num_classes"],
        dropout=0.0,
        num_heads=getattr(cfg, "unet_num_heads", 4),
    ).to(device)

    ae_ckpt = ae_checkpoint_override or ckpt["ae_checkpoint"]
    ae = load_frozen_ae(ae_ckpt, device, getattr(cfg, "ae_config", None))
    stats = LatentStats.from_state_dict(ckpt["latent_stats"]).to(device)
    return model, ae, stats, cfg, ckpt


@torch.no_grad()
def generate(model, ae, stats, cfg, args, labels, device, seed):
    c, h, w = ae.latent_shape(cfg.resize_img)
    steps = args.steps if args.steps is not None else getattr(cfg, "sample_steps", 50)
    scale = args.guidance_scale if args.guidance_scale is not None else getattr(cfg, "guidance_scale", 2.0)
    solver = args.solver or getattr(cfg, "sample_solver", "heun")

    images = []
    for i in range(0, labels.shape[0], args.batch_size):
        y = labels[i:i + args.batch_size].to(device)
        gen = torch.Generator(device=device).manual_seed(seed + i)
        z = sample_flow(model, (y.shape[0], c, h, w), y, num_steps=steps, guidance_scale=scale,
                        device=device, solver=solver, generator=gen)
        x = ae.decode(stats.denormalize(z).float())
        images.append(denormalize(x.float(), cfg.dataset_name, device).clamp(0, 1).cpu())
    return torch.cat(images), {"steps": steps, "guidance_scale": scale, "solver": solver}


def main():
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.noise_seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.out)

    model, ae, stats, cfg, ckpt = load_flow_checkpoint(args.checkpoint, device, args.ae_checkpoint)
    weights = ckpt["model"] if (args.raw_weights or ckpt.get("ema") is None) else ckpt["ema"]
    model.load_state_dict(weights)
    model.eval()
    num_classes = ckpt["num_classes"]
    class_names = ckpt.get("class_names") or [str(i) for i in range(num_classes)]
    print(f"[flow] epoch {ckpt['epoch']} | {'raw' if args.raw_weights else 'EMA'} weights | "
          f"{num_classes} classes | AE: {ckpt['ae_checkpoint']}")

    if args.classes:
        class_ids = [int(c) for c in args.classes.split(",")]
    elif num_classes > args.max_grid_classes > 0:
        # "All classes" is fine at imagenette's 10 and absurd at ImageNet's 1000: it would be
        # 1000 x n_per_class images, each a full solver run, in a single unopenably large PNG.
        # Fall back to a FIXED seeded subset (same classes every invocation, so two checkpoints
        # stay comparable) and say so, rather than silently starting hours of sampling.
        g = torch.Generator().manual_seed(args.noise_seed or 1234)
        class_ids = sorted(torch.randperm(num_classes, generator=g)[:args.max_grid_classes].tolist())
        print(f"[grid] {num_classes} classes > --max_grid_classes {args.max_grid_classes}; "
              f"showing a fixed subset of {len(class_ids)}. Pass --classes to choose explicitly, "
              f"or --max_grid_classes 0 to force all {num_classes}.")
    else:
        class_ids = list(range(num_classes))

    # ---- grid ----
    labels = torch.tensor(class_ids, dtype=torch.long).repeat_interleave(args.n_per_class)
    images, sample_cfg = generate(model, ae, stats, cfg, args, labels, device, args.noise_seed)
    grid_path = os.path.join(args.out, "samples_grid.png")
    vutils.save_image(vutils.make_grid(images, nrow=args.n_per_class, normalize=False), grid_path)
    print(f"[out] {grid_path}  (rows = {[class_names[i] for i in class_ids]})")

    if args.save_individual:
        for i, (img, y) in enumerate(zip(images, labels)):
            d = os.path.join(args.out, "images", class_names[int(y)].replace(" ", "_"))
            create_directory(d)
            vutils.save_image(img, os.path.join(d, f"{i:05d}.png"))
        print(f"[out] {len(images)} individual PNGs under {os.path.join(args.out, 'images')}")

    report = {"checkpoint": args.checkpoint, "ae_checkpoint": ckpt["ae_checkpoint"],
              "epoch": ckpt["epoch"], "ema": not args.raw_weights, **sample_cfg}

    # ---- generative FID / KID ----
    if args.fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        fid = FrechetInceptionDistance(normalize=True).to(device)
        kid = KernelInceptionDistance(subset_size=getattr(cfg, "kid_subset_size", 100),
                                      normalize=True).to(device)
        _, valset = get_labeled_datasets(cfg.dataset_name, path=cfg.dataset_path,
                                         resize_img=cfg.resize_img, seed=cfg.seed)
        loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        max_real = args.n_real or float("inf")
        n_real = 0
        for batch in loader:
            imgs = denormalize(batch["image"].float().to(device), cfg.dataset_name, device).clamp(0, 1)
            fid.update(imgs, real=True)
            kid.update(imgs, real=True)
            n_real += imgs.shape[0]
            if n_real >= max_real:
                break

        fake_labels = torch.arange(num_classes).repeat(int(np.ceil(args.n_samples / num_classes)))[:args.n_samples]
        for i in range(0, args.n_samples, args.batch_size):
            chunk = fake_labels[i:i + args.batch_size]
            imgs, _ = generate(model, ae, stats, cfg, args, chunk, device, args.noise_seed + 10_000 + i)
            fid.update(imgs.to(device), real=False)
            kid.update(imgs.to(device), real=False)

        kid_mean, kid_std = kid.compute()
        report.update({"gen_fid": fid.compute().item(), "gen_kid_mean": kid_mean.item(),
                       "gen_kid_std": kid_std.item(), "n_fake": args.n_samples, "n_real": n_real})
        print(f"[FID] {report['gen_fid']:.3f} | KID {report['gen_kid_mean']:.5f} "
              f"({args.n_samples} fake vs {n_real} real, cfg={sample_cfg})")

    with open(os.path.join(args.out, "sample_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[out] {os.path.join(args.out, 'sample_report.json')}")


if __name__ == "__main__":
    main()
