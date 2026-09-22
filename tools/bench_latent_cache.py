"""Throughput benchmark for the one-off latent-cache encode pass at ImageNet scale.

Runs the real frozen AE encode path (tools/latent_ae.py) over real HDF5 ImageNet batches,
so the measured rate includes JPEG decode + resize in the dataloader -- which is the part
that actually bottlenecks this pass, and which depends on how many CPUs the Slurm job asked
for (`--cpus-per-task`; the default of 1 is far too few).

Reports images/sec and the extrapolated wall-clock to encode the full train split, with and
without flip augmentation (flip doubles the number of encoder passes, not the image loads).

Usage:
    python tools/bench_latent_cache.py --ae-checkpoint checkpoints/dualvae/VQ_0.1/... --num-workers 8
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.latent_ae import load_frozen_ae
from data.datasets import HDF5ImageDataset, build_image_transform

N_TRAIN = 1_281_167


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-checkpoint", default="checkpoints/dualvae/VQ_0.1/dualvae_20260805-134849_09061b")
    p.add_argument("--dataset-path", default="/data/mgonzalez/datasets_HVAE/imagenet_full_256.h5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--flip", action="store_true", help="also encode the mirrored image")
    args = p.parse_args()

    device = "cuda"
    print(f"CPUs visible: {os.cpu_count()}  |  num_workers={args.num_workers}")
    ae = load_frozen_ae(args.ae_checkpoint, device=device)
    print(f"AE kind={ae.kind}  latent_shape={ae.latent_shape(256)}")

    ds = HDF5ImageDataset(args.dataset_path, split="train",
                          transform=build_image_transform("imagenet", 256), labeled=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    it = iter(loader)
    for _ in range(3):                                   # warmup: spin up workers, cudnn autotune
        b = next(it)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ae.encode(b["image"].to(device), sample=False)
    torch.cuda.synchronize()

    n = 0
    t0 = time.time()
    for _ in range(args.batches):
        b = next(it)
        imgs = b["image"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ae.encode(imgs, sample=False)
            if args.flip:
                ae.encode(torch.flip(imgs, dims=[3]), sample=False)
        n += imgs.size(0)
    torch.cuda.synchronize()
    dt = time.time() - t0

    ips = n / dt
    hours = N_TRAIN / ips / 3600
    print(f"\n{n} images in {dt:.1f}s  ->  {ips:.0f} img/s"
          f"   (flip={'on' if args.flip else 'off'})")
    print(f"full train split ({N_TRAIN:,} imgs): {hours:.2f} h per autoencoder")


if __name__ == "__main__":
    main()
