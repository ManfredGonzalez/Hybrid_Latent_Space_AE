"""Throughput / max-batch-size benchmark for the latent flow trainer at ImageNet scale.

Replays the EXACT training step from experiments/train_latent_flow.py::train_step --
flow_matching_loss under bf16 autocast, backward, grad clip, Adam step, EMA update -- on
synthetic latents of the real shape, so the numbers transfer directly to a real run. The
UNet is built with num_classes=1000 (ImageNet-1k), not imagenette's 10, because the class
embedding table and the CFG NULL row scale with it.

Reports, per batch size: peak GPU memory and steps/sec, then extrapolates the wall-clock
cost of one ImageNet epoch (1,281,167 latents).

Usage:
    python tools/bench_flow_train.py --batch-sizes 64 128 256 384 512
"""
import argparse
import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_unet import LatentFlowUNet
from losses.flow_matching import flow_matching_loss

N_TRAIN = 1_281_167          # ImageNet-1k train split


def build(args, device):
    model = LatentFlowUNet(
        in_channels=args.latent_channels,
        latent_size=args.latent_size,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        num_classes=args.num_classes,
        dropout=args.dropout,
        num_heads=args.num_heads,
    ).to(device)
    return model


def bench_one(args, bs, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build(args, device)
    ema = copy.deepcopy(model).eval() if args.ema else None      # ModelEMA holds a full copy
    for p in (ema.parameters() if ema else []):
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    z = torch.randn(bs, args.latent_channels, args.latent_size, args.latent_size, device=device)
    y = torch.randint(0, args.num_classes, (bs,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = flow_matching_loss(model, z, y, t_sampling="logit_normal",
                                         cfg_dropout_prob=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:                                       # EMA update costs a full pass
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.9999).add_(pm.detach(), alpha=1e-4)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30
    sps = args.iters / dt
    params = sum(p.numel() for p in model.parameters())

    del model, ema, opt, z, y
    torch.cuda.empty_cache()
    return sps, peak, params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 384, 512, 768])
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--latent-size", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--num-res-blocks", type=int, default=2)
    p.add_argument("--attention-resolutions", type=int, nargs="+", default=[16, 8])
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {name}  ({total_mem:.1f} GiB)")
    print(f"UNet: base={args.base_channels} mults={args.channel_mults} "
          f"res_blocks={args.num_res_blocks} attn={args.attention_resolutions} "
          f"classes={args.num_classes}\n")

    print(f"{'batch':>7}{'steps/s':>10}{'lat/s':>10}{'peak GiB':>11}"
          f"{'s/epoch':>11}{'h/epoch':>9}")
    print("-" * 58)

    for bs in args.batch_sizes:
        try:
            sps, peak, params = bench_one(args, bs, device)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>7}{'OOM':>10}")
            torch.cuda.empty_cache()
            continue
        lat_s = sps * bs
        epoch_s = N_TRAIN / lat_s
        print(f"{bs:>7}{sps:>10.2f}{lat_s:>10.0f}{peak:>11.2f}"
              f"{epoch_s:>11.0f}{epoch_s/3600:>9.2f}")

    print(f"\nparams: {params/1e6:.1f}M")


if __name__ == "__main__":
    main()
