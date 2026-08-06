"""Measure the largest per-GPU batch that fits, and the throughput at each size.

Runs the REAL training step -- same model, same LPIPS criterion, same GAN (generator term
active, which is the memory high-water mark), same EMA codebook -- on ONE GPU, by calling the
trainer's own train_one_epoch. Nothing here re-implements the step, so what it measures is
what the run will do.

    python tools/probe_batch_size.py --config configs/dualvae_imagenet_gan_C.yaml \
                                     --batch-sizes 16,24,32,40,48

Reads two numbers you need before launching:
  * peak VRAM per batch size  -> the largest size with headroom
  * img/s per GPU             -> multiply by 4 for the node, then size --time

OOM at a given size is caught and reported, not fatal: the sweep continues to the next size.
"""

import argparse
import sys
import time

import torch


def build_probe_loader(args, batch_size, n_batches):
    """A loader over a small slice of the REAL dataset -- same decode path and same tensor
    shapes as training, so throughput here is comparable to the real thing."""
    from torch.utils.data import DataLoader, Subset
    from data.datasets import get_benchmark_dataset
    from tools.utils import seed_worker

    trainset, _ = get_benchmark_dataset(args.dataset_name, path=args.dataset_path,
                                        resize_img=args.resize_img, seed=args.seed)
    # A couple of extra batches so the loader is never the thing that ends the loop.
    n_images = batch_size * (n_batches + 2)
    subset = Subset(trainset, list(range(min(n_images, len(trainset)))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=args.num_workers, worker_init_fn=seed_worker,
                      pin_memory=True, drop_last=True,
                      persistent_workers=args.num_workers > 0)


def probe_one(args, batch_size, n_batches):
    from experiments.train_dualvae import (initialize_model, build_recon_criterion,
                                           train_one_epoch)
    from losses.gan import build_gan

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    model = optimizer = gan = recon_criterion = loader = None
    try:
        model, optimizer = initialize_model(args)
        recon_criterion = build_recon_criterion(args)
        gan = build_gan(args, model, args.device)
        loader = build_probe_loader(args, batch_size, n_batches)

        # epoch = gan_start_epoch + ramp: the generator adversarial term is fully active, which
        # is the configuration that peaks memory. Probing at epoch 0 would under-report and hand
        # back a batch size that OOMs hours into the run, once the GAN switches on.
        epoch = int(getattr(args, 'gan_start_epoch', 0)) + int(getattr(args, 'gan_ramp_epochs', 0))

        # One warm-up call (2 batches) so cudnn.benchmark autotuning and the allocator's
        # first-touch growth land OUTSIDE the timed region.
        train_one_epoch(model, loader, optimizer, args.device, epoch, epoch + 1, args.kl_beta,
                        recon_criterion, use_amp=args.use_amp, gan=gan, limit_train_batches=2)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(args.device)

        t0 = time.time()
        train_one_epoch(model, loader, optimizer, args.device, epoch, epoch + 1, args.kl_beta,
                        recon_criterion, use_amp=args.use_amp, gan=gan, limit_train_batches=n_batches)
        torch.cuda.synchronize()
        dt = time.time() - t0

        peak = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
        return peak, batch_size * n_batches / dt
    finally:
        # Drop every reference HERE, inside the frame that owns them. Letting an OOM propagate
        # with these still bound would keep the model and its activations alive through the
        # exception's traceback, and the next (smaller) batch size would then OOM too --
        # turning one real limit into a cascade of fake ones.
        del model, optimizer, gan, recon_criterion, loader
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--batch-sizes', default='16,24,32,40,48')
    ap.add_argument('--n-batches', type=int, default=8, help="timed steps per batch size")
    opts = ap.parse_args()

    # parse_args() reads the config into a Namespace; feed it a clean argv so the probe's own
    # flags do not collide with the trainer's.
    sys.argv = ['probe', '--model', 'dualvae', '--config', opts.config]
    from tools.arguments import parse_args
    from tools.utils import set_seed

    args = parse_args()
    args.device = torch.device('cuda')
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)
    if getattr(args, 'allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    total = torch.cuda.get_device_properties(args.device).total_memory / 1024 ** 3
    print(f"\nGPU: {torch.cuda.get_device_name(args.device)}  ({total:.0f} GiB)")
    print(f"config: {opts.config}  |  resize_img={args.resize_img}  "
          f"perceptual_loss={getattr(args, 'perceptual_loss', 'none')}  "
          f"use_gan={getattr(args, 'use_gan', False)}\n")
    print(f"{'batch':>6} {'peak VRAM':>12} {'util':>7} {'img/s/GPU':>11} {'x4 GPUs':>9}   verdict")
    print("-" * 72)

    results = []
    for bs in [int(x) for x in opts.batch_sizes.split(',')]:
        args.batch_size = bs
        try:
            peak, ips = probe_one(args, bs, opts.n_batches)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{bs:>6} {'OOM':>12} {'-':>7} {'-':>11} {'-':>9}   too big")
            continue
        util = peak / total
        # DDP adds gradient-reduction buckets and NCCL buffers on top of this single-process
        # figure, and activation peaks move a little with input content, so anything above
        # ~85% here is not a batch size to launch a 40-epoch run with.
        verdict = "OK" if util < 0.85 else "RISKY (no DDP headroom)"
        print(f"{bs:>6} {peak:>9.1f} GiB {util:>6.0%} {ips:>11.1f} {ips*4:>9.0f}   {verdict}")
        results.append((bs, peak, util, ips))

    ok = [r for r in results if r[2] < 0.85]
    if ok:
        bs, peak, util, ips = max(ok, key=lambda r: r[0])
        epoch_s = 1281167 / (ips * 4)
        print(f"\nLargest safe per-GPU batch: {bs}  (peak {peak:.1f} GiB, {util:.0%})")
        print(f"  effective batch over 4 GPUs : {bs * 4}")
        print(f"  throughput                  : {ips*4:.0f} img/s")
        print(f"  one ImageNet epoch          : {epoch_s/60:.1f} min")
        print(f"  40 epochs + ~20% margin     : {40 * epoch_s * 1.2 / 3600:.1f} h  <- use for --time")
    print()


if __name__ == '__main__':
    main()
