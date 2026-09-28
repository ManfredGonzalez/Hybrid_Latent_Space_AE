"""Generate samples from a REPA-E checkpoint into ADM's .npz format, for offline evaluation.

    torchrun --standalone --nproc_per_node=4 generate_repae.py \
        --ckpt /gpfs/work3/0/prjs2260/checkpoints/repae_fsq_scratch/step_0100000.pt \
        --num-samples 50000 --cfg-scale 1.0 --num-steps 250 --mode sde

The .npz it writes is what the ADM / guided-diffusion evaluation suite consumes:
    https://github.com/openai/guided-diffusion/tree/main/evaluations
That suite (TensorFlow, run in its own environment, against ADM's reference batch) is what
produces gFID / sFID / IS / Precision / Recall on the same footing as REPA-E's tables. The
torchmetrics FID elsewhere in this repo is fine for comparing OUR arms to each other, but it
uses a different reference set and cannot be placed next to their published numbers.

THREE THINGS THIS SCRIPT HAS TO GET RIGHT, all of which are silent if wrong:
  * EMA WEIGHTS. Sampling from the raw weights instead of the EMA copy costs real FID. The
    checkpoint stores both; this loads `sit_ema`.
  * THE BATCHNORM STATISTICS. The SiT is trained on a normalized latent; the decoder needs the
    raw one. The running mean/var travel inside the SiT's state dict, and denormalizing with
    anything else (recomputed stats, another run's stats) puts the decoder off-distribution.
  * THE SAMPLER'S TIME CONVENTION, handled in tools/samplers_repae.py.

Sample count and per-rank sharding follow the reference implementation: every rank writes
`<index>.png` files into one folder, and rank 0 packs them into `<folder>.npz` at the end.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from experiments.train_repae import build_tokenizer, build_generator
from models.repr_encoder import DinoV2Features
from tools.distributed import ddp_setup, ddp_cleanup, is_dist, is_main_process, world_size
from tools.samplers_repae import euler_maruyama_sampler, euler_sampler


def load_from_checkpoint(ckpt_path, device):
    """Rebuild the tokenizer and the EMA SiT exactly as training left them."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**ckpt["args"])
    saved.device = str(device)

    vae, latent_channels, downsample = build_tokenizer(saved, device)
    vae.load_state_dict(ckpt["tokenizer"])
    vae.eval().requires_grad_(False)

    latent_size = saved.resize_img // downsample
    # z_dims only has to match what training used, so the projector shapes load; the
    # projectors are unused at sampling time.
    repr_dim = 1024 if "vitl" in getattr(saved, "repr_encoder", "dinov2_vitb14") else 768
    sit = build_generator(saved, latent_size, latent_channels, repr_dim, device)
    sit.load_state_dict(ckpt["sit_ema"])           # EMA weights, not the raw ones
    sit.eval().requires_grad_(False)

    step = ckpt.get("global_step", -1)
    if is_main_process():
        print(f"[ckpt] {ckpt_path} | epoch {ckpt.get('epoch')} | optimizer step {step}")
        mean, std = sit.latent_stats()
        print(f"[ckpt] latent std from BN: {[round(float(s), 4) for s in std]}")
    return vae, sit, saved, step


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="/gpfs/work3/0/prjs2260/samples")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32, help="per rank")
    p.add_argument("--mode", choices=["sde", "ode"], default="sde",
                   help="sde = Euler-Maruyama, what REPA-E reports; ode = deterministic Euler")
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--cfg-scale", type=float, default=1.0, help="1.0 disables guidance")
    p.add_argument("--guidance-low", type=float, default=0.0)
    p.add_argument("--guidance-high", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device, local_rank, rank, world = ddp_setup()
    torch.manual_seed(args.seed + rank)           # different noise per rank, reproducible
    vae, sit, saved, step = load_from_checkpoint(args.ckpt, device)
    mean, std = sit.latent_stats()
    mean, std = mean.view(1, -1, 1, 1).to(device), std.view(1, -1, 1, 1).to(device)

    tag = (f"{os.path.basename(args.ckpt).replace('.pt', '')}_{args.mode}{args.num_steps}"
           f"_cfg{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}")
    sample_dir = os.path.join(args.out_dir, tag)
    if is_main_process():
        os.makedirs(sample_dir, exist_ok=True)
        print(f"[out] {sample_dir}")
    if is_dist():
        dist.barrier()

    # Each rank takes a strided slice of the global index range, so the union is exactly
    # [0, num_samples) with no duplicates and no gaps regardless of world size.
    indices = list(range(rank, args.num_samples, world))
    latent_size = saved.resize_img // 8
    sampler = euler_maruyama_sampler if args.mode == "sde" else euler_sampler

    pbar = tqdm(total=len(indices), desc=f"rank {rank}", disable=not is_main_process())
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start:start + args.batch_size]
        n = len(chunk)
        # Class-balanced labels, matching the usual ImageNet evaluation protocol: sample the
        # label uniformly rather than following the training distribution.
        y = torch.randint(0, saved.num_classes, (n,), device=device)
        noise = torch.randn((n, sit.in_channels, latent_size, latent_size), device=device)

        z_norm = sampler(sit, noise, y, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                         guidance_low=args.guidance_low, guidance_high=args.guidance_high,
                         null_class=saved.num_classes)
        images = vae.decode_latent((z_norm * std + mean).to(next(vae.parameters()).dtype))
        images = ((images.float() + 1.0) / 2.0).clamp(0, 1)
        images = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for img, idx in zip(images, chunk):
            Image.fromarray(img).save(os.path.join(sample_dir, f"{idx:06d}.png"))
        pbar.update(n)
    pbar.close()

    if is_dist():
        dist.barrier()
    if is_main_process():
        npz_path = create_npz_from_sample_folder(sample_dir, args.num_samples)
        print(f"\nRun the ADM suite on it (separate TensorFlow environment):\n"
              f"  python evaluations/evaluator.py <reference_batch>.npz {npz_path}\n")
    ddp_cleanup()


def create_npz_from_sample_folder(sample_dir, num):
    """Pack <sample_dir>/000000.png ... into <sample_dir>.npz, ADM's expected layout."""
    samples = []
    for i in tqdm(range(num), desc="building .npz"):
        samples.append(np.asarray(Image.open(f"{sample_dir}/{i:06d}.png")).astype(np.uint8))
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3), samples.shape
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"saved {npz_path} [shape={samples.shape}]")
    return npz_path


if __name__ == "__main__":
    main()
