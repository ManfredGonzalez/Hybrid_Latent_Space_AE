"""PCA view of the end-to-end tokenizer's latent space, in the style of REPA-E Fig. 3a / 5.

    python tools/latent_pca_repae.py \
        --ckpts /data/image-models-project/manfred/eval_export/eval_step_00{5,10,15,20}0000.pt \
        --dataset-path /data/image-models-project/datasets/imagenet_full_256.h5

WHAT IS SHOWN. REPA-E (following their ref. [24]) projects each image's latent, a (C, h, w)
map, onto its top-3 principal components and shows them as R, G, B at latent resolution. Noisy,
high-frequency colour means a latent the denoiser has to fight; a flat wash means an
over-smoothed one; object structure means the latent carries semantics.

PCA is fit PER IMAGE and PER COLUMN, as in the paper, so colours are not comparable across
cells -- only spatial structure is. A PC's sign is arbitrary, so the same structure can appear
with inverted colours from one checkpoint to the next.

Two figures:
  1. latent_pca_steps.png -- RGB | the latent the SiT sees (z = e_k + Delta, posterior mean) at
     each checkpoint: how end-to-end training reshapes the space.
  2. latent_pca_parts.png -- at the LAST checkpoint, z split into its two parts: the FSQ code
     e_k and the continuous residual Delta. This is the hybrid-specific view the paper has no
     analogue for.

Both end with a DINOv2 column: the same PCA over DINOv2's patch tokens, i.e. the features the
REPA loss aligns the SiT to. By default it uses the checkpoint's own repr_encoder and
repr_image_size (dinov2_vitb14 at 224px -> a 16x16 grid, coarser than the 32x32 latent);
--dino-size 448 gives a 32x32 grid at the latent's resolution.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import HDF5ImageDataset, build_image_transform  # noqa: E402
from generate_repae import load_from_checkpoint  # noqa: E402
from models.repr_encoder import DinoV2Features  # noqa: E402


def pca_rgb(z):
    """(C, h, w) -> (h, w, 3) in [0, 1]: top-3 PCs of the h*w vectors, each min-max scaled."""
    c, h, w = z.shape
    x = z.reshape(c, -1).t().double()
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    p = x @ v[:3].t()
    lo, hi = p.min(0, keepdim=True).values, p.max(0, keepdim=True).values
    return ((p - lo) / (hi - lo + 1e-8)).reshape(h, w, 3).float().numpy()


def step_label(path):
    m = re.search(r"step_0*(\d+)", os.path.basename(path))
    return f"{int(m.group(1)) // 1000}k steps" if m else os.path.basename(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="eval_step_*.pt, oldest first")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="val indices to show instead of a random draw")
    p.add_argument("--dino-size", type=int, default=None,
                   help="DINOv2 input side (multiple of 14); default: the REPA training value")
    p.add_argument("--out-dir", default="reports/latent_pca")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Random val images: the val split is sorted by class, so the first N would all be fish.
    dataset = HDF5ImageDataset(args.dataset_path, "val",
                               transform=build_image_transform("imagenet", 256), labeled=True)
    idx = args.indices or torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_images].tolist()
    images = torch.stack([dataset[i]["image"] for i in idx]).to(device)
    print(f"[data] val indices {idx}")

    cols, parts = [], None
    for ck in args.ckpts:
        vae, _, saved, step = load_from_checkpoint(ck, device)
        if saved.resize_img != 256:
            raise ValueError(f"{ck} was trained at {saved.resize_img}px; images here are 256px")
        # sample=False: the posterior mean, so the picture is the latent's structure and not
        # one draw of reparameterization noise.
        z, aux = vae.encode_latent(images, sample=False)
        cols.append((step_label(ck), z.float().cpu()))
        if ck == args.ckpts[-1]:
            parts = {"z = FSQ code + residual": z.float().cpu(),
                     "FSQ code  e_k": aux["z_vq"].float().cpu(),
                     "continuous residual  Delta": aux["mean"].float().cpu()}
            last = step_label(ck)
        del vae
        torch.cuda.empty_cache()

    dino_name = getattr(saved, "repr_encoder", "dinov2_vitb14")
    dino_size = args.dino_size or getattr(saved, "repr_image_size", 224)
    dino = DinoV2Features(dino_name, dino_size).to(device)
    patch, _ = dino(images)                                        # (B, g*g, D)
    g = dino.grid_size
    dino_col = (f"DINOv2 ({dino_name.split('_')[-1]}, {g}x{g})",
                patch.float().cpu().transpose(1, 2).reshape(len(idx), -1, g, g))

    rgb = (images.float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    def grid(columns, title, fname):
        n, m = len(idx), len(columns) + 1
        fig, axes = plt.subplots(n, m, figsize=(1.9 * m, 1.9 * n), squeeze=False)
        for r in range(n):
            axes[r, 0].imshow(rgb[r])
            for c, (_, z) in enumerate(columns, start=1):
                axes[r, c].imshow(pca_rgb(z[r]), interpolation="nearest")
        for c, name in enumerate(["RGB image"] + [name for name, _ in columns]):
            axes[0, c].set_title(name, fontsize=9)
        for a in axes.flat:
            a.set_xticks([]), a.set_yticks([])
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    c, h, w = cols[0][1].shape[1:]
    grid(cols + [dino_col],
         f"PCA of the SiT's input latent ({c}x{h}x{w}) across end-to-end training, vs DINOv2",
         "latent_pca_steps.png")
    grid(list(parts.items()) + [dino_col], f"Latent parts at {last}, vs DINOv2",
         "latent_pca_parts.png")


if __name__ == "__main__":
    main()
