#!/usr/bin/env python
"""
DINO-style feature visualization for a trained DualSSL (SwAV) checkpoint.

Same trick as the DINO/DINOv2 papers, applied to your SSL encoder: run PCA over the dense
per-location backbone features, paint the top-3 components as RGB (and PC1 as a heatmap), and
upsample back onto the image -- so you can literally *see* what the self-supervised features
have learned. As an SSL-specific bonus we also show the **prototype-assignment map**: the
codebook prototype each patch is assigned to (colored), i.e. the emergent segmentation.

One PCA is fit over all shown images' locations, so colors are consistent across the row
(same feature -> same color). Works on any DualSSL checkpoint (cosine or gaussian), and is
cheap -- a handful of images through the encoder.

Example
-------
    python tools/ssl_feature_pca_images.py \
        --checkpoint-dir checkpoints/dualssl/dualssl_20260726-XXXXXX_xxxxxx \
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

from models.dual_ssl import DualSSL
from tools.feature_pca_images import feature_rgb, feature_pc1   # reuse the PCA->RGB helpers

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENETTE_MEAN = (0.5, 0.5, 0.5)
IMAGENETTE_STD = (0.5, 0.5, 0.5)


def load_config(checkpoint_dir):
    import yaml
    cands = (glob.glob(os.path.join(checkpoint_dir, "config_used.yaml"))
             or glob.glob(os.path.join(checkpoint_dir, "*.yaml")))
    if not cands:
        raise FileNotFoundError(f"No config yaml in {checkpoint_dir}.")
    with open(cands[0]) as f:
        return yaml.safe_load(f)


def find_weights(checkpoint_dir, prefer="best"):
    """SSL: prefer best.pt (best kNN features); fall back to final_epoch.pt."""
    order = (("best.pt", "final_epoch.pt") if prefer == "best"
             else ("final_epoch.pt", "best.pt"))
    for name in order:
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt weights in {checkpoint_dir}.")
    return pts[0]


def build_model(cfg, device):
    def g(k, d=None):
        return cfg.get(k, d)
    model = DualSSL(
        latent_channels=g("latent_channels", 8), downsample_factor=g("downsample_factor", 8),
        num_prototypes=g("num_prototypes", 256), embedding_mode=g("embedding_mode", "global"),
        proj_dim=g("proj_dim", None), proj_hidden_dim=g("proj_hidden_dim", 256),
        l2_normalize_codes=g("l2_normalize_codes", True), temperature=g("temperature", 0.1),
        sinkhorn_eps=g("sinkhorn_eps", 0.05), sinkhorn_iters=g("sinkhorn_iters", 3),
        assignment=g("assignment", "cosine"), ema_decay=g("ema_decay", 0.99),
        sigma2_floor=g("sigma2_floor", 0.1), sigma2_ceil=g("sigma2_ceil", 10.0),
    )
    return model.to(device)


def eval_transform(size):
    return T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
                      T.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STD)])


def denorm(img):
    return (img * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def extract(model, imgs):
    """Dense backbone feature map (B,C,h,w) and per-location prototype-assignment map (B,h,w)."""
    z = model._backbone(imgs)                                   # (B, C, h, w)
    b, c, h, w = z.shape
    protos = F.normalize(model.prototypes.embedding.weight, dim=1)
    loc = z.permute(0, 2, 3, 1).reshape(b * h * w, c)
    if model.projector is not None:
        loc = model.projector(loc)
    loc = F.normalize(loc, dim=1)
    assign = (loc @ protos.t()).argmax(dim=1).reshape(b, h, w)  # (B, h, w) prototype id
    return z.cpu().numpy(), assign.cpu().numpy()


def fit_pca(fmaps, seed):
    pooled = np.concatenate([f.reshape(f.shape[0], -1).T for f in fmaps], 0)   # (sum HW, C)
    pca = PCA(n_components=3, random_state=seed).fit(pooled)
    proj = pca.transform(pooled)
    return pca, np.percentile(proj, 1, 0), np.percentile(proj, 99, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--dataset-dir", default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--weights", default="best", choices=["best", "final"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.checkpoint_dir)
    w = find_weights(args.checkpoint_dir, args.weights)
    print(f"[info] device={device} | assignment={cfg.get('assignment', 'cosine')} | weights={w}")
    model = build_model(cfg, device)
    sd = torch.load(w, map_location=device)
    model.load_state_dict(sd if "encoder.0.weight" in sd else sd.get("model_state_dict", sd))
    model.eval()

    size = cfg.get("resize_img", 224)
    tf = eval_transform(size)
    files = sorted(f for f in os.listdir(os.path.join(args.dataset_dir, args.split))
                   if f.lower().endswith(VALID_EXTS))
    files = random.Random(args.seed).sample(files, min(args.num_images, len(files)))
    imgs = torch.stack([tf(Image.open(os.path.join(args.dataset_dir, args.split, f)).convert("RGB"))
                        for f in files]).to(device)

    fmap, assign = extract(model, imgs)                         # (N,C,h,w), (N,h,w)
    pca, lo, hi = fit_pca([fmap[i:i + 1][0] for i in range(len(files))], args.seed)
    out_hw = tuple(imgs.shape[2:])

    n = len(files)
    cols = ["original", "PCA features", "PC1", "prototype map"]
    fig, axes = plt.subplots(n, 4, figsize=(4 * 2.3, n * 2.35))
    if n == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("tab20")
    for i in range(n):
        axes[i, 0].imshow(denorm(imgs[i]).cpu().permute(1, 2, 0).numpy())
        axes[i, 1].imshow(feature_rgb(fmap[i], pca, lo, hi, out_hw))
        axes[i, 2].imshow(feature_pc1(fmap[i], pca, lo, hi, out_hw), cmap="viridis")
        # prototype-assignment map: color by (prototype id mod 20), nearest-upsample.
        amap = torch.from_numpy((assign[i] % 20).astype(np.float32))[None, None]
        amap = F.interpolate(amap, size=out_hw, mode="nearest")[0, 0].numpy()
        axes[i, 3].imshow(cmap(amap / 19.0))
        for j in range(4):
            axes[i, j].axis("off")
        if i == 0:
            for j, t in enumerate(cols):
                axes[0, j].set_title(t, fontsize=10, fontweight="bold")

    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    fig.suptitle(f"DualSSL features projected onto the image (PCA->RGB)  |  {tag}", fontsize=11, y=1.005)
    fig.tight_layout()
    out = os.path.join(out_dir, f"ssl_feature_pca_{tag}.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
