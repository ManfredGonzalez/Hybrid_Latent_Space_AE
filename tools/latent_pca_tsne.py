#!/usr/bin/env python
"""
Latent-space visualization for the (SW_)DUALVAE dual-branch autoencoder.

Samples N random images per category from an Imagenette-style flat directory,
runs them through a trained DUALVAE / SW_DUALVAE checkpoint, and extracts three
per-image latent embeddings:

    1. VQ (discrete) branch          -> z_vq
    2. Continuous (vanilla) branch   -> z_vanilla_post
    3. Summed embedding              -> z_vq + z_vanilla_post
       (exactly the tensor fed into the attention block, i.e. the residual sum
        *before* self-attention)

Each latent is a (C, H/8, W/8) feature map; it is flattened to one vector per
image. For every embedding type the vectors are reduced with PCA (keeping the
components that explain most of the variance) and then projected to 2D with
t-SNE. One PDF scatter plot is produced per embedding type, with every category
drawn in a different color.

The category of an image is the first token of its filename split on "_"
(e.g. "n02979186_9036.JPEG" -> "n02979186").

Point the script at a different checkpoint directory (one that contains a
best.pt / *.pt weight file and a config_used.yaml) and it will rebuild the
matching model and regenerate the plots.

Example
-------
    python tools/latent_pca_tsne.py \
        --checkpoint-dir checkpoints/dualvae/dualvae_20260724-121508_392b5b \
        --dataset-dir /media/tico/BACKUP-DIDI/imagenette/imagenette2-320 \
        --split train --per-category 10 --pca-components 30
"""
import os
import re
import sys
import glob
import random
import argparse

# Make the repo root importable when the script is launched from tools/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.dual_vae import DUALVAE
from models.sw_dualvae import SW_DUALVAE

# Human-readable names for the 10 Imagenette synsets (nice legend labels; any
# category not in this map just falls back to its raw synset id).
IMAGENETTE_CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# The three embedding types we extract and plot, in a fixed order.
EMBEDDING_TYPES = ["vq", "continuous", "summed"]
EMBEDDING_TITLES = {
    "vq": "VQ (discrete) branch",
    "continuous": "Continuous (vanilla) branch",
    "summed": "Summed embedding (z_vq + z_continuous, pre-attention)",
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Config / model loading
# --------------------------------------------------------------------------- #
def load_config(checkpoint_dir):
    """Loads the *_used.yaml (or any *.yaml) config sitting next to the weights."""
    import yaml
    candidates = (
        glob.glob(os.path.join(checkpoint_dir, "config_used.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*used*.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*.yml"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .yaml config found in {checkpoint_dir}. Expected a config_used.yaml "
            "describing the architecture that produced the checkpoint."
        )
    cfg_path = candidates[0]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg_path


def find_weights(checkpoint_dir):
    """Finds the checkpoint weight file: prefer final_epoch.pt, then best.pt."""
    for name in ("final_epoch.pt", "final_epoch.pth", "best.pt", "best.pth"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))
                 + glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not pts:
        raise FileNotFoundError(f"No .pt/.pth weight file found in {checkpoint_dir}.")
    return pts[0]


def build_model(cfg, device):
    """Rebuilds the dual-branch model exactly as configured for the checkpoint."""
    model_name = str(cfg.get("model", "dualvae")).lower()

    common = dict(
        commitment_cost=cfg.get("commitment_cost", 0.25),
        latent_channels=cfg.get("latent_channels", 8),
        num_embeddings=cfg.get("num_embeddings", 256),
        downsample_factor=cfg.get("downsample_factor", 8),
        l2_normalize_codes=cfg.get("l2_normalize_codes", False),
        use_ema_codebook=cfg.get("use_ema_codebook", False),
        rq_depth=cfg.get("rq_depth", 1),
        residual_continuous=cfg.get("residual_continuous", False),
        component_prior=cfg.get("component_prior", False),
    )
    # sigma2_floor / sigma2_ceil only matter for buffer shapes/values, pass them through.
    common["sigma2_floor"] = cfg.get("sigma2_floor", 1e-3)
    common["sigma2_ceil"] = cfg.get("sigma2_ceil", 10.0)

    if model_name == "dualvae":
        model = DUALVAE(**common)
    elif model_name == "swd_dualvae":
        model = SW_DUALVAE(
            combine_mode=cfg.get("combine_mode", "residual_addition"),
            wavelet_detail=cfg.get("wavelet_detail", False),
            wavelet_band_channels=cfg.get("wavelet_band_channels", None),
            learned_band_variance=cfg.get("learned_band_variance", False),
            band_sigma0_prior=cfg.get("swd_sigma0_bands", None),
            **common,
        )
    else:
        raise ValueError(
            f"This visualizer only supports dual-branch models (dualvae / swd_dualvae); "
            f"got model={model_name!r}."
        )
    return model.to(device)


# --------------------------------------------------------------------------- #
# Data sampling
# --------------------------------------------------------------------------- #
def category_of(filename):
    """Category = first token of the filename (basename) split on '_'."""
    return os.path.basename(filename).split("_")[0]


def sample_images(images_dir, per_category, include_regex, seed):
    """Groups images in images_dir by category and samples `per_category` per group.

    Returns a list of (filepath, category) and the sorted list of categories used.
    """
    files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(VALID_EXTS)
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}.")

    pattern = re.compile(include_regex) if include_regex else None
    groups = {}
    for f in files:
        cat = category_of(f)
        if pattern is not None and not pattern.match(cat):
            continue
        groups.setdefault(cat, []).append(f)

    if not groups:
        raise ValueError(
            f"No categories matched include-regex={include_regex!r} in {images_dir}."
        )

    rng = random.Random(seed)
    selected = []
    used_categories = []
    skipped = []
    for cat in sorted(groups):
        pool = sorted(groups[cat])
        if len(pool) < per_category:
            skipped.append((cat, len(pool)))
            continue
        chosen = rng.sample(pool, per_category)
        used_categories.append(cat)
        for fp in chosen:
            selected.append((fp, cat))

    if skipped:
        print("[warn] categories with fewer than "
              f"{per_category} images (skipped): "
              + ", ".join(f"{c}({n})" for c, n in skipped))
    return selected, used_categories


def build_transform(resize_img, dataset_name):
    """Recreates the exact eval-time preprocessing used during training."""
    tfs = []
    if resize_img and resize_img != -1:
        tfs.append(T.Resize((resize_img, resize_img)))
    tfs.append(T.ToTensor())
    if str(dataset_name).lower() == "imagenette":
        tfs.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif str(dataset_name).lower() == "cifar10":
        tfs.append(T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]))
    return T.Compose(tfs)


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(model, samples, transform, device, batch_size, seed):
    """Runs the samples through the model and collects the three per-image latents.

    Returns dict{emb_type: (N, D) float32 array} and the list of category labels
    aligned with the rows.
    """
    # Fix the RNG so the continuous branch's reparameterization noise (present even
    # in eval mode) is reproducible across runs.
    torch.manual_seed(seed)

    embeddings = {t: [] for t in EMBEDDING_TYPES}
    labels = []

    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        imgs = torch.stack([transform(_load_rgb(fp)) for fp, _ in batch]).to(device)

        _, vq_losses, vanilla_losses = model(imgs, ablation_mode=-1)
        z_vq = vq_losses["z_vq"]                       # (B, C, H, W)
        z_cont = vanilla_losses["z_vanilla_post"]      # (B, C, H, W)
        z_sum = z_vq + z_cont                          # exactly the pre-attention residual sum

        for name, tensor in (("vq", z_vq), ("continuous", z_cont), ("summed", z_sum)):
            flat = tensor.reshape(tensor.shape[0], -1).float().cpu().numpy()
            embeddings[name].append(flat)

        labels.extend(cat for _, cat in batch)

    embeddings = {t: np.concatenate(v, axis=0) for t, v in embeddings.items()}
    return embeddings, labels


def _load_rgb(path):
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
# Dimensionality reduction + plotting
# --------------------------------------------------------------------------- #
def select_pca_components_parallel_analysis(x_std, seed, n_perm=20, cap=None,
                                            min_comps=2):
    """Horn's parallel analysis: keep only the leading PCA components whose variance
    beats the variance the *same-shaped random data* produces at that rank.

    This is the principled "signal vs noise" cut for t-SNE preprocessing, and unlike a
    fixed variance-percentage threshold it behaves correctly in the n << p regime
    (where variance is spread thin across many noise components).

    We permute each feature column independently (destroys cross-feature covariance,
    preserves each marginal), fit PCA, and take the 95th-percentile explained-variance
    curve over `n_perm` such null datasets. The number of real leading components that
    exceed this null curve -- up to the first crossing -- is the signal dimensionality.
    """
    n_samples, n_features = x_std.shape
    max_comp = min(n_samples - 1, n_features)
    if cap is not None:
        max_comp = min(max_comp, cap)

    real_evr = PCA(n_components=max_comp, random_state=seed).fit(
        x_std).explained_variance_ratio_

    rng = np.random.default_rng(seed)
    null_curves = np.empty((n_perm, max_comp))
    for i in range(n_perm):
        perm = np.empty_like(x_std)
        for j in range(n_features):
            perm[:, j] = x_std[rng.permutation(n_samples), j]
        # Columns are already standardized; re-standardize is a no-op up to sampling,
        # so fit PCA directly on the shuffled matrix.
        null_curves[i] = PCA(n_components=max_comp,
                             random_state=seed).fit(perm).explained_variance_ratio_
    null_threshold = np.percentile(null_curves, 95, axis=0)

    # Count leading components until the first one that fails the test.
    n_sig = 0
    for k in range(max_comp):
        if real_evr[k] > null_threshold[k]:
            n_sig += 1
        else:
            break
    return int(max(min_comps, min(n_sig, max_comp)))


def pca_then_tsne(features, pca_components, seed, perplexity=None, pca_cap=None):
    """Standardize -> PCA (keep top variance components) -> t-SNE to 2D.

    pca_components: an int for a fixed component count, or the string "auto" to pick
    the number by Horn's parallel analysis (bounded by pca_cap).
    """
    n_samples = features.shape[0]

    x = StandardScaler().fit_transform(features)

    if isinstance(pca_components, str) and pca_components.lower() == "auto":
        n_comp = select_pca_components_parallel_analysis(x, seed, cap=pca_cap)
    else:
        # Fixed count, never more than the data supports.
        n_comp = min(int(pca_components), n_samples - 1, x.shape[1])

    pca = PCA(n_components=n_comp, random_state=seed)
    x_pca = pca.fit_transform(x)
    explained = float(pca.explained_variance_ratio_.sum())

    if perplexity is None:
        # t-SNE needs perplexity < n_samples; a common safe rule is ~ n/3 capped at 30.
        perplexity = max(5.0, min(30.0, (n_samples - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    x_2d = tsne.fit_transform(x_pca)
    return x_2d, n_comp, explained, perplexity


def plot_embedding(x_2d, labels, categories, title, subtitle, out_path):
    """Scatter plot with one color per category, saved as a PDF."""
    cmap = plt.get_cmap("tab20" if len(categories) > 10 else "tab10")
    color_for = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = np.asarray(labels)
    for cat in categories:
        mask = labels == cat
        legend_name = IMAGENETTE_CLASS_NAMES.get(cat, cat)
        ax.scatter(
            x_2d[mask, 0], x_2d[mask, 1],
            s=42, alpha=0.85, edgecolors="black", linewidths=0.3,
            color=color_for[cat], label=legend_name,
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="0.35")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Category", fontsize=8, title_fontsize=9,
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing best.pt (or *.pt) and config_used.yaml.")
    p.add_argument("--dataset-dir",
                   default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320",
                   help="Root of the imagenette2-320 dataset (contains train/ and val/).")
    p.add_argument("--split", default="train",
                   help="Sub-folder of the dataset to sample from (e.g. train or val).")
    p.add_argument("--per-category", type=int, default=10,
                   help="Number of random images to sample per category.")
    p.add_argument("--pca-components", default="auto",
                   help="Number of top-variance PCA components fed into t-SNE. 'auto' "
                        "(default) picks the signal dimensionality per branch via Horn's "
                        "parallel analysis (best noise reduction); or pass an integer to "
                        "force a fixed count.")
    p.add_argument("--pca-cap", type=int, default=50,
                   help="Upper bound on components when --pca-components=auto.")
    p.add_argument("--include-regex", default=r"^n\d+",
                   help="Only keep categories whose name matches this regex. Default "
                        r"'^n\d+' restricts to genuine Imagenette synsets and drops the "
                        "stray ILSVRC2012_val_* files. Pass '' to include every category.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the PDFs (default: the checkpoint directory).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--perplexity", type=float, default=None,
                   help="t-SNE perplexity (default: auto from sample count).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: cuda if available else cpu).")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[info] device: {device}")

    # --- config + model ---
    cfg, cfg_path = load_config(args.checkpoint_dir)
    print(f"[info] config: {cfg_path}  (model={cfg.get('model')})")
    weights_path = find_weights(args.checkpoint_dir)
    print(f"[info] weights: {weights_path}")

    model = build_model(cfg, device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print("[info] checkpoint loaded.")

    # --- sample images ---
    images_dir = os.path.join(args.dataset_dir, args.split)
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Split directory not found: {images_dir}")
    samples, categories = sample_images(
        images_dir, args.per_category, args.include_regex, args.seed
    )
    print(f"[info] {len(categories)} categories, {len(samples)} images total "
          f"({args.per_category}/category) from {images_dir}")

    # --- preprocessing (must match training) ---
    transform = build_transform(
        cfg.get("resize_img", 256), cfg.get("dataset_name", "imagenette")
    )

    # --- extract the three embeddings ---
    embeddings, labels = extract_embeddings(
        model, samples, transform, device, args.batch_size, args.seed
    )

    # --- reduce + plot each embedding type ---
    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt_tag = os.path.basename(os.path.normpath(args.checkpoint_dir))

    for emb_type in EMBEDDING_TYPES:
        feats = embeddings[emb_type]
        print(f"\n[info] {emb_type}: features {feats.shape} -> PCA -> t-SNE")
        x_2d, n_comp, explained, perpl = pca_then_tsne(
            feats, args.pca_components, args.seed, args.perplexity, args.pca_cap
        )
        how = "auto/parallel-analysis" if str(args.pca_components).lower() == "auto" else "fixed"
        print(f"[info] {emb_type}: PCA -> {n_comp} comps ({how}, {explained*100:.1f}% var)")
        subtitle = (f"PCA {n_comp} comps ({how}, {explained*100:.1f}% var) -> t-SNE 2D  |  "
                    f"perplexity={perpl:.0f}  |  {args.per_category} imgs x "
                    f"{len(categories)} categories")
        title = f"{EMBEDDING_TITLES[emb_type]}"
        out_path = os.path.join(out_dir, f"tsne_{emb_type}_{ckpt_tag}.pdf")
        plot_embedding(x_2d, labels, categories, title, subtitle, out_path)

    print("\n[done] all plots written to:", out_dir)


if __name__ == "__main__":
    main()
