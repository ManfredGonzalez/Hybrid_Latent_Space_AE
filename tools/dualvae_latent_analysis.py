#!/usr/bin/env python
"""
DUALVAE latent-space analysis: posterior-collapse diagnostics + Gaussian-mixture test.

This script probes two claims about a trained **DUALVAE** (dual-branch VQ + continuous
autoencoder), loading the architecture from the config_used.yaml found in the given
checkpoint directory. It only supports the `dualvae` model.

--------------------------------------------------------------------------- #
Claim A -- the continuous branch does NOT suffer posterior collapse.
--------------------------------------------------------------------------- #
Collapse = the continuous posterior q(z|x) drifts to the prior N(0, I): variance -> 1,
mean -> 0, KL -> 0, so z carries no information and the decoder ignores it. We measure
the OPPOSITE with the standard VAE-collapse diagnostics, computed on the continuous
branch's (mean, log_variance):

  * Active Units (Burda et al. 2016): a latent unit u is "active" if
    A_u = Var_x(E[z_u | x]) > 0.01. Collapse => AU ~ 0.
  * Per-dimension KL to N(0, 1) (nats). Collapse => all ~ 0.
  * Distributions of the posterior mean (collapse spikes at 0) and posterior std
    (collapse spikes at 1).

Plots (PDF): KL per channel, active-units histogram, posterior mean/std histograms.

--------------------------------------------------------------------------- #
Claim B -- summing the branches (z = z_vq + z_continuous) forms a Gaussian MIXTURE,
           not a single isotropic Gaussian.
--------------------------------------------------------------------------- #
Per spatial location z = e_k + Delta: e_k is one of the K codebook vectors (the depth-1
code = mixture component identity) and Delta is the continuous offset. Marginally,
p(z) = sum_k pi_k N(e_k, Sigma_k) -- a GMM whose component means are the codewords and
whose weights pi_k are the code usage frequencies. We test whether that structure is
real and separated (vs one smeared blob):

  * Visual: PCA-2D (exact) and t-SNE-2D of per-location latents, colored by code, with
    the per-code discrete centroids overlaid.
  * Model selection: BIC of a K-component GMM vs a single Gaussian, on held-out
    locations (free-fit), AND the held-out log-likelihood of the CODE-ANCHORED mixture
    (means fixed to the empirical code centroids) vs a single Gaussian -- the latter
    tests the specific "codes are the mixture means" claim.
  * Separability ratio = (typical distance between code centroids) / (continuous
    residual std). >> 1 => well-separated components (the config's own design condition:
    "keep sigma0 well under the codeword spacing so mixture components stay separable").
  * Silhouette of locations labeled by code; effective #components = code perplexity.
  * Hartigan's dip test for multimodality along the axis joining the two most-used code
    centroids (unimodal single Gaussian would fail to be multimodal there).

Plots (PDF): code-colored PCA & t-SNE, code-usage (mixture weights) bar, and the
projection histogram along the top-2-code axis.

A JSON report with every number is written next to the plots.

Example
-------
    python tools/dualvae_latent_analysis.py \
        --checkpoint-dir checkpoints/dualvae/dualvae_20260724-121508_392b5b \
        --dataset-dir /media/tico/BACKUP-DIDI/imagenette/imagenette2-320 \
        --split val --num-images 200
"""
import os
import sys
import json
import glob
import random
import argparse

# Make the repo root importable when launched from tools/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

try:
    import diptest as _diptest
    _HAS_DIPTEST = True
except Exception:
    _HAS_DIPTEST = False

from models.dual_vae import DUALVAE

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ACTIVE_UNIT_THRESHOLD = 0.01  # Burda et al. active-unit variance threshold


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_config(checkpoint_dir):
    import yaml
    candidates = (
        glob.glob(os.path.join(checkpoint_dir, "config_used.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*used*.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*.yaml"))
        or glob.glob(os.path.join(checkpoint_dir, "*.yml"))
    )
    if not candidates:
        raise FileNotFoundError(f"No .yaml config found in {checkpoint_dir}.")
    with open(candidates[0], "r") as f:
        return yaml.safe_load(f), candidates[0]


def find_weights(checkpoint_dir):
    """Prefer final_epoch.pt, then best.pt (as requested)."""
    for name in ("final_epoch.pt", "final_epoch.pth", "best.pt", "best.pth"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Neither final_epoch.pt nor best.pt found in {checkpoint_dir}."
    )


def build_dualvae(cfg, device):
    model_name = str(cfg.get("model", "dualvae")).lower()
    if model_name != "dualvae":
        raise ValueError(
            f"This analysis is DUALVAE-only; the config says model={model_name!r}."
        )
    model = DUALVAE(
        commitment_cost=cfg.get("commitment_cost", 0.25),
        latent_channels=cfg.get("latent_channels", 8),
        num_embeddings=cfg.get("num_embeddings", 256),
        downsample_factor=cfg.get("downsample_factor", 8),
        l2_normalize_codes=cfg.get("l2_normalize_codes", False),
        use_ema_codebook=cfg.get("use_ema_codebook", False),
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
    )
    return model.to(device)


def build_transform(resize_img, dataset_name):
    tfs = []
    if resize_img and resize_img != -1:
        tfs.append(T.Resize((resize_img, resize_img)))
    tfs.append(T.ToTensor())
    if str(dataset_name).lower() == "imagenette":
        tfs.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif str(dataset_name).lower() == "cifar10":
        tfs.append(T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]))
    return T.Compose(tfs)


def sample_image_paths(images_dir, num_images, seed):
    files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(VALID_EXTS)
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}.")
    files.sort()
    rng = random.Random(seed)
    if num_images and num_images < len(files):
        files = rng.sample(files, num_images)
    return files


# --------------------------------------------------------------------------- #
# Forward pass: capture code indices, continuous (mean, logvar), and the summed latent
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_pass(model, paths, transform, device, batch_size, seed):
    """Replicates DUALVAE.forward's encode path so we can capture:
        - depth-1 code index per location   (mixture component identity)
        - continuous posterior mean/logvar  (collapse diagnostics)
        - z_vq (discrete) and z_sum = z_vq + z_continuous (per-location GMM samples)

    Returns per-image posterior mean/logvar arrays and flattened per-location arrays.
    """
    torch.manual_seed(seed)  # reproducible reparameterization noise
    model.eval()
    ds = model.downsample_factor
    C = model.latent_channels

    per_img_mean, per_img_logvar = [], []      # (N_img, C, H, W)
    loc_zsum, loc_zvq, loc_codes = [], [], []  # (N_loc, C) / (N_loc,)

    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        b, _, hh, ww = imgs.shape

        z_e = model.encoder(imgs)
        z_e_vq = model.bottle_neck_VQ(z_e)
        z_vq, _, code_idx, _, _ = model.vq_layer(z_e_vq)   # code_idx: depth-1, (b*h*w,)

        if getattr(model, "wavelet_detail", False):
            _, hf = model.dwt(imgs)
            z_e_vanilla = model.wavelet_meanvar(model.detail_encoder(hf))
        elif model.residual_continuous:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
        else:
            z_e_vanilla = model.vanilla_VAE_bottle_neck(z_e)
        noise = torch.randn((b, C, hh // ds, ww // ds), device=device)
        z_cont, mean, log_variance = model.forward_vanilla_z(z_e_vanilla, noise)
        z_sum = z_vq + z_cont

        per_img_mean.append(mean.float().cpu().numpy())
        per_img_logvar.append(log_variance.float().cpu().numpy())

        # (b, C, h, w) -> (b*h*w, C) channel-last to align with code_idx ordering.
        def flat(t):
            return t.permute(0, 2, 3, 1).reshape(-1, C).float().cpu().numpy()
        loc_zsum.append(flat(z_sum))
        loc_zvq.append(flat(z_vq))
        loc_codes.append(code_idx.cpu().numpy())

    return {
        "mean": np.concatenate(per_img_mean, 0),
        "logvar": np.concatenate(per_img_logvar, 0),
        "loc_zsum": np.concatenate(loc_zsum, 0),
        "loc_zvq": np.concatenate(loc_zvq, 0),
        "loc_codes": np.concatenate(loc_codes, 0),
        "codebook": model.vq_layer.embedding.weight.detach().float().cpu().numpy(),
        "perplexity": float(model.vq_layer.perplexity.item()),
    }


# --------------------------------------------------------------------------- #
# Claim A -- collapse diagnostics
# --------------------------------------------------------------------------- #
def analyze_collapse(mean, logvar, out_dir, tag):
    """mean/logvar: (N_img, C, H, W). Returns a metrics dict and writes 3 PDFs."""
    n_img, C, H, W = mean.shape
    var = np.exp(logvar)

    # Per-element KL(N(mu, var) || N(0,1)) = 0.5*(mu^2 + var - 1 - logvar), averaged over images.
    kl_elem = 0.5 * (mean ** 2 + var - 1.0 - logvar)   # (N_img, C, H, W)
    kl_mean_map = kl_elem.mean(axis=0)                 # (C, H, W)
    kl_per_channel = kl_mean_map.reshape(C, -1).mean(axis=1)  # (C,)
    kl_per_element = float(kl_mean_map.mean())
    kl_per_image = float(kl_elem.reshape(n_img, -1).sum(axis=1).mean())

    # Active units: variance across images of the posterior mean, per element.
    au_var = mean.reshape(n_img, -1).var(axis=0)       # (C*H*W,)
    active_mask = au_var > ACTIVE_UNIT_THRESHOLD
    active_count = int(active_mask.sum())
    active_frac = float(active_mask.mean())
    # Per-channel active fraction (share of that channel's spatial units that are active).
    au_var_map = au_var.reshape(C, H, W)
    active_frac_per_channel = (au_var_map > ACTIVE_UNIT_THRESHOLD).reshape(C, -1).mean(axis=1)

    std = np.sqrt(var)
    metrics = {
        "n_images": n_img, "latent_channels": C, "latent_hw": [H, W],
        "kl_nats_per_image": kl_per_image,
        "kl_nats_per_element": kl_per_element,
        "kl_per_channel": kl_per_channel.tolist(),
        "active_units_count": active_count,
        "active_units_total": int(au_var.size),
        "active_units_fraction": active_frac,
        "active_fraction_per_channel": active_frac_per_channel.tolist(),
        "posterior_mean_abs_avg": float(np.abs(mean).mean()),
        "posterior_std_avg": float(std.mean()),
        "posterior_std_median": float(np.median(std)),
    }

    # --- Plot 1: KL per channel ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(C), kl_per_channel, color="#3b7dd8", edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="crimson", ls="--", lw=1, label="collapse (KL=0)")
    ax.set_xlabel("continuous latent channel")
    ax.set_ylabel("mean KL to N(0,1)  [nats]")
    ax.set_title("Claim A: per-channel KL of the continuous branch\n"
                 f"(bars ≫ 0 ⇒ no collapse)  |  {active_count}/{au_var.size} active units",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, f"collapse_kl_per_channel_{tag}.pdf"))

    # --- Plot 2: active-units histogram ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(au_var + 1e-12), bins=60, color="#5aa469", edgecolor="black", linewidth=0.3)
    ax.axvline(np.log10(ACTIVE_UNIT_THRESHOLD), color="crimson", ls="--", lw=1.2,
               label=f"active threshold (0.01)")
    ax.set_xlabel("log10  Var_x(E[z|x])  per latent unit")
    ax.set_ylabel("number of latent units")
    ax.set_title(f"Claim A: active units = {active_count}/{au_var.size} "
                 f"({active_frac*100:.1f}%)\nmass right of the line = used dimensions",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, f"collapse_active_units_{tag}.pdf"))

    # --- Plot 3: posterior mean & std distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(mean.ravel(), bins=80, color="#7b6cd8", edgecolor="none")
    axes[0].axvline(0.0, color="crimson", ls="--", lw=1.2, label="collapse spike (mean=0)")
    axes[0].set_title("posterior mean  E[z|x]", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("value"); axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)
    axes[1].hist(std.ravel(), bins=80, color="#d88a3b", edgecolor="none")
    axes[1].axvline(1.0, color="crimson", ls="--", lw=1.2, label="collapse spike (std=1)")
    axes[1].set_title("posterior std  sqrt(Var[z|x])", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("value"); axes[1].set_ylabel("count"); axes[1].legend(fontsize=8)
    fig.suptitle("Claim A: away from the red lines ⇒ posterior differs from the prior "
                 "⇒ no collapse", fontsize=10, y=1.02)
    _save(fig, os.path.join(out_dir, f"collapse_posterior_stats_{tag}.pdf"))

    return metrics


# --------------------------------------------------------------------------- #
# Claim B -- Gaussian-mixture structure of z = z_vq + z_continuous
# --------------------------------------------------------------------------- #
def _per_code_centroids(loc_zvq, loc_codes, used):
    """Empirical discrete centroid per depth-1 code (robust to RQ depth>1, where the
    discrete part is a SUM of codewords). Returns dict{code -> (C,) centroid}."""
    cents = {}
    for k in used:
        cents[k] = loc_zvq[loc_codes == k].mean(axis=0)
    return cents


def analyze_gmm(data, out_dir, tag, seed, top_codes, max_tsne, max_fit):
    z = data["loc_zsum"]            # (N_loc, C) summed latent (the decoder's input space)
    zvq = data["loc_zvq"]
    codes = data["loc_codes"]
    codebook = data["codebook"]     # (K, C)
    C = z.shape[1]
    rng = np.random.default_rng(seed)

    # Code usage -> mixture weights pi_k.
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq, counts = uniq[order], counts[order]
    pi = counts / counts.sum()
    n_used = int(len(uniq))
    eff_components = float(np.exp(-(pi * np.log(pi + 1e-12)).sum()))  # perplexity of pi

    top = uniq[:top_codes]
    centroids = _per_code_centroids(zvq, codes, uniq)

    # --- Separability ratio: code-centroid spacing vs continuous residual std ---
    cent_arr = np.stack([centroids[k] for k in uniq], axis=0)        # (n_used, C)
    # Usage-weighted mean nearest-neighbour distance between distinct centroids.
    if n_used > 1:
        d2 = ((cent_arr[:, None, :] - cent_arr[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_dist = np.sqrt(d2.min(axis=1))
        mean_centroid_spacing = float((pi * nn_dist).sum())
    else:
        mean_centroid_spacing = 0.0
    # Within-component residual std: std of (z - centroid) over each code's locations.
    resid = z - np.stack([centroids[c] for c in codes], axis=0)
    residual_std = float(resid.std())
    separability_ratio = mean_centroid_spacing / (residual_std + 1e-12)

    # --- Model selection on held-out locations ---
    # Subsample for tractable fitting.
    idx = rng.permutation(len(z))[:max_fit]
    zf = z[idx]
    split = int(0.7 * len(zf))
    z_tr, z_te = zf[:split], zf[split:]

    single = GaussianMixture(n_components=1, covariance_type="full",
                             random_state=seed).fit(z_tr)
    n_gmm = min(n_used, 60)  # free-fit GMM with as many comps as used codes (capped)
    gmm = GaussianMixture(n_components=n_gmm, covariance_type="full",
                          reg_covar=1e-5, random_state=seed).fit(z_tr)
    single_bic = float(single.bic(z_te))
    gmm_bic = float(gmm.bic(z_te))
    single_ll = float(single.score(z_te))         # avg log-likelihood / sample
    gmm_ll = float(gmm.score(z_te))

    # --- Code-anchored mixture: means FIXED to code centroids (tests the exact claim) ---
    anchored_ll = _code_anchored_loglik(z_tr, z_te, codes[idx][:split], centroids,
                                        uniq, pi, C)

    # --- Silhouette (locations labeled by code), on a small subsample ---
    sidx = rng.permutation(len(z))[:min(4000, len(z))]
    lab = codes[sidx]
    keep = np.isin(lab, top)                        # silhouette only over the top codes
    sil = float("nan")
    if keep.sum() > 10 and len(np.unique(lab[keep])) > 1:
        sil = float(silhouette_score(z[sidx][keep], lab[keep]))

    # --- Multimodality along the MOST-SEPARATED centroid axis ---
    # The two most-USED codes need not be the two most-separated, so a dip test on their
    # axis can miss real multimodality (a weak-axis false negative). We instead pick, among
    # the top codes, the centroid PAIR with the largest separation and project onto that
    # axis -- the fairest 1-D probe for a mixture. We also fit a 2-component GMM on that
    # axis and report how well-separated the two modes are (distance / pooled std).
    dip_p = dip_stat = float("nan")
    axis_pair = None
    proj_all = None
    twomode_sep = float("nan")
    if n_used >= 2:
        cset = np.stack([centroids[k] for k in top], axis=0)     # (T, C)
        d2 = ((cset[:, None, :] - cset[None, :, :]) ** 2).sum(-1)
        ia, ib = np.unravel_index(np.argmax(d2), d2.shape)
        axis_pair = (int(top[ia]), int(top[ib]))
        axis = centroids[top[ia]] - centroids[top[ib]]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        proj_all = z @ axis
        pj = proj_all[rng.permutation(len(proj_all))[:20000]].reshape(-1, 1)
        if _HAS_DIPTEST:
            dip_stat, dip_p = _diptest.diptest(pj.ravel())
            dip_stat, dip_p = float(dip_stat), float(dip_p)
        g2 = GaussianMixture(n_components=2, covariance_type="full",
                             random_state=seed).fit(pj)
        m0, m1 = g2.means_.ravel()
        s0, s1 = np.sqrt(g2.covariances_.ravel())
        twomode_sep = float(abs(m0 - m1) / (0.5 * (s0 + s1) + 1e-12))

    metrics = {
        "n_locations": int(len(z)),
        "n_codes_used": n_used,
        "codebook_size": int(codebook.shape[0]),
        "code_perplexity_model": data["perplexity"],
        "effective_components_pi": eff_components,
        "mean_centroid_spacing": mean_centroid_spacing,
        "residual_std": residual_std,
        "separability_ratio": separability_ratio,
        "single_gaussian_bic": single_bic,
        "gmm_bic": gmm_bic,
        "bic_improvement_gmm_vs_single": single_bic - gmm_bic,
        "single_gaussian_heldout_loglik": single_ll,
        "free_gmm_heldout_loglik": gmm_ll,
        "code_anchored_mixture_heldout_loglik": anchored_ll,
        "anchored_vs_single_loglik_gain": anchored_ll - single_ll,
        "silhouette_top_codes": sil,
        "dip_statistic": dip_stat,
        "dip_pvalue": dip_p,
        "dip_axis_codes": axis_pair,              # most-SEPARATED centroid pair used
        "two_mode_separation": twomode_sep,       # |mu0-mu1| / avg std of a 2-comp GMM on that axis
        "diptest_available": _HAS_DIPTEST,
    }

    _plot_gmm_scatter(z, codes, top, centroids, codebook, seed, max_tsne, out_dir, tag,
                      separability_ratio, n_used)
    _plot_code_usage(uniq, pi, eff_components, out_dir, tag)
    if proj_all is not None:
        _plot_projection_hist(proj_all, centroids, axis_pair, dip_p, twomode_sep,
                              rng, out_dir, tag)
    return metrics


def _code_anchored_loglik(z_tr, z_te, codes_tr, centroids, uniq, pi, C):
    """Avg held-out log-likelihood of a mixture whose MEANS are the code centroids,
    weights are pi_k, and each component gets an isotropic within-code variance
    estimated from the training locations. Tests 'codewords are the mixture means'."""
    # Isotropic per-component variance from training residuals; fall back to global.
    var_k = {}
    global_resid = []
    for k in uniq:
        m = codes_tr == k
        if m.sum() >= 2:
            r = z_tr[m] - centroids[k]
            var_k[k] = float((r ** 2).mean())
            global_resid.append(r.ravel())
    gvar = float(np.concatenate(global_resid).var()) if global_resid else 1.0
    logpi = {k: np.log(p + 1e-12) for k, p in zip(uniq, pi)}
    const = -0.5 * C * np.log(2 * np.pi)

    # log N(x; mu_k, v_k I) = const - 0.5*C*log v_k - 0.5*||x-mu_k||^2 / v_k
    comp_logpdf = np.empty((len(z_te), len(uniq)), dtype=np.float64)
    for j, k in enumerate(uniq):
        v = max(var_k.get(k, gvar), 1e-6)
        diff2 = ((z_te - centroids[k]) ** 2).sum(axis=1)
        comp_logpdf[:, j] = logpi[k] + const - 0.5 * C * np.log(v) - 0.5 * diff2 / v
    # logsumexp over components.
    mx = comp_logpdf.max(axis=1, keepdims=True)
    ll = (mx.ravel() + np.log(np.exp(comp_logpdf - mx).sum(axis=1)))
    return float(ll.mean())


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def _plot_gmm_scatter(z, codes, top, centroids, codebook, seed, max_tsne, out_dir, tag,
                      sep_ratio, n_used):
    """PCA-2D (exact, with code centroids overlaid) and t-SNE-2D, colored by code."""
    cmap = plt.get_cmap("tab20")
    color_for = {k: cmap(i % 20) for i, k in enumerate(top)}
    rng = np.random.default_rng(seed)

    def draw(ax, xy, xy_cents, title):
        other = ~np.isin(codes_sub, top)
        ax.scatter(xy[other, 0], xy[other, 1], s=4, c="0.8", alpha=0.4,
                   linewidths=0, label="other codes")
        for k in top:
            m = codes_sub == k
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.75, linewidths=0,
                           color=color_for[k], label=f"code {int(k)}")
        if xy_cents is not None:
            ax.scatter(xy_cents[:, 0], xy_cents[:, 1], s=170, marker="*",
                       c="black", edgecolors="white", linewidths=0.8, zorder=5,
                       label="code centroids")
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Subsample locations for the scatter.
    sub = rng.permutation(len(z))[:max_tsne]
    z_sub = z[sub]
    codes_sub = codes[sub]

    # PCA-2D: exact, so we can transform the code centroids into the same plane.
    pca = PCA(n_components=2, random_state=seed).fit(z_sub)
    xy_pca = pca.transform(z_sub)
    cent_pca = pca.transform(np.stack([centroids[k] for k in top], axis=0))

    fig, ax = plt.subplots(figsize=(9, 7))
    draw(ax, xy_pca, cent_pca, f"Claim B: per-location latent z = z_vq + z_cont — PCA 2D\n"
                               f"clusters centered on code centroids ⇒ GMM  "
                               f"(separability={sep_ratio:.2f}, {n_used} codes used)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, ncol=1)
    _save(fig, os.path.join(out_dir, f"gmm_pca_codes_{tag}.pdf"))

    # t-SNE-2D: nonlinear view (no exact centroid projection; show empirical t-SNE means).
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(z_sub) // 100)),
                init="pca", learning_rate="auto", random_state=seed)
    xy_tsne = tsne.fit_transform(z_sub)
    cent_tsne = []
    for k in top:
        m = codes_sub == k
        cent_tsne.append(xy_tsne[m].mean(axis=0) if m.any() else [np.nan, np.nan])
    cent_tsne = np.array(cent_tsne)

    fig, ax = plt.subplots(figsize=(9, 7))
    draw(ax, xy_tsne, cent_tsne, "Claim B: per-location latent — t-SNE 2D\n"
                                 "(colored by code; ★ = per-code cluster mean)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, ncol=1)
    _save(fig, os.path.join(out_dir, f"gmm_tsne_codes_{tag}.pdf"))


def _plot_code_usage(uniq, pi, eff_components, out_dir, tag):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    show = min(40, len(uniq))
    ax.bar(np.arange(show), pi[:show], color="#c0567a", edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"code rank (top {show} of {len(uniq)} used)")
    ax.set_ylabel("mixture weight  π_k")
    ax.set_title("Claim B: mixture weights = code usage frequencies\n"
                 f"effective #components (perplexity of π) = {eff_components:.1f}",
                 fontsize=11, fontweight="bold")
    _save(fig, os.path.join(out_dir, f"gmm_code_usage_{tag}.pdf"))


def _plot_projection_hist(proj_all, centroids, axis_pair, dip_p, twomode_sep,
                          rng, out_dir, tag):
    ka, kb = axis_pair
    pj = proj_all[rng.permutation(len(proj_all))[:40000]]
    axis = centroids[ka] - centroids[kb]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c0 = centroids[ka] @ axis
    c1 = centroids[kb] @ axis
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(pj, bins=120, color="#4b9cd3", edgecolor="none")
    ax.axvline(c0, color="crimson", ls="--", lw=1.4, label=f"centroid code {ka}")
    ax.axvline(c1, color="darkgreen", ls="--", lw=1.4, label=f"centroid code {kb}")
    sub = "" if np.isnan(dip_p) else f"  |  dip p={dip_p:.3g} " \
          f"({'multimodal' if dip_p < 0.05 else 'unimodal'})"
    ax.set_title("Claim B: latent projected onto the MOST-SEPARATED code axis\n"
                 f"2-mode separation={twomode_sep:.2f}{sub}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"projection onto (centroid {ka} − centroid {kb})")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, f"gmm_projection_hist_{tag}.pdf"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True,
                   help="DUALVAE checkpoint dir (final_epoch.pt or best.pt + config_used.yaml).")
    p.add_argument("--dataset-dir",
                   default="/media/tico/BACKUP-DIDI/imagenette/imagenette2-320")
    p.add_argument("--split", default="val", help="Sub-folder to sample (train/val).")
    p.add_argument("--num-images", type=int, default=200,
                   help="Random images used to build the latent statistics.")
    p.add_argument("--top-codes", type=int, default=15,
                   help="How many of the most-used codes to color in the scatter plots.")
    p.add_argument("--max-tsne", type=int, default=6000,
                   help="Max per-location points fed to the scatter/t-SNE plots.")
    p.add_argument("--max-fit", type=int, default=40000,
                   help="Max per-location points used to fit the GMM/BIC comparison.")
    p.add_argument("--output-dir", default=None, help="Default: the checkpoint dir.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device: {device}")

    cfg, cfg_path = load_config(args.checkpoint_dir)
    print(f"[info] config: {cfg_path}  (model={cfg.get('model')})")
    weights = find_weights(args.checkpoint_dir)
    print(f"[info] weights: {weights}")

    model = build_dualvae(cfg, device)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print("[info] checkpoint loaded.")

    images_dir = os.path.join(args.dataset_dir, args.split)
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Split directory not found: {images_dir}")
    paths = sample_image_paths(images_dir, args.num_images, args.seed)
    print(f"[info] {len(paths)} images from {images_dir}")

    transform = build_transform(cfg.get("resize_img", 256),
                                cfg.get("dataset_name", "imagenette"))
    data = encode_pass(model, paths, transform, device, args.batch_size, args.seed)
    print(f"[info] latents: {data['loc_zsum'].shape[0]} locations "
          f"({data['mean'].shape[0]} images x {np.prod(data['mean'].shape[2:])} loc/img), "
          f"C={data['mean'].shape[1]}")

    out_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))

    print("\n[==] Claim A: posterior-collapse diagnostics (continuous branch)")
    collapse = analyze_collapse(data["mean"], data["logvar"], out_dir, tag)

    print("\n[==] Claim B: Gaussian-mixture test (z = z_vq + z_continuous)")
    gmm = analyze_gmm(data, out_dir, tag, args.seed,
                      args.top_codes, args.max_tsne, args.max_fit)

    report = {"checkpoint_dir": args.checkpoint_dir, "weights": weights,
              "split": args.split, "num_images": len(paths),
              "collapse": collapse, "gmm": gmm}
    report_path = os.path.join(out_dir, f"latent_analysis_{tag}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(collapse, gmm)
    print(f"\n[saved] numeric report: {report_path}")
    print("[done] plots + report written to:", out_dir)


def _print_summary(c, g):
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("-" * 68)
    print("Claim A -- posterior collapse (want: high AU, KL≫0, std≠1):")
    print(f"  active units          : {c['active_units_count']}/{c['active_units_total']} "
          f"({c['active_units_fraction']*100:.1f}%)")
    print(f"  KL nats / element     : {c['kl_nats_per_element']:.4f}")
    print(f"  KL nats / image       : {c['kl_nats_per_image']:.1f}")
    print(f"  posterior std (avg)   : {c['posterior_std_avg']:.3f}  "
          f"(1.0 would mean collapse)")
    print(f"  |posterior mean| avg  : {c['posterior_mean_abs_avg']:.3f}  "
          f"(0.0 would mean collapse)")
    print("\nClaim B -- Gaussian mixture (want: sep≫1, GMM BIC < single, dip p<0.05):")
    print(f"  codes used            : {g['n_codes_used']}/{g['codebook_size']}  "
          f"(effective comps ≈ {g['effective_components_pi']:.1f})")
    print(f"  separability ratio    : {g['separability_ratio']:.2f}  "
          f"(spacing {g['mean_centroid_spacing']:.3f} / resid std {g['residual_std']:.3f})")
    print(f"  BIC single Gaussian   : {g['single_gaussian_bic']:.0f}")
    print(f"  BIC K-comp GMM        : {g['gmm_bic']:.0f}   "
          f"(improvement {g['bic_improvement_gmm_vs_single']:.0f}; >0 favors mixture)")
    print(f"  held-out loglik       : single {g['single_gaussian_heldout_loglik']:.3f} | "
          f"code-anchored {g['code_anchored_mixture_heldout_loglik']:.3f} "
          f"(gain {g['anchored_vs_single_loglik_gain']:+.3f})")
    if not np.isnan(g["silhouette_top_codes"]):
        print(f"  silhouette (top codes): {g['silhouette_top_codes']:.3f}")
    if not np.isnan(g["two_mode_separation"]):
        print(f"  2-mode separation     : {g['two_mode_separation']:.2f}  "
              f"(most-separated code axis {g['dip_axis_codes']})")
    if not np.isnan(g["dip_pvalue"]):
        print(f"  Hartigan dip p-value  : {g['dip_pvalue']:.3g}  "
              f"({'MULTIMODAL' if g['dip_pvalue']<0.05 else 'not significant'})")
    print("=" * 68)


if __name__ == "__main__":
    main()
