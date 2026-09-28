"""Representation alignment (CKNNA) between the SiT's hidden states and DINOv2 features.

    python eval_cknna.py --ckpt /projects/prjs2260/checkpoints/repae_fsq_scratch/step_0050000.pt

WHAT THIS MEASURES. REPA-E's argument runs: end-to-end tuning raises representation alignment
(their Fig. 3c, where vanilla REPA saturates near 0.42 and back-propagating into the tokenizer
breaks past it), and higher alignment predicts better generation (Fig. 3b). CKNNA is the
statistic behind both. Tracking it across our checkpoints tests whether the same mechanism is
operating on our tokenizer -- which is the assumption the whole design rests on.

REPA-E DOES NOT SHIP THIS. Their released code contains no CKNNA implementation; the metric
comes from Huh et al. 2024 ("The Platonic Representation Hypothesis"). What follows is the
standard formulation -- CKA restricted to mutually-nearest-neighbour pairs -- and several
choices they do not document (k, which features, which timestep, how many images) will shift
the absolute value. Trends across OUR checkpoints are sound; equality with their printed
numbers is not claimed.

WHY THE RAW HIDDEN STATE, NOT THE PROJECTED ONE. The REPA loss operates through a trained
projector. Measuring alignment after it would conflate "the SiT's features became DINOv2-like"
with "the projector learned to map them onto DINOv2". CKNNA asks the first question, so this
reads the block-8 hidden state directly (SiT.features) and reports the projected cosine
similarity separately, as the training-time reference.

WHY A TIMESTEP SWEEP. The SiT sees a noised latent, so its features depend on t. At t near 0
the input is nearly pure noise and carries little about the image; near 1 it is nearly clean.
A single t would be an arbitrary choice presented as a number, so this reports several.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import HDF5ImageDataset, build_image_transform
from generate_repae import load_from_checkpoint
from models.repr_encoder import DinoV2Features


def hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012), as used by the reference implementation."""
    n = K.shape[0]
    Kt = K.clone().fill_diagonal_(0)
    Lt = L.clone().fill_diagonal_(0)
    term1 = (Kt * Lt.t()).sum()
    term2 = Kt.sum() * Lt.sum() / ((n - 1) * (n - 2))
    term3 = 2 * (Kt.sum(dim=0) @ Lt.sum(dim=1)) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def cknna(x, y, topk=10, eps=1e-6):
    """Mutual-kNN-restricted kernel alignment between (n, d1) and (n, d2) feature matrices.

    CKA is a normalized HSIC between two similarity kernels. CKNNA (Huh et al. 2024) restricts
    it to k-nearest-neighbour structure, so it measures whether the two representations agree
    about LOCAL neighbourhoods rather than about global similarity -- the property that matters
    when asking whether one space could stand in for the other.

    THE NORMALIZATION IS THE WHOLE TRICK, and getting it wrong is not visibly wrong. A first
    version here restricted the numerator AND both denominators to the mutual mask. Masked
    pairs are by construction the largest entries of both kernels, so that ratio tends to 1 for
    ANY pair of feature sets: independent random features scored 0.975 and a row-shuffled copy
    of the same features scored 0.978. The fix, following the reference: the numerator uses the
    MUTUAL mask (agreement between the two), while each denominator uses its OWN self-kNN mask.
    The sanity cases then behave -- see tools/test_cknna.py.
    """
    n = x.shape[0]
    if topk >= n:
        raise ValueError(f"topk ({topk}) must be smaller than the number of samples ({n}).")
    if topk < 2:
        raise ValueError("CKNNA needs topk >= 2.")
    K = x @ x.t()
    L = y @ y.t()

    def _knn_mask(M):
        M = M.clone().fill_diagonal_(float("-inf"))   # a point is not its own neighbour
        idx = M.topk(topk, dim=1).indices
        return torch.zeros_like(M).scatter_(1, idx, 1.0)

    mask_K, mask_L = _knn_mask(K), _knn_mask(L)
    mask = mask_K * mask_L                            # mutual neighbours, for the numerator

    sim_kl = hsic_unbiased(mask * K, mask * L)
    sim_kk = hsic_unbiased(mask_K * K, mask_K * K)    # each denominator: its OWN mask
    sim_ll = hsic_unbiased(mask_L * L, mask_L * L)
    return float(sim_kl / (torch.sqrt(sim_kk * sim_ll) + eps))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset-path", default="/gpfs/work3/0/prjs2260/imagenet_full_256.h5")
    p.add_argument("--num-images", type=int, default=2048,
                   help="kernels are num_images^2; 2048 is 16 MB and plenty for a stable value")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="in OUR convention: t=0 noise, t=1 data")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON path (default: alongside the checkpoint)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    vae, sit, saved, step = load_from_checkpoint(args.ckpt, device)
    dino = DinoV2Features(getattr(saved, "repr_encoder", "dinov2_vitb14"),
                          getattr(saved, "repr_image_size", 224)).to(device)
    mean, std = sit.latent_stats()
    mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)

    # The VALIDATION split: alignment on data the tokenizer was not fitted to.
    dataset = HDF5ImageDataset(args.dataset_path, "val",
                               transform=build_image_transform("imagenet", saved.resize_img),
                               labeled=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    feats = {t: [] for t in args.timesteps}
    dino_patch, dino_cls, proj_cos = [], [], []
    seen = 0
    for batch in tqdm(loader, desc="features"):
        if seen >= args.num_images:
            break
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)
        patch_tokens, cls_token = dino(images)

        z, _ = vae.encode_latent(images, sample=False)
        z_norm = (z.float() - mean) / std
        # ONE noise draw shared by every timestep, so differences across t are about t and
        # not about which noise happened to be sampled.
        noise = torch.randn_like(z_norm)

        for t in args.timesteps:
            x_t = (1.0 - t) * noise + t * z_norm
            t_vec = torch.full((images.shape[0],), t, device=device)
            h = sit.features(x_t.to(z.dtype), t_vec, labels)          # (B, T, D)
            feats[t].append(h.float().mean(dim=1).cpu())               # pool over tokens

        dino_patch.append(patch_tokens.float().mean(dim=1).cpu())
        dino_cls.append(cls_token.float().cpu())

        # The training-time alignment number, for continuity: patch-wise cosine similarity
        # through the trained projector, at the middle timestep.
        t_mid = args.timesteps[len(args.timesteps) // 2]
        x_t = (1.0 - t_mid) * noise + t_mid * z_norm
        t_vec = torch.full((images.shape[0],), t_mid, device=device)
        h_mid = sit.features(x_t.to(z.dtype), t_vec, labels)
        n_, t_, d_ = h_mid.shape
        z_tilde = sit.projectors[0](h_mid.reshape(-1, d_)).reshape(n_, t_, -1)
        proj_cos.append((F.normalize(z_tilde.float(), dim=-1)
                         * F.normalize(patch_tokens.float(), dim=-1)).sum(-1).mean(-1).cpu())
        seen += images.shape[0]

    n = min(seen, args.num_images)
    dino_patch = torch.cat(dino_patch)[:n].to(device)
    dino_cls = torch.cat(dino_cls)[:n].to(device)

    results = {
        "checkpoint": args.ckpt,
        "optimizer_step": step,
        "epoch": int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]),
        "num_images": n,
        "topk": args.topk,
        "repa_cosine_projected": float(torch.cat(proj_cos)[:n].mean()),
        "cknna": {},
    }
    for t in args.timesteps:
        h = torch.cat(feats[t])[:n].to(device)
        results["cknna"][f"t={t}"] = {
            "vs_dino_patch_mean": cknna(h, dino_patch, topk=args.topk),
            "vs_dino_cls": cknna(h, dino_cls, topk=args.topk),
        }

    out = args.out or os.path.join(os.path.dirname(args.ckpt),
                                   f"cknna_{os.path.basename(args.ckpt).replace('.pt', '')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
