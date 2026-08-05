"""How low can Gen/FID possibly go under our evaluation protocol?

Gen/FID is reported against 3925 real validation images using 3925 generated samples. FID is a
BIASED estimator: it compares two 2048-dimensional Gaussians whose means and covariances are
estimated from finitely many samples, and at N=3925 the 2048x2048 covariance is estimated from
barely twice as many samples as it has dimensions. Two disjoint sets of REAL images drawn from the
same distribution therefore do NOT score 0 -- they score the estimator's noise floor.

That floor is the number a perfect generator would achieve. Without it, "FID 28.75" is
uninterpretable: it could be 28 points of model error, or 12 points of floor plus 16 of model
error. This script measures it directly.

Three references, all at the SAME N as the generative evaluation:
  train-vs-val   : disjoint real sets, same distribution -> finite-sample floor + any split shift.
                   This is the honest floor for our protocol.
  val-vs-val     : val split randomly halved, then each half scored against the other, subsampled
                   to the same N. Isolates the pure finite-sample effect with no split shift, but
                   at half the sample count so it reads higher.
  train-vs-train : two disjoint halves of train. Same idea, cross-checks the above.

Usage:
    python -m tools.fid_floor --dataset_path /path/to/imagenette2-320 --n 3925
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_labeled_datasets
from tools.normalization import denormalize
from tools.utils import select_device


@torch.no_grad()
def features(fid, dataset, indices, cfg_name, device, fid_device, real, bs=32, desc="feats"):
    loader = DataLoader(Subset(dataset, indices), batch_size=bs, shuffle=False, num_workers=4)
    n = 0
    for batch in tqdm(loader, desc=desc, unit="batch", mininterval=10.0, ncols=80):
        imgs = denormalize(batch["image"].float().to(device), cfg_name, device).clamp(0, 1)
        fid.update(imgs.to(fid_device), real=real)
        n += imgs.shape[0]
    return n


def score(a_ds, a_idx, b_ds, b_idx, dsname, device, fid_device, label):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).to(fid_device)
    na = features(fid, a_ds, a_idx, dsname, device, fid_device, True, desc=f"{label} A")
    nb = features(fid, b_ds, b_idx, dsname, device, fid_device, False, desc=f"{label} B")
    v = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    print(f"  {label:<16} {v:8.3f}   (N = {na} vs {nb})", flush=True)
    return {"label": label, "fid": v, "n_a": na, "n_b": nb}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--dataset_name", default="imagenette")
    p.add_argument("--resize_img", default=256, type=int)
    p.add_argument("--n", default=3925, type=int,
                   help="Samples per side; match the generative evaluation (default 3925).")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="reports/fid_floor.json",
                   help="Where to persist the measured floor. These numbers are cheap to "
                        "compute but easy to lose if they only ever land in a console log.")
    args = p.parse_args()

    device = select_device(args.device)
    trainset, valset = get_labeled_datasets(args.dataset_name, path=args.dataset_path,
                                            resize_img=args.resize_img, seed=args.seed)
    print(f"[data] train={len(trainset)}  val={len(valset)}")

    g = torch.Generator().manual_seed(args.seed)
    tr = torch.randperm(len(trainset), generator=g).tolist()
    va = torch.randperm(len(valset), generator=g).tolist()

    print(f"\nFID floor at N={args.n} per side (lower bound for ANY generator):")
    out = []
    # The protocol reference: reals are the val split, so this is what a perfect generator
    # producing genuine ImageNet-like images (but not these exact ones) would score.
    out.append(score(trainset, tr[:args.n], valset, va[:args.n],
                     args.dataset_name, device, device, "train-vs-val"))
    # Pure finite-sample effects, no train/val shift -- necessarily at half the sample count.
    half = min(len(valset) // 2, args.n)
    out.append(score(valset, va[:half], valset, va[half:2 * half],
                     args.dataset_name, device, device, "val-vs-val"))
    out.append(score(trainset, tr[:args.n], trainset, tr[args.n:2 * args.n],
                     args.dataset_name, device, device, "train-vs-train"))

    print("\nRead: a generator scoring X has roughly X - (train-vs-val) of genuine model error;")
    print("the remainder is the estimator's finite-sample floor at this N.")

    import json
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dataset_path": args.dataset_path, "n_per_side": args.n,
                   "resize_img": args.resize_img, "seed": args.seed,
                   "references": out}, f, indent=2)
    print(f"Wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
