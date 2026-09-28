"""gFID / KID for a generated .npz, scored with torchmetrics against real images.

    python tools/score_npz_fid.py --npz <samples>.npz \
        --dataset-path /data/image-models-project/datasets/imagenet_full_256.h5

WHY THIS EXISTS, AND WHAT IT IS NOT. REPA-E's tables come from the ADM / guided-diffusion
evaluation suite: TensorFlow, run against ADM's precomputed reference batch over the ImageNet
TRAINING set. That suite is the only way to produce a number directly comparable to theirs.
This script is the pragmatic alternative: same Inception lineage (torchmetrics uses the ported
TF-Inception that the original FID used), but a different reference set and different
resizing, so the absolute value will NOT match theirs.

Use it to compare OUR checkpoints against each other -- which is what a training curve needs --
and run the ADM suite for anything that goes in a table beside their numbers. Mixing the two
sources in one table would be wrong even if the numbers looked plausible.

The reference set defaults to the ImageNet VALIDATION split, which is also what our rFID uses,
so gFID and rFID here are at least on the same footing as each other.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import HDF5ImageDataset, build_image_transform


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, nargs="+", help="one or more generated .npz files")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--num-real", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--kid-subset", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=None, help="JSON output (default: next to the first npz)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    # normalize=True means the metrics expect float images in [0, 1].
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(normalize=True, subset_size=args.kid_subset).to(device)

    # --- real images -------------------------------------------------------------------
    # Exactly the training preprocessing, then mapped [-1,1] -> [0,1]. A different resize here
    # would change the reference statistics and quietly shift every number.
    dataset = HDF5ImageDataset(args.dataset_path, args.split,
                               transform=build_image_transform("imagenet", 256), labeled=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    seen = 0
    for batch in tqdm(loader, desc=f"real ({args.split})"):
        if seen >= args.num_real:
            break
        x = batch["image"].to(device)
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        fid.update(x, real=True)
        kid.update(x, real=True)
        seen += x.shape[0]
    print(f"[real] {seen} images from the {args.split} split")

    # --- generated images, one npz at a time -------------------------------------------
    results = {"reference": {"split": args.split, "num_real": seen}, "runs": {}}
    for npz_path in args.npz:
        # Reset only the fake side: the real statistics are shared across every npz, which is
        # both faster and removes a source of difference between runs.
        fid.fake_features = []
        kid.fake_features = []
        arr = np.load(npz_path)["arr_0"]
        for i in tqdm(range(0, len(arr), args.batch_size), desc=os.path.basename(npz_path)):
            chunk = torch.from_numpy(arr[i:i + args.batch_size]).to(device)
            x = chunk.permute(0, 3, 1, 2).float() / 255.0
            fid.update(x, real=False)
            kid.update(x, real=False)
        kid_mean, kid_std = kid.compute()
        results["runs"][os.path.basename(npz_path)] = {
            "num_fake": int(len(arr)),
            "fid": float(fid.compute()),
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
        }
        print(f"{os.path.basename(npz_path)}: FID {results['runs'][os.path.basename(npz_path)]['fid']:.3f} "
              f"| KID {kid_mean:.5f}")

    out = args.out or os.path.join(os.path.dirname(args.npz[0]), "gfid_torchmetrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("NOTE: torchmetrics FID against the val split -- comparable across OUR checkpoints, "
          "NOT with REPA-E's ADM-suite numbers.")


if __name__ == "__main__":
    main()
