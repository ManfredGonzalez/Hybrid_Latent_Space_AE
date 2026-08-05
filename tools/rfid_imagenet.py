"""Reconstruction quality (rFID + PSNR/SSIM/LPIPS) of a trained autoencoder on ARBITRARY
image sets -- built to place our Imagenette-trained models against ImageNet-scale numbers.

WHY THIS EXISTS, AND THE TRAP IT DOCUMENTS
------------------------------------------
The obvious way to ask "how would my autoencoder do on ImageNet?" is to run it on ImageNet's
validation images for the ten Imagenette classes. That evaluation is nearly worthless, and the
reason is worth recording:

    ImageNet train, 10 classes  12894
    ImageNet val,   10 classes    500
    total                       13394
    Imagenette train + val      9469 + 3925 = 13394        <-- identical

Imagenette IS the complete set of ImageNet images for those ten classes. There is no held-out
ImageNet data for them at all. Of the 500 ImageNet val images, 366 are in Imagenette's TRAIN
split (i.e. the model was fitted on them) and 134 are in its val split. So the "ImageNet val,
10 classes" number is 73% training data and is optimistically biased.

This script therefore supports splitting that set into its seen/unseen halves, so the
train-vs-val gap is visible rather than hidden, and -- more usefully -- evaluating the FULL
1000-class ImageNet validation set, which is the protocol SD-VAE (rFID 0.62) and RAE
(rFID 0.49) actually report. That is the only measurement here that is directly comparable to
published numbers, and it is a genuine generalization test: 990 of its classes were never seen.

Two caveats to carry into any comparison:
  * FID is biased upward at small N (the Inception covariance is 2048x2048). Comparing a
    500-image rFID against a 3925- or 50000-image rFID is meaningless, which is why
    `--subset imagenette-val-matched` exists: same N as the ImageNet-10 set, same protocol.
  * Published rFIDs use each paper's own preprocessing. Ours is the training preprocessing
    (Resize((256,256)), ToTensor, Normalize(0.5, 0.5)) -- no center crop, aspect ratio
    squashed. Numbers are comparable in spirit, not to the second decimal.

Usage
-----
    python -m tools.rfid_imagenet --checkpoint-dir checkpoints/dualvae/<B+ run> \
        --subset in10-val in10-val-seen in10-val-unseen imagenette-val-matched
    python -m tools.rfid_imagenet --checkpoint-dir <run> --subset imagenet-val-full
"""

import argparse
import json
import os
import random
import re
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dualvae_latent_analysis import (build_dualvae, build_transform, find_weights,
                                           load_config)

IMAGENET_ROOT = "/media/tico/BACKUP-DIDI/imageNet"
IMAGENETTE_ROOT = "/media/tico/BACKUP-DIDI/imagenette/imagenette2-320"

# The ten Imagenette classes, as ImageNet wnids.
WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
         "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]


def imagenet_val_labels(root=IMAGENET_ROOT):
    """{basename.JPEG: wnid} for ImageNet's flat val directory.

    CLS-LOC ships val labels only inside the per-image annotation XML, so they are parsed out
    here. An image with several annotated objects contributes its first <name>.
    """
    ann = os.path.join(root, "Annotations/CLS-LOC/val")
    pat = re.compile(r"<name>(n\d+)</name>")
    out = {}
    for fn in tqdm(sorted(os.listdir(ann)), desc="val labels", unit="xml",
                   mininterval=10.0, ncols=80):
        if not fn.endswith(".xml"):
            continue
        with open(os.path.join(ann, fn)) as f:
            m = pat.search(f.read())
        if m:
            out[fn[:-4] + ".JPEG"] = m.group(1)
    return out


def build_subset(name, seed=42):
    """-> (list of absolute image paths, human-readable description)."""
    val_dir = os.path.join(IMAGENET_ROOT, "Data/CLS-LOC/val")
    nette_tr = os.path.join(IMAGENETTE_ROOT, "train")
    nette_va = os.path.join(IMAGENETTE_ROOT, "val")

    if name == "imagenette-val":
        files = sorted(os.listdir(nette_va))
        return [os.path.join(nette_va, f) for f in files], \
               "Imagenette val (the model's own held-out split)"

    if name == "imagenet-val-full":
        files = sorted(f for f in os.listdir(val_dir) if f.endswith(".JPEG"))
        return [os.path.join(val_dir, f) for f in files], \
               "FULL ImageNet-1k val, 1000 classes -- comparable to SD-VAE / RAE rFID"

    # Everything below needs the 10-class ImageNet val subset.
    labels = imagenet_val_labels()
    keep = sorted(f for f, w in labels.items() if w in WNIDS)
    train_names = set(os.listdir(nette_tr))
    val_names = set(os.listdir(nette_va))

    if name == "in10-val":
        return [os.path.join(val_dir, f) for f in keep], \
               "ImageNet val, 10 Imagenette classes (WARNING: 73% was in the training set)"
    if name == "in10-val-seen":
        sel = [f for f in keep if f in train_names]
        return [os.path.join(val_dir, f) for f in sel], \
               "ImageNet val, 10 classes, SEEN during training (Imagenette train split)"
    if name == "in10-val-unseen":
        sel = [f for f in keep if f in val_names]
        return [os.path.join(val_dir, f) for f in sel], \
               "ImageNet val, 10 classes, held out (Imagenette val split)"
    if name == "imagenette-val-matched":
        # Same N as in10-val, so the two rFIDs are on the same side of FID's small-sample bias.
        files = sorted(os.listdir(nette_va))
        sel = random.Random(seed).sample(files, min(len(keep), len(files)))
        return [os.path.join(nette_va, f) for f in sel], \
               f"Imagenette val subsampled to N={len(keep)} (matched-N control for in10-val)"

    raise ValueError(f"unknown subset {name!r}")


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def evaluate(model, paths, transform, device, batch_size, kid_subset=100, desc="eval"):
    """Full-model reconstruction metrics, identical protocol to tools/reconstruction_ablation.py
    (denormalized [0,1] pixel space, AlexNet LPIPS, torchmetrics rFID/KID)."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    import lpips as lpips_lib

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=min(kid_subset, max(len(paths) // 2, 2)),
                                  normalize=True).to(device)

    acc = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
    model.eval()
    for start in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch",
                      mininterval=15.0, ncols=80):
        batch = paths[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB"))
                            for p in batch]).to(device)
        real = denorm(imgs)
        torch.manual_seed(1234)          # same reparameterization noise as the ablation tool
        recon = model(imgs, ablation_mode=-1)[0]
        fake = denorm(recon)
        bs = imgs.size(0)
        acc["mse"] += F.mse_loss(fake, real).item() * bs
        acc["psnr"] += psnr(fake, real).item() * bs
        acc["ssim"] += ssim(fake, real).item() * bs
        acc["lpips"] += lpips_fn(fake * 2 - 1, real * 2 - 1).mean().item() * bs
        acc["n"] += bs
        fid.update(real, real=True)
        fid.update(fake, real=False)
        kid.update(real, real=True)
        kid.update(fake, real=False)

    n = acc["n"]
    kid_mean, _ = kid.compute()
    return {"n_images": n,
            "mse": acc["mse"] / n, "psnr": acc["psnr"] / n,
            "ssim": acc["ssim"] / n, "lpips": acc["lpips"] / n,
            "rfid": float(fid.compute().item()), "kid": float(kid_mean.item())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--subset", nargs="+", default=["in10-val"],
                   choices=["in10-val", "in10-val-seen", "in10-val-unseen",
                            "imagenette-val", "imagenette-val-matched", "imagenet-val-full"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--kid-subset", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="JSON output (default: <checkpoint-dir>/rfid_imagenet.json)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, _ = load_config(args.checkpoint_dir)
    model = build_dualvae(cfg, device)
    sd = torch.load(find_weights(args.checkpoint_dir), map_location=device)
    model.load_state_dict(sd if "encoder.0.weight" in sd else sd.get("model_state_dict", sd))
    model.eval()

    # The training preprocessing, verbatim -- a different resize or normalization would put the
    # frozen encoder off-distribution and silently inflate every number here.
    transform = build_transform(cfg.get("resize_img", 256), cfg.get("dataset_name", "imagenette"))
    print(f"[cfg] resize={cfg.get('resize_img', 256)}  kl_beta={cfg.get('kl_beta')}  "
          f"component_prior={cfg.get('component_prior')}  device={device}")

    results = {}
    for name in args.subset:
        paths, desc = build_subset(name, args.seed)
        print(f"\n=== {name}  ({len(paths)} images) ===\n    {desc}", flush=True)
        m = evaluate(model, paths, transform, device, args.batch_size,
                     args.kid_subset, desc=name)
        m["description"] = desc
        results[name] = m
        print(f"    rFID {m['rfid']:8.3f} | KID {m['kid']:.5f} | LPIPS {m['lpips']:.4f} "
              f"| PSNR {m['psnr']:.2f} | SSIM {m['ssim']:.4f} | MSE {m['mse']:.5f}", flush=True)

    out = args.out or os.path.join(args.checkpoint_dir, "rfid_imagenet.json")
    with open(out, "w") as f:
        json.dump({"checkpoint_dir": args.checkpoint_dir, "results": results}, f, indent=2)
    print(f"\nWrote {out}")

    print(f"\n{'subset':<26}{'N':>7}{'rFID':>9}{'LPIPS':>9}{'PSNR':>8}{'SSIM':>8}")
    print("-" * 67)
    for k, m in results.items():
        print(f"{k:<26}{m['n_images']:>7}{m['rfid']:>9.3f}{m['lpips']:>9.4f}"
              f"{m['psnr']:>8.2f}{m['ssim']:>8.4f}")


if __name__ == "__main__":
    main()
