"""Matched re-evaluation of trained latent-flow runs: Gen/FID + held-out flow MSE + NFE sweep.

WHY THIS EXISTS
---------------
The FIDs logged during training are NOT comparable across the runs in this project: the
flow_vae run was configured with `gen_fid_n_samples: 1000` while both flow_dualvae runs used
2000. FID is a biased estimator whose bias grows as the sample count falls, and the Inception
feature covariance is 2048x2048 -- estimated from 1000 samples it is rank-deficient, so the
1000-sample number is inflated by an unknown, non-constant amount. Any cross-run comparison
has to re-score every run with the SAME number of reals and the SAME number of fakes.

This script does that, without touching any config file: every setting is read back from the
flow checkpoint's own stored `args`, and only the dataset path (the data lives on an external
disk) and the sampler step count (the sweep axis) are overridden at runtime.

WHAT IT REPORTS, per run
------------------------
1. Gen/FID + Gen/KID against the reals, at each requested ODE step count. The step sweep is
   the point: rectified-flow-style training is trained with a straight conditional path but
   an *independently coupled* source, so the learned marginal field is curved and the FID
   at low NFE measures how much curvature there is. A latent space whose FID degrades less
   as steps fall is the straighter one -- a sharper claim than FID at 100 NFE alone.
2. The held-out flow-matching MSE, computed with the exact protocol of
   train_latent_flow.validation_step (fixed seed, val_repeats Monte-Carlo passes,
   cfg_dropout_prob = 0 so the conditional model is what gets scored).
3. A t-binned breakdown of that MSE under UNIFORM t. This is a diagnostic, not a headline
   number: the CFM loss decomposes into model error plus the irreducible conditional variance
   Var[x1 - x0 | x_t, t], and the second term is a property of the latent distribution, not
   of model quality. A more multimodal latent has a higher floor. The per-bin curve shows
   *where* in t two runs differ, which tells you whether a loss gap is mid-trajectory
   (structure) or at the endpoints (fit).

CAVEAT that survives all of this: the flow MSE is computed in each model's OWN standardized
latent space, so it ranks epochs within a run. It is reported here for completeness and for
the t-binned diagnostic -- it is NOT a cross-run quality metric. Gen/FID is.

Real images are Inception-featurized ONCE and reused for every run and every step count
(torchmetrics' `reset_real_features=False`), which is both faster and guarantees the reals are
bit-identical across the whole comparison.

Usage
-----
    python -m tools.eval_flow_runs \
        --runs checkpoints/flow_vae/flow_vae_.../final_epoch.pt \
               checkpoints/flow_dualvae/flow_dualvae_.../final_epoch.pt \
        --dataset_path /media/tico/BACKUP-DIDI/imagenette/imagenette2-320 \
        --steps 4 8 16 50 \
        --out reports/flow_eval_matched.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This script is meant to be run detached with its output tee'd to a log; a progress bar that
# repaints every batch turns that log into tens of thousands of lines.
TQDM_KW = dict(mininterval=15.0, ncols=80)

from data.datasets import get_labeled_datasets
from experiments.train_latent_flow import generate_images
from losses.flow_matching import flow_matching_loss, sample_timesteps
from models.flow_unet import build_flow_model
from tools.latent_ae import LatentStats, load_frozen_ae
from tools.normalization import denormalize
from tools.utils import select_device, set_seed


# ---------------------------------------------------------------------------------------
# Checkpoint / data plumbing
# ---------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remap_ae_checkpoint(stored):
    """Resolve the AE path recorded in a flow checkpoint against THIS machine.

    These runs were trained on a cluster, so the checkpoint stores an absolute path like
    /gpfs/home3/<user>/Hybrid_Latent_Space_AE/checkpoints/vae/<run>/final_epoch.pt. The
    directory layout below `checkpoints/` is identical in the local clone, so the stored path
    is re-rooted at the local repo. An existing path is always honoured as-is.
    """
    if os.path.exists(stored):
        return stored
    parts = stored.replace("\\", "/").split("/")
    if "checkpoints" in parts:
        local = os.path.join(REPO_ROOT, *parts[parts.index("checkpoints"):])
        if os.path.exists(local) or os.path.isdir(os.path.dirname(local)):
            print(f"[AE] remapped {stored}\n  -> {local}")
            return local
    raise FileNotFoundError(
        f"AE checkpoint {stored!r} from the flow checkpoint does not exist here and could not "
        f"be re-rooted at {REPO_ROOT}."
    )


def load_flow_run(ckpt_path, dataset_path, device, use_ema=True, autoguide=None):
    """Rebuild a trained flow model from its checkpoint alone.

    The checkpoint stores the training `args` (scalars only), the latent stats, the resolved
    AE path, the class list and the latent shape -- everything needed to reproduce the
    training-time sampling protocol without reading any YAML.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ckpt["args"])
    cfg.dataset_path = dataset_path          # runtime override: data lives on an external disk

    ae_path = remap_ae_checkpoint(ckpt["ae_checkpoint"])
    ae = load_frozen_ae(ae_path, device, getattr(cfg, "ae_config", None))
    c, h, w = ckpt["latent_shape"]
    model = build_flow_model(cfg, in_channels=c, latent_size=h,
                             num_classes=ckpt["num_classes"]).to(device)

    # Sample from the EMA weights: that is what train_latent_flow.py logs its Gen/FID from
    # (`sample_model = ema.ema`), so anything else would not reproduce the reported numbers.
    if use_ema and ckpt.get("ema") is not None:
        model.load_state_dict(ckpt["ema"])
        weights = "ema"
    else:
        model.load_state_dict(ckpt["model"])
        weights = "raw"
    model.eval()

    # The raw (non-EMA) weights are what validation_step scored during training, so the flow
    # MSE is computed from those to stay comparable with the logged Val/Flow Loss.
    raw = build_flow_model(cfg, in_channels=c, latent_size=h,
                           num_classes=ckpt["num_classes"]).to(device)
    raw.load_state_dict(ckpt["model"])
    raw.eval()

    # AutoGuidance guiding branch: an EARLIER checkpoint of this same run (Karras et al. 2025
    # use a deliberately degraded version of the model -- same architecture, same data, less
    # training). `best.pt` is selected on the held-out flow loss, which plateaus around epoch
    # 400 of 2000 here, so it is a genuinely weaker model at zero extra training cost. Its
    # latent stats are identical (same run, same train latents), which AutoGuidance requires.
    bad = None
    if autoguide:
        bad_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), autoguide)
        if not os.path.exists(bad_path):
            raise FileNotFoundError(f"AutoGuidance checkpoint not found: {bad_path}")
        bad_ckpt = torch.load(bad_path, map_location="cpu", weights_only=False)
        bad = build_flow_model(cfg, in_channels=c, latent_size=h,
                               num_classes=ckpt["num_classes"]).to(device)
        bad.load_state_dict(bad_ckpt["ema"] if bad_ckpt.get("ema") is not None
                            else bad_ckpt["model"])
        bad.eval()
        print(f"[autoguidance] guiding epoch {ckpt.get('epoch')} with epoch "
              f"{bad_ckpt.get('epoch')} ({os.path.basename(bad_path)})")

    stats = LatentStats.from_state_dict(ckpt["latent_stats"]).to(device)
    return {"cfg": cfg, "ae": ae, "model": model, "raw_model": raw, "stats": stats,
            "bad_model": bad, "bad_epoch": None if bad is None else bad_ckpt.get("epoch"),
            "num_classes": ckpt["num_classes"], "class_names": ckpt.get("class_names"),
            "latent_shape": (c, h, w), "epoch": ckpt.get("epoch"), "weights": weights,
            "ae_checkpoint": ae.ae_checkpoint, "path": ckpt_path}


def build_valset(cfg):
    """The `val/` split of the dataset, with class ids forced onto the train split's ordering.

    NOTE `val_ratio` is irrelevant for imagenette: get_labeled_datasets uses the on-disk
    train/ and val/ directories directly, so this is the full 3925-image val split.
    """
    trainset, valset = get_labeled_datasets(
        cfg.dataset_name, path=cfg.dataset_path, resize_img=cfg.resize_img,
        val_ratio=getattr(cfg, "val_ratio", 0.2), seed=cfg.seed,
        labels_csv=getattr(cfg, "labels_csv", None),
        label_column=getattr(cfg, "label_column", "noisy_labels_0"))
    return trainset, valset


@torch.no_grad()
def encode_val_latents(ae, valset, cfg, device):
    """Encode the whole val split once -> (latents, labels), matching the trainer's cache.

    `latent_sample` is read from the config so this reproduces exactly what the run trained
    on (all three runs here use the posterior MEAN, i.e. noise = 0).
    """
    loader = DataLoader(valset, batch_size=getattr(cfg, "encode_batch_size", 32), shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 4))
    sample = getattr(cfg, "latent_sample", False)
    lat, labels = [], []
    for batch in tqdm(loader, desc="encode val", unit="batch", **TQDM_KW):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=getattr(cfg, "use_amp", True) and device == "cuda"):
            z = ae.encode(images, sample=sample)
        lat.append(z.float().cpu())
        labels.append(batch["label"].long())
    return torch.cat(lat), torch.cat(labels)


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def val_flow_mse(model, latents, labels, stats, cfg, device, num_repeats=4):
    """Held-out flow-matching MSE -- byte-for-byte the protocol of validation_step()."""
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    bs = getattr(cfg, "batch_size", 64)
    total, n = 0.0, 0
    for _ in range(num_repeats):
        for i in range(0, latents.shape[0], bs):
            z = stats.normalize(latents[i:i + bs].to(device))
            y = labels[i:i + bs].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=getattr(cfg, "use_amp", True) and device == "cuda"):
                loss, _ = flow_matching_loss(
                    model, z, y,
                    t_sampling=getattr(cfg, "t_sampling", "logit_normal"),
                    logit_normal_mean=getattr(cfg, "logit_normal_mean", 0.0),
                    logit_normal_std=getattr(cfg, "logit_normal_std", 1.0),
                    cfg_dropout_prob=0.0,
                    generator=gen,
                )
            total += loss.item()
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def flow_mse_by_t(model, latents, labels, stats, cfg, device, n_bins=10, repeats=2):
    """Per-t-bin flow MSE under UNIFORM t, to see where in the trajectory the loss sits.

    Uniform (not logit-normal) t so every bin gets comparable sample counts. The endpoints
    are near-trivial to predict and the middle is where path crossings -- and therefore the
    irreducible conditional variance -- concentrate.
    """
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    bs = getattr(cfg, "batch_size", 64)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for _ in range(repeats):
        for i in range(0, latents.shape[0], bs):
            x1 = stats.normalize(latents[i:i + bs].to(device))
            y = labels[i:i + bs].to(device)
            b = x1.shape[0]
            x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
            t = sample_timesteps(b, device, "uniform", generator=gen)
            tb = t.view(b, *([1] * (x1.dim() - 1)))
            x_t = (1.0 - tb) * x0 + tb * x1
            target = x1 - x0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=getattr(cfg, "use_amp", True) and device == "cuda"):
                pred = model(x_t, t, y)
            # per-sample MSE so each element lands in its own t bin
            per = (pred.float() - target.float()).pow(2).flatten(1).mean(1).cpu().numpy()
            idx = np.clip((t.cpu().numpy() * n_bins).astype(int), 0, n_bins - 1)
            for k in range(n_bins):
                m = idx == k
                if m.any():
                    sums[k] += per[m].sum()
                    counts[k] += m.sum()
    return (sums / np.maximum(counts, 1)).tolist(), counts.astype(int).tolist()


def build_metrics(cfg, fid_device, n_min):
    """FID/KID that KEEP their real-image statistics across reset(), so the reals are
    featurized once and shared by every run and every step count.

    KID's subset_size must be strictly smaller than the number of samples on both sides, so
    it is clamped against whichever of reals/fakes is smaller.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    subset = min(getattr(cfg, "kid_subset_size", 100), max(n_min - 1, 2))
    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False).to(fid_device)
    kid = KernelInceptionDistance(subset_size=subset, normalize=True,
                                  reset_real_features=False).to(fid_device)
    return fid, kid


@torch.no_grad()
def load_reals(fid, kid, valset, cfg, device, fid_device, max_real=0):
    """Featurize the FULL val split as the real distribution.

    `max_real` exists only for smoke-testing the pipeline; leave it at 0 for real results --
    capping the reals is precisely the bias this script was written to remove.
    """
    loader = DataLoader(valset, batch_size=getattr(cfg, "encode_batch_size", 32), shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 4))
    cap = max_real or float("inf")
    n = 0
    for batch in tqdm(loader, desc="reals", unit="batch", **TQDM_KW):
        imgs = denormalize(batch["image"].float().to(device), cfg.dataset_name, device).clamp(0, 1)
        fid.update(imgs.to(fid_device), real=True)
        kid.update(imgs.to(fid_device), real=True)
        n += imgs.shape[0]
        if n >= cap:
            break
    return n


@torch.no_grad()
def gen_fid_at_steps(run, fid, kid, cfg, device, fid_device, n_fake, steps, guidance=None):
    """Generate `n_fake` class-balanced samples at `steps` ODE steps and score them.

    Everything except the step count (and, if given, the guidance scale) is held at the run's
    own trained configuration -- solver, sample seed and batch size all come from the checkpoint.
    """
    cfg.sample_steps = steps                       # runtime override; no config file is touched
    if guidance is not None:
        cfg.guidance_scale = guidance
    num_classes = run["num_classes"]
    labels = torch.arange(num_classes).repeat(int(np.ceil(n_fake / num_classes)))[:n_fake]
    bs = getattr(cfg, "sample_batch_size", 16)
    t0 = time.time()
    for i in tqdm(range(0, n_fake, bs), desc=f"fakes @ {steps} steps", unit="batch", **TQDM_KW):
        chunk = labels[i:i + bs]
        imgs = generate_images(run["model"], run["ae"], run["stats"], cfg, device, chunk,
                               seed=getattr(cfg, "sample_seed", 1234) + i,
                               bad_model=run.get("bad_model"))
        fid.update(imgs.to(fid_device), real=False)
        kid.update(imgs.to(fid_device), real=False)
    kid_mean, kid_std = kid.compute()
    out = {"steps": steps,
           "guidance": float(getattr(cfg, "guidance_scale", 2.0)),
           "scheme": "autoguidance" if run.get("bad_model") is not None else "cfg",
           "nfe": steps * (2 if getattr(cfg, "sample_solver", "heun") == "heun" else 1),
           "gen_fid": fid.compute().item(),
           "gen_kid": kid_mean.item(), "gen_kid_std": kid_std.item(),
           "n_fake": int(n_fake), "seconds": round(time.time() - t0, 1)}
    fid.reset()        # reset_real_features=False -> drops the fakes, keeps the reals
    kid.reset()
    return out


# ---------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Flow checkpoints (final_epoch.pt).")
    p.add_argument("--dataset_path", required=True, type=str,
                   help="imagenette2-320 root (contains train/, val/, noisy_imagenette.csv).")
    p.add_argument("--steps", nargs="+", type=int, default=[4, 8, 16, 50],
                   help="ODE step counts to sweep. The largest should be the trained default.")
    p.add_argument("--autoguidance", default=None, type=str, metavar="CKPT_NAME",
                   help="Filename (inside each run's own directory, e.g. 'best.pt') of a "
                        "WEAKER checkpoint to use as the AutoGuidance branch instead of "
                        "classifier-free guidance. Karras et al. 2025: guiding with a "
                        "degraded version of the same model improves FID where CFG only "
                        "trades it against diversity. w=1.0 is unguided under both schemes.")
    p.add_argument("--guidance", nargs="+", type=float, default=None,
                   help="Classifier-free guidance scales to sweep (default: the run's trained "
                        "value only). FID is strongly non-monotonic in this knob -- it was never "
                        "tuned for these runs, so a sweep is the cheapest available improvement.")
    p.add_argument("--n_fake", default=0, type=int,
                   help="Generated samples per FID (0 = match the number of real val images).")
    p.add_argument("--val_repeats", default=None, type=int,
                   help="MC repeats for the flow MSE (default: the run's own val_repeats).")
    p.add_argument("--t_bins", default=10, type=int)
    p.add_argument("--max_real", default=0, type=int,
                   help="Cap on real images. SMOKE TESTING ONLY -- 0 (all 3925) for real results.")
    p.add_argument("--max_latents", default=0, type=int,
                   help="Cap on val latents for the flow MSE. Smoke testing only; 0 = all.")
    p.add_argument("--out", default="reports/flow_eval_matched.json", type=str)
    p.add_argument("--device", default="cuda", type=str)
    args = p.parse_args()

    device = select_device(args.device)
    results = []
    fid = kid = None
    n_real = None
    valset = None

    for run_path in args.runs:
        print(f"\n{'=' * 78}\n{run_path}\n{'=' * 78}")
        run = load_flow_run(run_path, args.dataset_path, device,
                            autoguide=args.autoguidance)
        cfg = run["cfg"]
        # Determinism for the encode pass and any dataloader shuffling.
        set_seed(cfg.seed, getattr(cfg, "deterministic", True),
                 getattr(cfg, "cudnn_benchmark", False))
        fid_device = getattr(cfg, "gen_fid_device", device)

        trainset, valset_run = build_valset(cfg)
        print(f"[data] val={len(valset_run)} images, {len(trainset.classes)} classes "
              f"(labels: {getattr(valset_run, 'labels_csv', None) or 'dir/filename'})")

        # Reals are identical across runs (same split, same resize, same normalization), so
        # they are featurized once and reused -- which also makes the comparison exact.
        if fid is None:
            valset = valset_run
            expected_real = min(len(valset), args.max_real or len(valset))
            fid, kid = build_metrics(cfg, fid_device,
                                     min(expected_real, args.n_fake or expected_real))
            n_real = load_reals(fid, kid, valset, cfg, device, fid_device, args.max_real)
            print(f"[fid] {n_real} real images featurized (shared by all runs); "
                  f"KID subset_size={kid.subset_size}")
        n_fake = args.n_fake or n_real

        # ---- held-out flow MSE (raw weights, training protocol) + t-binned diagnostic ----
        latents, labels = encode_val_latents(run["ae"], valset_run, cfg, device)
        if args.max_latents:
            latents, labels = latents[:args.max_latents], labels[:args.max_latents]
        print(f"[latents] val {tuple(latents.shape)}")
        repeats = args.val_repeats or getattr(cfg, "val_repeats", 4)
        mse_raw = val_flow_mse(run["raw_model"], latents, labels, run["stats"], cfg, device, repeats)
        mse_ema = val_flow_mse(run["model"], latents, labels, run["stats"], cfg, device, repeats)
        t_curve, t_counts = flow_mse_by_t(run["raw_model"], latents, labels, run["stats"],
                                          cfg, device, n_bins=args.t_bins)
        print(f"[flow mse] raw={mse_raw:.5f}  ema={mse_ema:.5f}  (val_repeats={repeats})")

        # ---- Gen/FID sweep over ODE steps ----
        sweep = []
        guidances = args.guidance if args.guidance else [None]
        for gscale in guidances:
            for s in sorted(args.steps):
                m = gen_fid_at_steps(run, fid, kid, cfg, device, fid_device, n_fake, s, gscale)
                print(f"[fid] steps={s:>3} (NFE {m['nfe']:>3}) cfg={m['guidance']:.2f}  "
                      f"FID={m['gen_fid']:8.3f}  KID={m['gen_kid']:.5f}  [{m['seconds']}s]")
                sweep.append(m)

        results.append({
            "run": os.path.basename(os.path.dirname(run_path)),
            "checkpoint": run_path,
            "ae_checkpoint": run["ae_checkpoint"],
            "epoch": run["epoch"],
            "sampling": {"solver": getattr(cfg, "sample_solver", "heun"),
                         "guidance_scale": getattr(cfg, "guidance_scale", 2.0),
                         "sample_seed": getattr(cfg, "sample_seed", 1234),
                         "weights": run["weights"],
                         "scheme": "autoguidance" if run.get("bad_model") is not None else "cfg",
                         "autoguidance_from_epoch": run.get("bad_epoch")},
            "n_real": n_real, "n_fake": n_fake,
            "val_flow_mse_raw": mse_raw, "val_flow_mse_ema": mse_ema,
            "val_repeats": repeats,
            "flow_mse_by_t": {"bins": args.t_bins, "mse": t_curve, "counts": t_counts},
            "fid_sweep": sweep,
        })

        del run
        torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"n_real": n_real, "dataset_path": args.dataset_path,
                       "runs": results}, f, indent=2)

    # ---- summary ----
    print(f"\n{'=' * 78}\nMATCHED EVALUATION ({n_real} reals, {results[0]['n_fake']} fakes)\n{'=' * 78}")
    guidances = sorted({m["guidance"] for r in results for m in r["fid_sweep"]})
    for g in guidances:
        if len(guidances) > 1:
            print(f"\n-- guidance scale {g:.2f} --")
        hdr = (f"{'run':<38}{'val MSE':>9}"
               + "".join(f"{'FID@' + str(s):>10}" for s in sorted(args.steps)))
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            by_steps = {m["steps"]: m["gen_fid"] for m in r["fid_sweep"] if m["guidance"] == g}
            row = f"{r['run']:<38}{r['val_flow_mse_raw']:>9.4f}"
            row += "".join(f"{by_steps[s]:>10.2f}" if s in by_steps else f"{'--':>10}"
                           for s in sorted(args.steps))
            print(row)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
