"""MM-FM mode tests on the frozen FSQ_0.1 latent (reports/mmfm_mode_tests_plan.md).

Nothing here trains a generator and nothing here touches a model, loss or trainer file: the
autoencoder is loaded frozen through tools.latent_ae.load_frozen_ae, the quantizer's fitted
mixture is read out of its checkpoint buffers, and the only modules instantiated fresh are
the ones the plan asks to be *used as mode assigners* (VQEmbedding / FSQEmbedding on DINOv2
[CLS]), never re-trained into the autoencoder.

One subcommand per step of the plan:

  step0   FIT/EVAL subsets, DINOv2-B/14 [CLS], per-channel latent normalization ("flow space")
  step1   unit checks on the autoencoder (attention relocation, mixture-statistic consistency)
  step2   per-patch FSQ mixture statistics (buffers + a pre-quantization histogram)
  step3   image-level mode assignments: k-means / EMA-VQ / FSQ-on-PCA / baselines
  step4   metrics per (method, M, variant) in the normalized latent space
  step5   coherence of per-patch sampling (the codemap PNG)
  report  reports/mmfm_mode_tests/summary.md from results.json

Every step appends to reports/mmfm_mode_tests/results.json, so a failure in one step never
costs the DINOv2 extraction done in step0.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import HDF5ImageDataset, build_image_transform          # noqa: E402
from models.modules.embedding import VQEmbedding                            # noqa: E402
from models.modules.fsq import FSQEmbedding                                 # noqa: E402
from tools.distributed import all_reduce_sum_, is_dist                      # noqa: E402
from tools.latent_ae import load_frozen_ae                                  # noqa: E402

# --------------------------------------------------------------------------- #
# Inputs, exactly as configs/flow_in_fsq01.yaml uses them.
# --------------------------------------------------------------------------- #
AE_RUN_DIR = "./checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305"
CACHE_STEM = ("/work/mgonzalez/latent_cache/"
              "dualvae_dualvae_20260811-093454_af2305_final_epoch_imagenet_{split}_256")
H5_PATH = "/data/image-models-project/datasets/imagenet_full_256.h5"
# Optional reference arm (configs/flow_in_vae.yaml). Skipped when its cache is absent.
VAE_RUN_DIR = "./checkpoints/vae/vae_20260810-165000_c9374d"
# The stem `experiments/train_latent_flow.py::_cache_stem` would produce for this run, so a
# cache written by `encode_vae` below is byte-for-byte the one that trainer would write and
# reuse -- point `latent_cache_dir:` at the directory and it maps it instead of re-encoding.
VAE_CACHE_NAME = "vae_vae_20260810-165000_c9374d_final_epoch_imagenet_{split}_256"
# /data (5 T, lustre) first: /work has a 100 G quota already carrying the 21 G dualvae cache,
# and this one is another 20.3 G. The /work path stays in the search order so a cache written
# there by an earlier flow_in_vae run is still found.
VAE_CACHE_DIRS = ["/data/image-models-project/manfred/latent_cache",
                  "/work/mgonzalez/latent_cache"]

# FSQ level grids per image-level mode count M (plan, step 3C). prod(levels) == M.
LEVELS_BY_M = {
    64: [4, 4, 4],            # smoke only
    1024: [4, 4, 4, 4, 4],
    4096: [8, 8, 8, 8],
    8192: [8, 8, 8, 4, 4],
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# results.json bookkeeping
# --------------------------------------------------------------------------- #
def results_path(out_dir):
    return os.path.join(out_dir, "results.json")


def load_results(out_dir):
    p = results_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_results(out_dir, res):
    os.makedirs(out_dir, exist_ok=True)
    with open(results_path(out_dir), "w") as f:
        json.dump(res, f, indent=2, sort_keys=False, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def record(out_dir, step, payload, elapsed=None):
    res = load_results(out_dir)
    res[step] = payload
    if elapsed is not None:
        res.setdefault("runtime_sec", {})[step] = round(float(elapsed), 2)
    save_results(out_dir, res)


# --------------------------------------------------------------------------- #
# Input verification (the plan's Inputs table; nothing runs until this passes)
# --------------------------------------------------------------------------- #
def verify_inputs(args, need_cache=True, need_h5=True, need_ae=True):
    missing, found = [], {}

    def chk(label, path, required):
        ok = os.path.exists(path)
        found[label] = {"path": path, "exists": bool(ok)}
        if required and not ok:
            missing.append(f"{label}: {path}")
        return ok

    chk("ae_run_dir", AE_RUN_DIR, need_ae)
    chk("ae_final_epoch", os.path.join(AE_RUN_DIR, "final_epoch.pt"), need_ae)
    chk("ae_config_used", os.path.join(AE_RUN_DIR, "config_used.yaml"), need_ae)
    for split in ("train", "val"):
        stem = CACHE_STEM.format(split=split)
        chk(f"latent_cache_{split}", stem + ".latents.npy", need_cache)
        chk(f"latent_labels_{split}", stem + ".labels.npy", need_cache)
    chk("dataset_h5", H5_PATH, need_h5)
    # Optional arm: recorded, never required.
    chk("vae_run_dir", VAE_RUN_DIR, False)
    for stem in vae_stem_candidates(args):
        for split in ("train", "val"):
            chk(f"vae_latent_cache_{split} [{os.path.dirname(stem)}]",
                stem.format(split=split) + ".latents.npy", False)

    if missing:
        print("[inputs] MISSING required input(s):", file=sys.stderr)
        for m in missing:
            print("  - " + m, file=sys.stderr)
        raise SystemExit(2)
    print(f"[inputs] all required inputs present ({sum(v['exists'] for v in found.values())}"
          f"/{len(found)} listed paths exist).")
    return found


def vae_stem_candidates(args=None):
    dirs = []
    if getattr(args, "vae_cache_dir", None):
        dirs.append(args.vae_cache_dir)
    dirs.extend(VAE_CACHE_DIRS)
    seen = list(dict.fromkeys(dirs))
    return [os.path.join(d, VAE_CACHE_NAME) for d in seen]


def _stem_complete(stem):
    return all(os.path.exists(stem.format(split=s) + ".latents.npy")
               and os.path.exists(stem.format(split=s) + ".labels.npy")
               for s in ("train", "val"))


def resolve_vae_stem(args=None):
    """The first complete plain-VAE cache among the candidate directories, else None."""
    for stem in vae_stem_candidates(args):
        if _stem_complete(stem):
            return stem
    return None


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def open_cache(split, stem=CACHE_STEM):
    s = stem.format(split=split)
    lat = np.load(s + ".latents.npy", mmap_mode="r")
    lab = np.load(s + ".labels.npy")
    return lat, lab


def read_rows(memmap, indices, chunk=2048):
    """(n, prod(shape[1:])) fp16 array of the requested rows. `indices` must be sorted."""
    n = len(indices)
    d = int(np.prod(memmap.shape[1:]))
    out = np.empty((n, d), dtype=np.float16)
    for i in range(0, n, chunk):
        sl = np.asarray(indices[i:i + chunk])
        out[i:i + len(sl)] = np.asarray(memmap[sl]).reshape(len(sl), d)
    return out


def iter_chunks(arr, chunk):
    for i in range(0, arr.shape[0], chunk):
        yield i, arr[i:i + chunk]


def perplexity_of(counts):
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def nmi(a, b):
    """Normalized mutual information, arithmetic normalization (sklearn's default)."""
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    n = a.shape[0]
    _, ia, ca = np.unique(a, return_inverse=True, return_counts=True)
    _, ib, cb = np.unique(b, return_inverse=True, return_counts=True)
    if len(ca) == 1 or len(cb) == 1:
        return 0.0
    nb = len(cb)
    pair = ia.astype(np.int64) * nb + ib.astype(np.int64)
    vals, cnt = np.unique(pair, return_counts=True)
    i = vals // nb
    j = vals % nb
    pij = cnt / n
    pi_ = ca[i] / n
    pj_ = cb[j] / n
    mi = float((pij * np.log(pij / (pi_ * pj_))).sum())
    ha = float(-((ca / n) * np.log(ca / n)).sum())
    hb = float(-((cb / n) * np.log(cb / n)).sum())
    denom = 0.5 * (ha + hb)
    return mi / denom if denom > 0 else 0.0


def class_cond_perplexity(modes, labels):
    """Mean over classes of the perplexity of p(m | y): how many modes a class spreads over."""
    modes = np.asarray(modes)
    labels = np.asarray(labels)
    vals = []
    for y in np.unique(labels):
        m = modes[labels == y]
        if m.size == 0:
            continue
        vals.append(perplexity_of(np.bincount(m)))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Step 0: data subsets, DINOv2 [CLS], flow-space normalization
# --------------------------------------------------------------------------- #
def build_dinov2(device, torch_home):
    if torch_home:
        os.environ["TORCH_HOME"] = torch_home
        torch.hub.set_dir(os.path.join(torch_home, "hub"))
    # The weights must already sit in <TORCH_HOME>/hub/checkpoints: compute nodes have no
    # internet, and torch.hub only skips its GitHub round-trip when the repo dir is cached.
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_cls(model, split, indices, args, device, desc):
    """DINOv2-B/14 [CLS] (768-d) for the given h5 rows, in the plan's preprocessing.

    x in [-1, 1] -> (x+1)/2 -> bicubic-antialias resize to 224 -> ImageNet mean/std.
    """
    from torch.utils.data import DataLoader, Subset

    ds = HDF5ImageDataset(H5_PATH, split,
                          transform=build_image_transform("imagenet", 256), labeled=True)
    sub = Subset(ds, list(map(int, indices)))
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    out = np.empty((len(indices), 768), dtype=np.float16)
    labels = np.empty(len(indices), dtype=np.int64)
    seen_idx = np.empty(len(indices), dtype=np.int64)
    i = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        x = (x - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cls = model.forward_features(x)["x_norm_clstoken"]
        b = cls.shape[0]
        out[i:i + b] = cls.float().cpu().numpy().astype(np.float16)
        labels[i:i + b] = batch["label"].numpy()
        seen_idx[i:i + b] = batch["idx"].numpy()
        i += b
        if bi % 200 == 0:
            done = max(i, 1)
            rate = done / (time.time() - t0)
            print(f"  [{desc}] {i}/{len(indices)}  {rate:.0f} img/s", flush=True)
    assert i == len(indices), f"{desc}: got {i} rows for {len(indices)} indices"
    # The cache/label alignment the whole study rests on: batch['idx'] IS the h5 row.
    assert np.array_equal(seen_idx, np.asarray(indices)), f"{desc}: dataloader reordered the subset"
    return out, labels


def cmd_step0(args):
    t0 = time.time()
    verify_inputs(args)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    lat_tr, lab_tr = open_cache("train")
    lat_va, lab_va = open_cache("val")
    n_train, n_val = lat_tr.shape[0], lat_va.shape[0]

    # FIT: uniform without replacement, seed 0, then SORTED for h5 read locality.
    rng = np.random.default_rng(0)
    fit_n = min(args.fit_n, n_train)
    idx_fit = np.sort(rng.choice(n_train, size=fit_n, replace=False)).astype(np.int64)

    # EVAL: the whole held-out val split. A smaller --eval-n (smoke) is drawn uniformly with
    # seed 1 rather than taken as a prefix, because the val split is ordered by class.
    if args.eval_n and args.eval_n < n_val:
        idx_eval = np.sort(np.random.default_rng(1).choice(n_val, size=args.eval_n,
                                                           replace=False)).astype(np.int64)
    else:
        idx_eval = np.arange(n_val, dtype=np.int64)

    np.save(os.path.join(args.out, "indices_fit.npy"), idx_fit)
    np.save(os.path.join(args.out, "indices_eval.npy"), idx_eval)

    # --- DINOv2 [CLS] -----------------------------------------------------
    model = build_dinov2(device, args.torch_home)
    cls_fit, lab_fit_h5 = extract_cls(model, "train", idx_fit, args, device, "cls_fit")
    cls_eval, lab_eval_h5 = extract_cls(model, "val", idx_eval, args, device, "cls_eval")
    del model
    torch.cuda.empty_cache()

    # Cache row i must be h5 image i of the same split; the labels are the cheap proof.
    lab_fit_cache = lab_tr[idx_fit]
    lab_eval_cache = lab_va[idx_eval]
    align_fit = bool(np.array_equal(lab_fit_cache, lab_fit_h5))
    align_eval = bool(np.array_equal(lab_eval_cache, lab_eval_h5))
    if not (align_fit and align_eval):
        raise SystemExit("latent-cache labels disagree with the h5 labels at the same rows; "
                         "the cache is not row-aligned with the dataset.")

    np.save(os.path.join(args.out, "cls_fit.npy"), cls_fit)
    np.save(os.path.join(args.out, "cls_eval.npy"), cls_eval)
    np.save(os.path.join(args.out, "labels_fit.npy"), lab_fit_cache)
    np.save(os.path.join(args.out, "labels_eval.npy"), lab_eval_cache)

    # --- flow space: per-channel mean/std on the FIT latents ----------------
    c = lat_tr.shape[1]
    s = np.zeros(c, dtype=np.float64)
    s2 = np.zeros(c, dtype=np.float64)
    n_elem = 0
    for i in range(0, fit_n, 2048):
        blk = np.asarray(lat_tr[idx_fit[i:i + 2048]]).astype(np.float64)
        s += blk.sum(axis=(0, 2, 3))
        s2 += (blk ** 2).sum(axis=(0, 2, 3))
        n_elem += blk.shape[0] * blk.shape[2] * blk.shape[3]
    chan_mean = s / n_elem
    chan_std = np.sqrt(np.maximum(s2 / n_elem - chan_mean ** 2, 0.0))
    np.savez(os.path.join(args.out, "latent_stats.npz"),
             mean=chan_mean.astype(np.float32), std=chan_std.astype(np.float32))

    payload = {
        "n_train": int(n_train), "n_val": int(n_val),
        "fit_n": int(fit_n), "eval_n": int(len(idx_eval)),
        "fit_seed": 0, "eval_is_full_val": bool(len(idx_eval) == n_val),
        "cache_alignment_verified": bool(align_fit and align_eval),
        "cls_dim": 768, "cls_model": "dinov2_vitb14",
        "latent_shape": list(lat_tr.shape[1:]),
        "flow_dim": int(np.prod(lat_tr.shape[1:])),
        "latent_chan_mean": chan_mean.tolist(),
        "latent_chan_std": chan_std.tolist(),
        "n_classes_fit": int(len(np.unique(lab_fit_cache))),
        "n_classes_eval": int(len(np.unique(lab_eval_cache))),
        "vae_cache_available": bool(vae_cache_available()),
    }
    record(args.out, "step0", payload, time.time() - t0)
    print(json.dumps(payload, indent=2)[:1200])


# --------------------------------------------------------------------------- #
# Step 1: unit checks on the autoencoder
# --------------------------------------------------------------------------- #
def unit_indices(args):
    """A seeded 20k subsample OF the FIT set (not its prefix: the h5 is class-ordered)."""
    idx_fit = np.load(os.path.join(args.out, "indices_fit.npy"))
    n = min(args.unit_n, len(idx_fit))
    sel = np.random.default_rng(2).choice(len(idx_fit), size=n, replace=False)
    return np.sort(idx_fit[sel]).astype(np.int64)


def image_loader(split, indices, args):
    from torch.utils.data import DataLoader, Subset
    ds = HDF5ImageDataset(H5_PATH, split,
                          transform=build_image_transform("imagenet", 256), labeled=True)
    return DataLoader(Subset(ds, list(map(int, indices))), batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers, pin_memory=True)


@torch.no_grad()
def encode_parts(model, x):
    """encode_for_diffusion(x, noise=0), reproduced step by step (noise = 0 => Delta = mean).

    Returns the pieces the plan's checks need; `z_pre` is z_q + m, i.e. the mixture-space
    latent BEFORE the attention block relocates it into flow space.
    """
    z_e = model.encoder(x)
    z_e_vq = model.bottle_neck_VQ(z_e)
    z_q, _, idx, _, _ = model.vq_layer(z_e_vq)
    pq = model.vq_layer.pre_quant(z_e_vq)
    r = pq - z_q.detach()
    z_e_vanilla = model.vanilla_VAE_bottle_neck(r)
    mean, logvar = torch.chunk(z_e_vanilla, 2, dim=1)
    logvar = torch.clamp(logvar, -30, 20)
    z_pre = z_q + mean
    return {"z_e_vq": z_e_vq, "pre_quant": pq, "z_q": z_q, "idx": idx, "r": r,
            "mean": mean, "var": logvar.exp(), "z_pre": z_pre}


def cmd_step1(args):
    t0 = time.time()
    verify_inputs(args)
    device = torch.device(args.device)
    ae = load_frozen_ae(AE_RUN_DIR, device)
    model = ae.model
    assert not model.training, "the frozen AE must be in eval mode (EMA buffers must not move)"
    vq = model.vq_layer
    assert isinstance(vq, FSQEmbedding), f"expected an FSQEmbedding, got {type(vq).__name__}"
    assert model.residual_continuous, "this check assumes the residual_continuous wiring"

    idx = unit_indices(args)
    loader = image_loader("train", idx, args)

    K, d = vq.num_embeddings, vq.embedding_dim
    cnt = torch.zeros(K, dtype=torch.float64, device=device)
    s_r = torch.zeros(K, d, dtype=torch.float64, device=device)
    s_r2 = torch.zeros(K, d, dtype=torch.float64, device=device)
    s_m = torch.zeros(K, d, dtype=torch.float64, device=device)
    s_m2 = torch.zeros(K, d, dtype=torch.float64, device=device)
    s_v = torch.zeros(K, d, dtype=torch.float64, device=device)

    max_abs_diff = 0.0
    rel_reloc_num, rel_reloc_den = 0.0, 0.0
    n_img = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True).float()
            p = encode_parts(model, x)

            # 1a: attention(z_pre) must equal encode_for_diffusion(x, noise=0).
            z_ref = model.encode_for_diffusion(x, noise=torch.zeros_like(p["z_pre"]))
            z_att = model.attention(p["z_pre"])
            max_abs_diff = max(max_abs_diff, float((z_att - z_ref).abs().max()))
            # How far the flow space sits from the mixture space.
            rel_reloc_num += float((z_att - p["z_pre"]).pow(2).sum())
            rel_reloc_den += float(p["z_pre"].pow(2).sum())

            # 1b: per-location statistics, keyed on the code.
            i = p["idx"].reshape(-1)
            flat = lambda t: t.permute(0, 2, 3, 1).reshape(-1, d).double()
            cnt.index_add_(0, i, torch.ones_like(i, dtype=torch.float64))
            s_r.index_add_(0, i, flat(p["r"]))
            s_r2.index_add_(0, i, flat(p["r"]) ** 2)
            s_m.index_add_(0, i, flat(p["mean"]))
            s_m2.index_add_(0, i, flat(p["mean"]) ** 2)
            s_v.index_add_(0, i, flat(p["var"]))
            n_img += x.shape[0]
            if bi % 50 == 0:
                print(f"  [step1] {n_img}/{len(idx)} images", flush=True)

    hit = cnt >= args.min_hits
    n_hit = int(hit.sum())
    if n_hit == 0:
        raise SystemExit(f"no code reached {args.min_hits} hits in {n_img} images; "
                         f"raise --unit-n or lower --min-hits.")
    c = cnt[hit].unsqueeze(1)
    Er = s_r[hit] / c
    Em = s_m[hit] / c
    Vr = (s_r2[hit] / c - Er ** 2).clamp_min(0)
    Vm = (s_m2[hit] / c - Em ** 2).clamp_min(0)
    Es2 = s_v[hit] / c

    mu_ema = vq.mu[hit].double()
    s2_ema = vq.sigma2_raw[hit].double()

    def med_rel(a, b):
        """median over codes of ||a_k - b_k|| / ||b_k|| (per-code vectors over the d channels)."""
        num = (a - b).norm(dim=1)
        den = b.norm(dim=1).clamp_min(1e-12)
        return float((num / den).median())

    # residual head's deviation from identity
    W = model.vanilla_VAE_bottle_neck.weight[:, :, 0, 0]     # (2C, C)
    C = model.latent_channels
    Wm = W[:C].double()
    I = torch.eye(C, dtype=torch.float64, device=device)
    b = model.vanilla_VAE_bottle_neck.bias.double()

    payload = {
        "n_images": n_img,
        "min_hits": args.min_hits,
        "codes_with_min_hits": n_hit,
        "codes_touched": int((cnt > 0).sum()),
        "locations": int(cnt.sum()),
        "attention_relocation": {
            "max_abs_diff_vs_encode_for_diffusion": max_abs_diff,
            "rel_move_attention_over_pre": float(math.sqrt(rel_reloc_num / max(rel_reloc_den, 1e-30))),
        },
        "mixture_consistency_median_rel_err": {
            "ema_mu_vs_E_r_given_k": med_rel(mu_ema, Er),
            "E_m_given_k_vs_E_r_given_k": med_rel(Em, Er),
            "sigma2_raw_vs_Var_r_given_k": med_rel(s2_ema, Vr),
            "sigma2_raw_vs_Var_m_plus_E_s2": med_rel(s2_ema, Vm + Es2),
        },
        "magnitudes": {
            "mean_abs_E_r_given_k": float(Er.abs().mean()),
            "mean_abs_ema_mu": float(mu_ema.abs().mean()),
            "mean_Var_r_given_k": float(Vr.mean()),
            "mean_sigma2_raw": float(s2_ema.mean()),
            "mean_Var_m_given_k": float(Vm.mean()),
            "mean_E_s2_given_k": float(Es2.mean()),
        },
        "residual_head": {
            "rel_dev_W_mean_from_identity": float((Wm - I).norm() / I.norm()),
            "mean_bias_norm": float(b[:C].norm()),
            "logvar_bias_mean": float(b[C:].mean()),
        },
    }
    record(args.out, "step1", payload, time.time() - t0)
    print(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Step 2: per-patch FSQ mixture statistics
# --------------------------------------------------------------------------- #
def cmd_step2(args):
    t0 = time.time()
    verify_inputs(args)
    device = torch.device(args.device)
    ae = load_frozen_ae(AE_RUN_DIR, device)
    model = ae.model
    vq = model.vq_layer
    assert isinstance(vq, FSQEmbedding)

    # ---- from the checkpoint buffers only --------------------------------
    seen = vq.codes_seen.bool()
    n_seen = int(seen.sum())
    pi = vq.pi.double()
    s2 = vq.sigma2_raw.double()
    mu = vq.mu.double()
    s2_seen = s2[seen]
    mu_seen = mu[seen]
    pi_seen = pi[seen]
    pi_ren = pi_seen / pi_seen.sum().clamp_min(1e-12)

    def pct(t, qs=(10, 50, 90)):
        q = torch.tensor([x / 100.0 for x in qs], dtype=torch.float64, device=t.device)
        return torch.quantile(t, q).tolist()

    sigma_seen = s2_seen.clamp_min(0).sqrt()
    # Grid spacing in the FSQ-normalized space: 1 / half_width per channel (0.5 for L=5).
    spacing = (1.0 / vq._half_width.double())
    sep = spacing.unsqueeze(0) / sigma_seen.clamp_min(1e-12)

    # A code that was hit once and then never again has an EMA mass decayed to ~0 by
    # ema_decay^steps, which makes its sigma2_raw a float32 denormal and its separation ratio
    # meaningless (0.5 / ~0). `codes_seen` is cumulative and does NOT filter those out, so
    # every statistic below is reported twice: over all seen codes (the honest state of the
    # checkpoint) and over the LIVE ones, whose EMA mass is still worth at least one hit.
    n_ema = vq.ema_cluster_size.double()
    live = seen & (vq.ema_cluster_size >= args.live_count_min)
    s2_live = s2[live]
    mu_live = mu[live]
    sigma_live = s2_live.clamp_min(0).sqrt()
    sep_live = (1.0 / vq._half_width.double()).unsqueeze(0) / sigma_live.clamp_min(1e-12)

    buffers = {
        "num_codes": int(vq.num_embeddings),
        "levels": list(vq.levels),
        "codes_seen_frac": float(vq.codes_seen.mean()),
        "codes_seen": n_seen,
        "live_count_min": float(args.live_count_min),
        "codes_live": int(live.sum()),
        "codes_live_frac_of_seen": float(live.sum() / max(n_seen, 1)),
        "ema_cluster_size_percentiles_over_seen": pct(n_ema[seen]),
        "perplexity_pi": float(torch.exp(-(pi * (pi + 1e-30).log()).sum())),
        "perplexity_pi_seen_only": float(torch.exp(-(pi_ren * (pi_ren + 1e-30).log()).sum())),
        "pi_note": ("ema_cluster_size is initialised to 1 for every code, so perplexity_pi "
                    "includes the un-decayed init mass; perplexity_pi_seen_only renormalizes "
                    "over the codes the data actually reached."),
        "sigma2_raw_percentiles_per_channel": {
            f"ch{c}": pct(s2_seen[:, c]) for c in range(s2_seen.shape[1])
        },
        "sigma2_raw_percentiles_per_channel_live": {
            f"ch{c}": pct(s2_live[:, c]) for c in range(s2_live.shape[1])
        } if int(live.sum()) else {},
        "sigma2_floor": float(vq.sigma2_floor),
        "sigma2_ceil": float(vq.sigma2_ceil),
        "frac_at_or_below_floor": float((s2_seen <= vq.sigma2_floor).double().mean()),
        "frac_at_or_above_ceil": float((s2_seen >= vq.sigma2_ceil).double().mean()),
        "frac_at_or_below_floor_live": float((s2_live <= vq.sigma2_floor).double().mean())
                                       if int(live.sum()) else None,
        "floor_ceil_frac_note": "fractions are over (code, channel) entries, not codes",
        "mean_abs_mu_over_mean_sigma": float(mu_seen.abs().mean() / sigma_seen.mean().clamp_min(1e-12)),
        "mean_abs_mu_over_mean_sigma_live": (
            float(mu_live.abs().mean() / sigma_live.mean().clamp_min(1e-12))
            if int(live.sum()) else None),
        "grid_spacing_per_channel": spacing.tolist(),
        "separation_ratio_percentiles": pct(sep.reshape(-1)),
        "separation_ratio_percentiles_live": pct(sep_live.reshape(-1)) if int(live.sum()) else None,
        "separation_ratio_percentiles_per_channel": {
            f"ch{c}": pct(sep[:, c]) for c in range(sep.shape[1])
        },
    }

    # ---- from data: the pre-quantization histogram ------------------------
    idx = unit_indices(args)
    loader = image_loader("train", idx, args)
    nbins = 200
    hist = torch.zeros(vq.embedding_dim, nbins, dtype=torch.float64, device=device)
    n_img = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True).float()
            z_e_vq = model.bottle_neck_VQ(model.encoder(x))
            pq = model.vq_layer.pre_quant(z_e_vq)          # (B, d, H, W), ~[-1, 1]
            for c in range(vq.embedding_dim):
                hist[c] += torch.histc(pq[:, c].float().reshape(-1), bins=nbins,
                                       min=-1.0, max=1.0).double()
            n_img += x.shape[0]
            if bi % 50 == 0:
                print(f"  [step2] {n_img}/{len(idx)} images", flush=True)

    edges = np.linspace(-1.0, 1.0, nbins + 1)
    centers_bin = 0.5 * (edges[:-1] + edges[1:])
    cell_centres = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])     # levels=5 -> ints/2
    cell_bounds = np.array([-0.75, -0.25, 0.25, 0.75])
    win = 0.05
    m_c = np.any(np.abs(centers_bin[None, :] - cell_centres[:, None]) <= win, axis=0)
    m_b = np.any(np.abs(centers_bin[None, :] - cell_bounds[:, None]) <= win, axis=0)

    h = hist.cpu().numpy()
    dens = h / h.sum(axis=1, keepdims=True).clip(min=1e-30) / (edges[1] - edges[0])
    ratio = {}
    for c in range(dens.shape[0]):
        rc = float(dens[c][m_c].mean() / max(dens[c][m_b].mean(), 1e-30))
        ratio[f"ch{c}"] = rc
    ratio_all = float(np.mean(list(ratio.values())))

    png = os.path.join(args.out, "fsq_channel_hist.png")
    _plot_channel_hist(centers_bin, dens, cell_centres, cell_bounds, ratio, png)

    payload = {
        "from_buffers": buffers,
        "from_data": {
            "n_images": n_img,
            "n_bins": nbins,
            "window": win,
            "cell_centres": cell_centres.tolist(),
            "cell_boundaries": cell_bounds.tolist(),
            "centre_over_boundary_density_ratio_per_channel": ratio,
            "centre_over_boundary_density_ratio_mean": ratio_all,
        },
        "png": png,
    }
    record(args.out, "step2", payload, time.time() - t0)
    print(json.dumps(payload, indent=2)[:3000])


def _plot_channel_hist(centers, dens, cell_centres, cell_bounds, ratio, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nch = dens.shape[0]
    ncol = 4
    nrow = int(math.ceil(nch / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    for c in range(nch):
        ax = axes[c // ncol][c % ncol]
        ax.plot(centers, dens[c], lw=1.0, color="#1f77b4")
        for v in cell_centres:
            ax.axvline(v, color="#d62728", lw=0.9)
        for v in cell_bounds:
            ax.axvline(v, color="#7f7f7f", lw=0.8, ls="--")
        ax.set_title(f"ch{c}  centre/boundary = {ratio[f'ch{c}']:.2f}", fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.tick_params(labelsize=7)
    for c in range(nch, nrow * ncol):
        axes[c // ncol][c % ncol].axis("off")
    fig.suptitle("pre_quant(z_e_vq) density per channel  (red = FSQ cell centres, "
                 "dashed = cell boundaries)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[step2] wrote {path}")


# --------------------------------------------------------------------------- #
# Step 3: image-level mode assignments
# --------------------------------------------------------------------------- #
def assign_dir(out):
    d = os.path.join(out, "assign")
    os.makedirs(d, exist_ok=True)
    return d


def save_assign(out, key, fit, ev):
    d = assign_dir(out)
    np.save(os.path.join(d, key + "_fit.npy"), np.asarray(fit, dtype=np.int32))
    np.save(os.path.join(d, key + "_eval.npy"), np.asarray(ev, dtype=np.int32))


def load_assign(out, key):
    d = assign_dir(out)
    return (np.load(os.path.join(d, key + "_fit.npy")),
            np.load(os.path.join(d, key + "_eval.npy")))


def _cos_assign(X, C, chunk=8192):
    """argmax cosine similarity; X rows and C rows are both L2-normalized."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = (X[i:i + chunk] @ C.t()).argmax(dim=1)
    return out


def kmeanspp_init(X, k, gen):
    n = X.shape[0]
    C = torch.empty(k, X.shape[1], device=X.device, dtype=X.dtype)
    i0 = int(torch.randint(n, (1,), generator=gen, device=X.device))
    C[0] = X[i0]
    closest = (2.0 - 2.0 * (X @ C[0])).clamp_min_(0)
    for j in range(1, k):
        tot = closest.sum()
        if tot <= 1e-20:
            C[j] = X[int(torch.randint(n, (1,), generator=gen, device=X.device))]
        else:
            pick = int(torch.multinomial(closest / tot, 1, generator=gen))
            C[j] = X[pick]
        closest = torch.minimum(closest, (2.0 - 2.0 * (X @ C[j])).clamp_min_(0))
    return C


def cosine_kmeans(X, k, iters, seed, init="kmeans++", log=""):
    """Cosine k-means on GPU. X is (N, d) and already L2-normalized."""
    gen = torch.Generator(device=X.device).manual_seed(seed)
    n, d = X.shape
    if init == "kmeans++" and k <= n:
        C = kmeanspp_init(X, k, gen)
    else:
        C = X[torch.randperm(n, generator=gen, device=X.device)[:k]].clone()
        if k > n:   # only reachable in tiny smoke settings
            C = torch.cat([C, X[torch.randint(n, (k - n,), generator=gen, device=X.device)]])
    C = F.normalize(C, dim=1)
    a = None
    for it in range(iters):
        a = _cos_assign(X, C)
        newC = torch.zeros_like(C)
        newC.index_add_(0, a, X)
        counts = torch.bincount(a, minlength=k)
        empty = counts == 0
        n_empty = int(empty.sum())
        if n_empty:
            newC[empty] = X[torch.randint(n, (n_empty,), generator=gen, device=X.device)]
        C = F.normalize(newC, dim=1)
        if it % 10 == 0 or it == iters - 1:
            print(f"  [kmeans{log}] iter {it + 1}/{iters}  empty={n_empty}", flush=True)
    return C, _cos_assign(X, C)


def build_kmeans(Xf, Xe, M, args, tag):
    C, af = cosine_kmeans(Xf, M, args.kmeans_iters, seed=100 + M, init=args.kmeans_init, log=f" {tag} M={M}")
    ae_ = _cos_assign(Xe, C)
    return af.cpu().numpy(), ae_.cpu().numpy(), {"iters": args.kmeans_iters, "init": args.kmeans_init}


def dead_threshold_for(mode, M, args):
    """`ema_dead_threshold` for one B variant.

    'default' is VQEmbedding's own 1.0. That value is calibrated for the autoencoder's
    regime, where ONE image contributes 1024 spatial vectors, so a code collects ~64 hits per
    batch. Here one image contributes ONE [CLS] token: `ema_cluster_size` is an EMA of the
    per-batch counts, so its steady state is batch/M -- 256/8192 = 0.031 at the largest M,
    permanently below 1.0. Every code is then flagged dead on every step, teleported onto a
    random vector, decayed back under the threshold, and flagged again; the codebook never
    settles. 'scaled' sets the threshold a fixed fraction of the uniform-usage rate batch/M,
    so it flags only codes used far less than their share, which is what the valve is for.
    Both are reported: 'default' is the literal reading of the plan, 'scaled' is the one that
    actually tests whether streaming EMA-VQ yields usable modes.
    """
    if mode == "default":
        return 1.0
    if mode == "scaled":
        return args.vq_dead_scale * args.vq_batch / float(M)
    raise SystemExit(f"unknown --vq-dead-modes entry {mode!r}")


def build_emavq(Xf_raw, Xe_raw, M, decay, epochs_list, args, dead_threshold, tag=""):
    """EMA-VQ streamed over the FIT set: the 'free' online candidate.

    Mirrors experiments/train_dualvae.py's `initialize_from_data` path (k-means seeding of
    the codebook + its EMA statistics from ONE batch, never the whole FIT set), then streams
    the FIT set in train mode under no_grad so only the EMA buffers move.
    """
    device = Xf_raw.device
    vq = VQEmbedding(num_embeddings=M, embedding_dim=Xf_raw.shape[1], l2_normalize=True,
                     use_ema=True, ema_decay=decay,
                     ema_dead_threshold=dead_threshold).to(device)

    # train_dualvae seeds from one batch whose location count exceeds num_embeddings (1024
    # vectors per image there). One [CLS] token is ONE vector, so a 256-image batch cannot
    # seed M > 256 centroids; the init batch is grown to M when needed and stays a single
    # contiguous batch, never the whole FIT set.
    n_init = max(args.vq_init_batch, M)
    n_init = min(n_init, Xf_raw.shape[0])
    g = torch.Generator(device="cpu").manual_seed(7)
    init_sel = torch.randperm(Xf_raw.shape[0], generator=g)[:n_init].to(device)
    with torch.no_grad():
        vq.init_from_data(Xf_raw[init_sel].t().reshape(1, Xf_raw.shape[1], n_init, 1))

    # all_reduce_sum_ must be a no-op here (single process): the EMA statistics are SUMS and
    # a live collective would double-count them.
    probe = torch.ones(3, device=device)
    all_reduce_sum_(probe)
    noop_ok = bool((not is_dist()) and torch.equal(probe, torch.ones(3, device=device)))

    out = {}
    restarts = []
    vq.train()
    max_ep = max(epochs_list)
    n = Xf_raw.shape[0]
    bs = args.vq_batch
    for ep in range(1, max_ep + 1):
        perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(1000 + ep)).to(device)
        ep_restarts = 0.0
        with torch.no_grad():
            for i in range(0, n, bs):
                sel = perm[i:i + bs]
                z = Xf_raw[sel].t().reshape(1, Xf_raw.shape[1], len(sel), 1)
                vq(z)
                ep_restarts += float(vq.restarted_codes)
        restarts.append(ep_restarts)
        print(f"  [emavq M={M} d={decay}{tag}] epoch {ep}/{max_ep}  "
              f"dead-code restarts={ep_restarts:.0f} "
              f"({ep_restarts / max(len(range(0, n, bs)), 1):.1f}/batch of {M} codes)", flush=True)
        if ep in epochs_list:
            vq.eval()
            with torch.no_grad():
                af = _vq_assign(vq, Xf_raw)
                ae_ = _vq_assign(vq, Xe_raw)
            out[ep] = (af, ae_, {"decay": decay, "epochs": ep, "init_batch": int(n_init),
                                 "dead_restarts_per_epoch": list(restarts),
                                 "all_reduce_sum_is_noop": noop_ok,
                                 "ema_dead_threshold": float(dead_threshold),
                                 "uniform_usage_rate_batch_over_M": bs / float(M),
                                 "batch_size": bs})
            vq.train()
    return out


@torch.no_grad()
def _vq_assign(vq, X, chunk=4096):
    was = vq.training
    vq.eval()
    out = np.empty(X.shape[0], dtype=np.int64)
    for i in range(0, X.shape[0], chunk):
        z = X[i:i + chunk].t().reshape(1, X.shape[1], -1, 1)
        _, _, idx, _, _ = vq(z)
        out[i:i + idx.shape[0]] = idx.cpu().numpy()
    vq.train(was)
    return out


def pca_fit(X, n_comp):
    """Centered PCA on GPU. Returns (mean, components (d, n_comp), eigenvalues (n_comp,))."""
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / max(X.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.flip(0)[:n_comp].clamp_min(1e-12)
    evecs = evecs.flip(1)[:, :n_comp]
    return mu, evecs.to(X.dtype), evals.to(X.dtype)


def build_fsq_pca(Xf, Xe, M, scale, args):
    """FSQ on a whitened PCA projection of the L2-normalized [CLS] ('FSQ on CLS')."""
    levels = LEVELS_BY_M.get(M)
    if levels is None:
        raise SystemExit(f"no FSQ level grid defined for M={M}; add it to LEVELS_BY_M.")
    d = len(levels)
    mu, V, ev = pca_fit(Xf, d)
    W = V / ev.sqrt().unsqueeze(0)        # whiten to unit variance on FIT
    fsq = FSQEmbedding(levels=levels, embedding_dim=d).to(Xf.device).eval()

    def proj_assign(X, chunk=16384):
        out = np.empty(X.shape[0], dtype=np.int64)
        with torch.no_grad():
            for i in range(0, X.shape[0], chunk):
                p = ((X[i:i + chunk] - mu) @ W) * scale
                _, _, idx, _, _ = fsq(p.t().reshape(1, d, -1, 1))
                out[i:i + idx.shape[0]] = idx.cpu().numpy()
        return out

    info = {"levels": levels, "pca_dim": d, "scale": scale,
            "explained_var_ratio_topd": float((ev.sum() / Xf.var(dim=0).sum()).clamp(0, 1)),
            "note": "the PCA projection is a one-off stand-in for a LEARNED projection"}
    return proj_assign(Xf), proj_assign(Xe), info


def latent_pca_features(args, device, n_comp=256):
    """Top-`n_comp` PCA of the flattened NORMALIZED latent, FIT-fitted, for method E."""
    Zf, Ze = load_flow_latents(args)
    mean, std = load_chan_stats(args, device)
    D = Zf.shape[1]
    cov = torch.zeros(D, D, dtype=torch.float32, device=device)
    s = torch.zeros(D, dtype=torch.float64, device=device)
    n = Zf.shape[0]
    for i, blk in iter_chunks(Zf, 4096):
        z = normalize_block(blk, mean, std, device)
        cov += z.t() @ z
        s += z.sum(dim=0).double()
    muv = (s / n).float()
    cov = cov / n - torch.outer(muv, muv)
    # float32, not float64: eigh on 8192x8192 in double would run at the L40S's 1/64 FP64
    # rate and take hours, and the top-256 subspace does not need that precision.
    evals, evecs = torch.linalg.eigh(cov)
    evecs = evecs.flip(1)[:, :n_comp].contiguous()

    def project(Z):
        out = torch.empty(Z.shape[0], n_comp, device=device)
        for i, blk in iter_chunks(Z, 4096):
            z = normalize_block(blk, mean, std, device)
            out[i:i + z.shape[0]] = (z - muv) @ evecs
        return out

    return project(Zf), project(Ze)


def cmd_step3(args):
    t0 = time.time()
    verify_inputs(args)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    cls_f = torch.from_numpy(np.load(os.path.join(args.out, "cls_fit.npy"))).float().to(device)
    cls_e = torch.from_numpy(np.load(os.path.join(args.out, "cls_eval.npy"))).float().to(device)
    lab_f = np.load(os.path.join(args.out, "labels_fit.npy"))
    lab_e = np.load(os.path.join(args.out, "labels_eval.npy"))
    Xf = F.normalize(cls_f, dim=1)
    Xe = F.normalize(cls_e, dim=1)

    methods = []

    def emit(method, M, variant, af, ae_, info):
        key = f"{method}_M{M}_{variant}"
        save_assign(args.out, key, af, ae_)
        methods.append({"key": key, "method": method, "M": int(M),
                        "variant": variant, "info": info})
        print(f"[step3] {key}: "
              f"{len(np.unique(af))} non-empty FIT modes", flush=True)

    for M in args.modes:
        # A. k-means (offline reference)
        af, ae_, info = build_kmeans(Xf, Xe, M, args, "cls")
        emit("A_kmeans_cls", M, "base", af, ae_, info)

        # B. EMA-VQ streaming (the 'free' online candidate)
        for decay in args.vq_decays:
            for tmode in args.vq_dead_modes:
                thr = dead_threshold_for(tmode, M, args)
                res = build_emavq(Xf, Xe, M, decay, sorted(set(args.vq_epochs)), args, thr,
                                  tag=f" thr={tmode}:{thr:g}")
                for ep, (a1, a2, inf) in res.items():
                    inf["dead_threshold_mode"] = tmode
                    emit("B_emavq_cls", M, f"d{decay}_ep{ep}_thr{tmode[:3]}", a1, a2, inf)

        # C. FSQ on a PCA projection
        for s in args.fsq_scales:
            af, ae_, info = build_fsq_pca(Xf, Xe, M, s, args)
            emit("C_fsq_pca_cls", M, f"s{s}", af, ae_, info)

        # D. uniform random baseline at the same M
        g = np.random.default_rng(500 + M)
        emit("D_random", M, "base",
             g.integers(0, M, size=len(lab_f)), g.integers(0, M, size=len(lab_e)),
             {"note": "uniform random assignment"})

    # D. class labels (M = 1000) and its matched random baseline
    n_cls = int(max(lab_f.max(), lab_e.max())) + 1
    emit("D_labels", n_cls, "base", lab_f, lab_e, {"note": "ImageNet class labels"})
    g = np.random.default_rng(500 + n_cls)
    emit("D_random", n_cls, "base",
         g.integers(0, n_cls, size=len(lab_f)), g.integers(0, n_cls, size=len(lab_e)),
         {"note": "uniform random assignment, matched to the label arm"})

    # E. own-latent reference: does the study need DINOv2 at all?
    if not args.skip_own_latent:
        print("[step3] method E: PCA of the normalized latent ...", flush=True)
        Lf, Le = latent_pca_features(args, device, n_comp=args.own_latent_pca)
        Lf = F.normalize(Lf, dim=1)
        Le = F.normalize(Le, dim=1)
        for M in args.modes:
            af, ae_, info = build_kmeans(Lf, Le, M, args, "latent")
            info["pca_dim"] = args.own_latent_pca
            emit("E_kmeans_latent", M, "base", af, ae_, info)
        del Lf, Le
        torch.cuda.empty_cache()

    record(args.out, "step3", {"methods": methods,
                               "modes": list(args.modes),
                               "n_classes": n_cls}, time.time() - t0)
    print(f"[step3] built {len(methods)} assignments in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- #
# Step 4: metrics per (method, M, variant), in the normalized latent space
# --------------------------------------------------------------------------- #
_FLOW_CACHE = {}


def load_chan_stats(args, device, out=None):
    st = np.load(os.path.join(out or args.out, "latent_stats.npz"))
    mean = torch.from_numpy(st["mean"]).float().to(device).view(1, -1, 1)
    std = torch.from_numpy(st["std"]).float().to(device).view(1, -1, 1).clamp_min(1e-6)
    return mean, std


def normalize_block(blk, mean, std, device):
    """(B, C*H*W) fp16 -> (B, C*H*W) fp32 per-channel normalized, on `device`."""
    c = mean.shape[1]
    z = torch.from_numpy(np.ascontiguousarray(blk)).to(device, non_blocking=True).float()
    z = z.view(z.shape[0], c, -1)
    z = (z - mean) / std
    return z.reshape(z.shape[0], -1)


def load_flow_latents(args, stem=CACHE_STEM):
    """FIT and EVAL latents as (n, D) fp16 host arrays, read once per process."""
    ck = stem
    if ck in _FLOW_CACHE:
        return _FLOW_CACHE[ck]
    idx_f = np.load(os.path.join(args.out, "indices_fit.npy"))
    idx_e = np.load(os.path.join(args.out, "indices_eval.npy"))
    lat_tr, _ = open_cache("train", stem)
    lat_va, _ = open_cache("val", stem)
    print(f"[latents] reading {len(idx_f)} FIT + {len(idx_e)} EVAL rows from {stem} ...", flush=True)
    Zf = read_rows(lat_tr, idx_f)
    Ze = read_rows(lat_va, idx_e)
    _FLOW_CACHE[ck] = (Zf, Ze)
    return Zf, Ze


def mode_moments(Z, assign, M, device, mean=None, std=None, chunk=4096):
    """FIT-side per-mode counts / sums / sums-of-squares, accumulated with index_add_."""
    D = Z.shape[1] if mean is None else Z.shape[1]
    a = torch.from_numpy(np.asarray(assign, dtype=np.int64)).to(device)
    assert int(a.max()) < M and int(a.min()) >= 0, f"assignment out of range for M={M}"
    s = torch.zeros(M, D, dtype=torch.float32, device=device)
    s2 = torch.zeros(M, D, dtype=torch.float32, device=device)
    counts = torch.bincount(a, minlength=M).float()
    tot = torch.zeros(D, dtype=torch.float64, device=device)
    tot2 = torch.zeros((), dtype=torch.float64, device=device)
    for i, blk in iter_chunks(Z, chunk):
        z = normalize_block(blk, mean, std, device) if mean is not None else \
            torch.from_numpy(np.ascontiguousarray(blk)).to(device).float()
        ai = a[i:i + z.shape[0]]
        s.index_add_(0, ai, z)
        s2.index_add_(0, ai, z * z)
        tot += z.sum(dim=0).double()
        tot2 += z.pow(2).sum().double()
    return counts, s, s2, tot, tot2


def eval_pass(Z, assign, mu, tr_sigma, mu_global, device, mean=None, std=None, chunk=4096):
    """G_eval / T_eval numerators and denominators in one pass over EVAL."""
    a = torch.from_numpy(np.asarray(assign, dtype=np.int64)).to(device)
    num_g = torch.zeros((), dtype=torch.float64, device=device)
    den_g = torch.zeros((), dtype=torch.float64, device=device)
    num_t = torch.zeros((), dtype=torch.float64, device=device)
    den_t = torch.zeros((), dtype=torch.float64, device=device)
    D = mu.shape[1]
    for i, blk in iter_chunks(Z, chunk):
        z = normalize_block(blk, mean, std, device) if mean is not None else \
            torch.from_numpy(np.ascontiguousarray(blk)).to(device).float()
        ai = a[i:i + z.shape[0]]
        d2 = (z - mu[ai]).pow(2).sum(dim=1).double()
        num_g += d2.sum()
        den_g += (z - mu_global).pow(2).sum(dim=1).double().sum()
        num_t += (d2 + tr_sigma[ai].double()).sum()
        den_t += (z.pow(2).sum(dim=1).double() + D).sum()
    return float(num_g / den_g), float(num_t / den_t)


def space_metrics(Zf, Ze, af, ae_, M, device, mean=None, std=None, shrink=10.0):
    """R_fit / G_eval / T_eval plus the usage numbers, for one (method, M, variant)."""
    counts, s, s2, tot, tot2 = mode_moments(Zf, af, M, device, mean, std)
    n = float(Zf.shape[0])
    D = s.shape[1]
    mu_global = (tot / n).float()
    nz = counts > 0
    cc = counts.clamp_min(1.0).unsqueeze(1)
    mu = s / cc
    mu[~nz] = mu_global                       # empty modes fall back to the global mean
    v = (s2 / cc - mu * mu).clamp_min(0)
    v[~nz] = 0.0

    # R_fit = WSS / TSS, both computed from the same moments.
    wss = float(tot2 - (counts.unsqueeze(1) * mu * mu).sum().double())
    tss = float(tot2 - n * mu_global.pow(2).sum().double())
    r_fit = wss / tss if tss > 0 else float("nan")

    # Diagonal per-mode variance shrunk toward the global per-dim variance.
    v_glob = ((tot2 / n / D) - mu_global.pow(2).mean().double()).clamp_min(0).float()
    v_glob_vec = (s2.sum(dim=0) / n - mu_global * mu_global).clamp_min(0)
    sig = (counts.unsqueeze(1) * v + shrink * v_glob_vec.unsqueeze(0)) / (counts.unsqueeze(1) + shrink)
    tr_sigma = sig.sum(dim=1)

    g_eval, t_eval = eval_pass(Ze, ae_, mu, tr_sigma, mu_global, device, mean, std)

    cnp = counts.cpu().numpy()
    small = np.where(cnp < 5)[0]
    frac_small = float(np.isin(np.asarray(ae_), small).mean()) if len(small) else 0.0
    return {
        "nonempty_modes": int(nz.sum()),
        "usage_perplexity": perplexity_of(cnp),
        "usage_perplexity_over_M": perplexity_of(cnp) / M,
        "largest_mode_share": float(cnp.max() / max(cnp.sum(), 1)),
        "frac_eval_in_modes_with_lt5_fit": frac_small,
        "R_fit": r_fit,
        "G_eval": g_eval,
        "T_eval": t_eval,
        "v_glob_scalar": float(v_glob),
    }


def cmd_step4(args):
    t0 = time.time()
    verify_inputs(args)
    device = torch.device(args.device)
    res = load_results(args.out)
    if "step3" not in res:
        raise SystemExit("step3 has not run: no assignments to score.")
    methods = res["step3"]["methods"]

    Zf, Ze = load_flow_latents(args)
    mean, std = load_chan_stats(args, device)
    # L2-normalized [CLS]: the space the k-means arm actually optimizes, so the
    # "k-means should win here" sanity check is stated in its own metric.
    Cf = np.load(os.path.join(args.out, "cls_fit.npy")).astype(np.float32)
    Ce = np.load(os.path.join(args.out, "cls_eval.npy")).astype(np.float32)
    Cf /= np.linalg.norm(Cf, axis=1, keepdims=True).clip(min=1e-12)
    Ce /= np.linalg.norm(Ce, axis=1, keepdims=True).clip(min=1e-12)
    lab_e = np.load(os.path.join(args.out, "labels_eval.npy"))

    # Matched random baseline R_fit for every M that appears in the table.
    random_R = {}
    for m in methods:
        if m["method"] == "D_random":
            af, ae_ = load_assign(args.out, m["key"])
            random_R[m["M"]] = space_metrics(Zf, Ze, af, ae_, m["M"], device, mean, std)
    kmeans_ref = {m["M"]: m["key"] for m in methods if m["method"] == "A_kmeans_cls"}

    rows = []
    for m in methods:
        key, M = m["key"], m["M"]
        af, ae_ = load_assign(args.out, key)
        print(f"[step4] {key} ...", flush=True)
        row = {"key": key, "method": m["method"], "M": M, "variant": m["variant"]}
        row.update(space_metrics(Zf, Ze, af, ae_, M, device, mean, std))
        rb = random_R.get(M)
        row["R_fit_random"] = rb["R_fit"] if rb else None
        row["G_eval_random"] = rb["G_eval"] if rb else None
        row["T_eval_random"] = rb["T_eval"] if rb else None

        # Same R and G in [CLS] space -- the sanity check (k-means should win there).
        cls_m = space_metrics(Cf, Ce, af, ae_, M, device)
        row["R_fit_cls"] = cls_m["R_fit"]
        row["G_eval_cls"] = cls_m["G_eval"]

        # Agreement, on the held-out EVAL assignments.
        ref = kmeans_ref.get(M)
        if ref and ref != key:
            row["NMI_vs_kmeans"] = nmi(ae_, load_assign(args.out, ref)[1])
        elif ref == key:
            row["NMI_vs_kmeans"] = 1.0
        else:
            row["NMI_vs_kmeans"] = None      # no k-means was built at this M
        row["NMI_vs_labels"] = nmi(ae_, lab_e)
        row["class_cond_mode_perplexity"] = class_cond_perplexity(ae_, lab_e)
        rows.append(row)

    payload = {"rows": rows,
               "metric_space": "per-channel-normalized latent, D=%d" % Zf.shape[1],
               "nmi_computed_on": "EVAL assignments",
               "sigma_shrinkage": 10.0}

    # Optional second latent space: method A on the plain-VAE cache, same FIT/EVAL rows.
    vae_stem = resolve_vae_stem(args)
    if vae_stem:
        print(f"[step4] plain-VAE arm: method A on the VAE latent ({vae_stem}) ...", flush=True)
        payload["plain_vae_arm"] = _plain_vae_arm(args, device, methods, vae_stem)
    else:
        payload["plain_vae_arm"] = {
            "available": False,
            "reason": "no complete plain-VAE latent cache in " +
                      ", ".join(os.path.dirname(x) for x in vae_stem_candidates(args))}

    record(args.out, "step4", payload, time.time() - t0)
    print(f"[step4] scored {len(rows)} rows in {time.time() - t0:.1f}s")


def _plain_vae_arm(args, device, methods, stem):
    Zf2, Ze2 = load_flow_latents(args, stem)
    c = int(np.load(stem.format(split="train") + ".latents.npy", mmap_mode="r").shape[1])
    s = np.zeros(c); s2 = np.zeros(c); ne = 0
    for i, blk in iter_chunks(Zf2, 4096):
        b = blk.reshape(blk.shape[0], c, -1).astype(np.float64)
        s += b.sum(axis=(0, 2)); s2 += (b ** 2).sum(axis=(0, 2)); ne += b.shape[0] * b.shape[2]
    mean = torch.from_numpy(s / ne).float().to(device).view(1, c, 1)
    std = torch.from_numpy(np.sqrt(np.maximum(s2 / ne - (s / ne) ** 2, 0))).float().to(device).view(1, c, 1).clamp_min(1e-6)
    out = {"available": True, "stem": stem, "flow_dim": int(Zf2.shape[1]), "rows": []}
    for m in methods:
        if m["method"] not in ("A_kmeans_cls", "D_random"):
            continue
        af, ae_ = load_assign(args.out, m["key"])
        r = space_metrics(Zf2, Ze2, af, ae_, m["M"], device, mean, std)
        out["rows"].append({"key": m["key"], "M": m["M"], "R_fit": r["R_fit"],
                            "G_eval": r["G_eval"], "T_eval": r["T_eval"]})
    return out


# --------------------------------------------------------------------------- #
# Step 5: coherence of per-patch sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_step5(args):
    t0 = time.time()
    verify_inputs(args)
    device = torch.device(args.device)
    ae = load_frozen_ae(AE_RUN_DIR, device)
    model = ae.model
    vq = model.vq_layer
    n_show = args.codemap_n

    idx_eval = np.load(os.path.join(args.out, "indices_eval.npy"))
    pick = np.sort(np.random.default_rng(3).choice(len(idx_eval), size=n_show, replace=False))
    sel = idx_eval[pick].astype(np.int64)
    loader = image_loader("val", sel, args)
    x = torch.cat([b["image"] for b in loader]).to(device).float()[:n_show]

    mean, std = load_chan_stats(args, device)          # flow-space per-channel stats

    # row 1: reconstruction through the deterministic latent the cache stores
    z1 = model.encode_for_diffusion(x, noise=torch.zeros(x.shape[0], model.latent_channels,
                                                         x.shape[2] // 8, x.shape[3] // 8,
                                                         device=device))
    r1 = model.decoder(z1)

    # the fitted per-patch mixture, straight out of the checkpoint buffers
    cb = vq.codebook.to(device)                        # (K, d)
    mu_k = vq.mu.to(device)                            # (K, d)
    sd_k = vq.sigma2_raw.clamp_min(0).sqrt().to(device)
    b, c, h, w = z1.shape

    def mixture_decode(idx_map):
        e = cb[idx_map] + mu_k[idx_map] + sd_k[idx_map] * torch.randn(
            idx_map.shape + (cb.shape[1],), device=device)
        z = e.permute(0, 3, 1, 2).contiguous()
        return model.decoder(model.attention(z))

    # row 2: the REAL code map, Delta resampled from the mixture
    p = encode_parts(model, x)
    k_real = p["idx"].reshape(b, h, w)
    torch.manual_seed(11)
    r2 = mixture_decode(k_real)

    # row 3: i.i.d. code map, k ~ pi per location
    pi = vq.pi.to(device).double()
    k_iid = torch.multinomial(pi, b * h * w, replacement=True).reshape(b, h, w)
    r3 = mixture_decode(k_iid)

    # row 4: Gaussian in flow space
    zg = torch.randn(b, c, h * w, device=device) * std + mean
    r4 = model.decoder(zg.reshape(b, c, h, w))

    png = os.path.join(args.out, "codemap_sampling.png")
    _plot_codemaps([r1, r2, r3, r4],
                   ["1. recon (noise=0)",
                    "2. real code map,\nDelta ~ mixture",
                    "3. i.i.d. code map,\nDelta ~ mixture",
                    "4. Gaussian in\nflow space"], png)

    payload = {
        "n_images": int(b),
        "png": png,
        "iid_code_map_unique_codes": int(torch.unique(k_iid).numel()),
        "real_code_map_unique_codes": int(torch.unique(k_real).numel()),
        "rows": ["decoder(encode_for_diffusion(x, noise=0))",
                 "decoder(attention(codebook[k] + mu_k + sigma_k * eps)) with the real k",
                 "same, with k ~ pi i.i.d. per location",
                 "decoder(z), z ~ N(per-channel mean, std) of the flow latent"],
    }
    record(args.out, "step5", payload, time.time() - t0)
    print(json.dumps(payload, indent=2))


def _plot_codemaps(rows, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_np(t):
        return ((t.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).float().cpu().numpy()

    imgs = [to_np(r) for r in rows]
    n = imgs[0].shape[0]
    canvas = np.concatenate([np.concatenate(list(r), axis=1) for r in imgs], axis=0)
    hpx = imgs[0].shape[1]
    fig_w = min(28.0, 1.1 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * canvas.shape[0] / canvas.shape[1] + 0.3))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([hpx * (i + 0.5) for i in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[step5] wrote {path}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    return f"{v:.{nd}f}"


def cmd_report(args):
    res = load_results(args.out)
    L = []
    A = L.append
    cfg = res.get("step0", {})
    A("# MM-FM mode tests on the frozen FSQ_0.1 latent")
    A("")
    A(f"Generated by `tools/mmfm_mode_tests.py` from `{results_path(args.out)}`.")
    A("")
    A(f"FIT = {cfg.get('fit_n')} train images (seed 0, sorted) · "
      f"EVAL = {cfg.get('eval_n')} val images"
      f"{' (the whole held-out split)' if cfg.get('eval_is_full_val') else ''} · "
      f"latent {cfg.get('latent_shape')} → D = {cfg.get('flow_dim')} · "
      f"modes from DINOv2-B/14 [CLS] (768-d).")
    A("")
    rt = res.get("runtime_sec", {})
    if rt:
        A("| step | runtime |")
        A("|---|---|")
        for k in ("step0", "step1", "step2", "step3", "step4", "step5"):
            if k in rt:
                A(f"| {k} | {rt[k] / 60:.1f} min |")
        A("")

    # ---- step 1
    s1 = res.get("step1")
    if s1:
        A("## Step 1 — unit checks on the autoencoder")
        A("")
        ar = s1["attention_relocation"]
        A(f"**1a. Attention relocation** ({s1['n_images']} images). "
          f"`attention(z_q + m)` vs `encode_for_diffusion(x, noise=0)`: "
          f"max abs diff **{ar['max_abs_diff_vs_encode_for_diffusion']:.3e}**. "
          f"Relative move `||attention(z_pre) − z_pre|| / ||z_pre||` = "
          f"**{ar['rel_move_attention_over_pre']:.4f}** — how far the flow space sits from "
          f"the mixture space.")
        A("")
        A(f"**1b. Consistency of the mixture statistics** "
          f"({s1['codes_with_min_hits']} codes with ≥ {s1['min_hits']} hits, "
          f"{s1['codes_touched']} codes touched, {s1['locations']} locations).")
        A("")
        A("| quantity | median relative error |")
        A("|---|---|")
        mc = s1["mixture_consistency_median_rel_err"]
        A(f"| EMA `mu` vs E[r\\|k] (is the EMA stale?) | {_f(mc['ema_mu_vs_E_r_given_k'])} |")
        A(f"| E[m\\|k] vs E[r\\|k] (residual head still identity?) | {_f(mc['E_m_given_k_vs_E_r_given_k'])} |")
        A(f"| `sigma2_raw` vs Var[r\\|k] | {_f(mc['sigma2_raw_vs_Var_r_given_k'])} |")
        A(f"| `sigma2_raw` vs Var[m\\|k] + E[s²\\|k] (what the decoder receives) | {_f(mc['sigma2_raw_vs_Var_m_plus_E_s2'])} |")
        A("")
        mg = s1["magnitudes"]
        A("| magnitude | value |")
        A("|---|---|")
        for k, v in mg.items():
            A(f"| {k} | {v:.4e} |")
        rh = s1["residual_head"]
        A("")
        A(f"Residual head: `||W_mean − I|| / ||I||` = **{_f(rh['rel_dev_W_mean_from_identity'])}**, "
          f"mean-bias norm = {_f(rh['mean_bias_norm'])}, "
          f"logvar-bias mean = {_f(rh['logvar_bias_mean'])}.")
        A("")

    # ---- step 2
    s2 = res.get("step2")
    if s2:
        b = s2["from_buffers"]
        d = s2["from_data"]
        A("## Step 2 — per-patch FSQ mixture statistics")
        A("")
        A(f"Grid `levels = {b['levels']}`, |C| = {b['num_codes']:,}.")
        A("")
        def _sep(v):
            return "—" if v is None else f"{v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}"

        nlive = b.get("codes_live")
        A("| quantity | all seen codes | live codes |")
        A("|---|---|---|")
        A(f"| count | {b['codes_seen']:,} ({_f(b['codes_seen_frac'])} of all codes) | "
          f"{nlive:,} ({_f(b.get('codes_live_frac_of_seen'))} of seen) |")
        A(f"| perplexity of pi | {b['perplexity_pi']:.1f} | "
          f"{b['perplexity_pi_seen_only']:.1f} (seen, renormalized) |")
        A(f"| fraction of sigma2_raw ≤ floor ({b['sigma2_floor']:g}) | "
          f"{_f(b['frac_at_or_below_floor'])} | {_f(b.get('frac_at_or_below_floor_live'))} |")
        A(f"| fraction of sigma2_raw ≥ ceil ({b['sigma2_ceil']:g}) | "
          f"{_f(b['frac_at_or_above_ceil'])} | — |")
        A(f"| mean \\|mu\\| / mean sigma | {_f(b['mean_abs_mu_over_mean_sigma'])} | "
          f"{_f(b.get('mean_abs_mu_over_mean_sigma_live'))} |")
        A(f"| separation ratio (spacing / sigma), p10 / p50 / p90 | "
          f"{_sep(b['separation_ratio_percentiles'])} | "
          f"{_sep(b.get('separation_ratio_percentiles_live'))} |")
        A("")
        ecp = b.get("ema_cluster_size_percentiles_over_seen")
        A(f"A code is **live** when `ema_cluster_size` ≥ {b.get('live_count_min')}, i.e. its EMA "
          f"mass is still worth at least one effective hit. `codes_seen` is cumulative and "
          f"does not decay, so a code hit once early in training still counts as seen while "
          f"its statistics have decayed to a float32 denormal — which is what drives the "
          f"floor fraction and the separation ratio in the left column. "
          + (f"`ema_cluster_size` over seen codes, p10/p50/p90: "
             f"{ecp[0]:.3e} / {ecp[1]:.3e} / {ecp[2]:.3e}. " if ecp else "")
          + "Floor/ceil fractions are over (code, channel) entries, not codes.")
        A("")
        A("`sigma2_raw` percentiles per channel:")
        A("")
        A("| channel | p10 (seen) | p50 (seen) | p90 (seen) | p10 (live) | p50 (live) | p90 (live) |")
        A("|---|---|---|---|---|---|---|")
        lv = b.get("sigma2_raw_percentiles_per_channel_live", {})
        for ch, v in b["sigma2_raw_percentiles_per_channel"].items():
            w = lv.get(ch)
            wcells = (f"{w[0]:.3e} | {w[1]:.3e} | {w[2]:.3e}" if w else "— | — | —")
            A(f"| {ch} | {v[0]:.3e} | {v[1]:.3e} | {v[2]:.3e} | {wcells} |")
        A("")
        A(f"Centre/boundary density ratio of `pre_quant(z_e_vq)` over {d['n_images']} images "
          f"(> 1 means mass piles up at cell centres, i.e. real per-channel multimodality): "
          f"**mean {_f(d['centre_over_boundary_density_ratio_mean'], 3)}**.")
        A("")
        A("| channel | " + " | ".join(d["centre_over_boundary_density_ratio_per_channel"].keys()) + " |")
        A("|---|" + "---|" * len(d["centre_over_boundary_density_ratio_per_channel"]))
        A("| ratio | " + " | ".join(f"{v:.3f}" for v in
                                    d["centre_over_boundary_density_ratio_per_channel"].values()) + " |")
        A("")
        A(f"![FSQ channel histograms]({os.path.basename(s2['png'])})")
        A("")

    # ---- steps 3 & 4
    s4 = res.get("step4")
    if s4:
        A("## Steps 3 & 4 — image-level modes and their metrics")
        A("")
        A("All metrics in the per-channel-normalized latent space "
          f"({s4.get('metric_space')}), per-mode statistics estimated on FIT. "
          "NMI is computed on the EVAL assignments.")
        A("")
        A("**Two traps when reading this table.**")
        A("")
        A("1. *NMI has no fixed scale here.* Between two partitions of thousands of tiny "
          "clusters it is inflated by finite-sample bias, so a value near 0.7 can be at or "
          "BELOW chance. Read every NMI against the `D_random` row at the SAME M "
          "(the per-M baselines are listed in the reading guide below); the absolute number "
          "means nothing on its own.")
        A("2. *G_eval is inflated by mode-mean estimation noise, not only by a worse "
          "partition.* A mode mean built from n_m = N/M FIT samples in D dimensions carries "
          "variance ~ 1/n_m per dimension, which costs roughly M/N of the total when applied "
          "to held-out data. In-sample R_fit is biased down by about the same amount, so "
          "expect `G_eval ~= R_fit + 2M/N` even for a partition that transfers perfectly. "
          "At M = 8192 with N = 256k that is 6.4 points, which is why the high-M rows drift "
          "above 1 without the partition itself having got worse.")
        A("")
        A("| method | M | variant | non-empty | perp/M | largest share | <5 FIT | "
          "R_fit | R_fit(rand) | G_eval | T_eval | R_fit[CLS] | G_eval[CLS] | "
          "NMI vs k-means | NMI vs labels | modes/class |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = {"A_kmeans_cls": 0, "B_emavq_cls": 1, "C_fsq_pca_cls": 2,
                 "D_random": 3, "D_labels": 4, "E_kmeans_latent": 5}
        rows = sorted(s4["rows"], key=lambda r: (r["M"], order.get(r["method"], 9), r["variant"]))
        for r in rows:
            A("| {m} | {M} | {v} | {ne} | {pm} | {ls} | {sm} | {R} | {Rr} | {G} | {T} | "
              "{Rc} | {Gc} | {nk} | {nl} | {cc} |".format(
                  m=r["method"], M=r["M"], v=r["variant"], ne=r["nonempty_modes"],
                  pm=_f(r["usage_perplexity_over_M"], 3), ls=_f(r["largest_mode_share"], 4),
                  sm=_f(r["frac_eval_in_modes_with_lt5_fit"], 4),
                  R=_f(r["R_fit"]), Rr=_f(r["R_fit_random"]), G=_f(r["G_eval"]),
                  T=_f(r["T_eval"]), Rc=_f(r["R_fit_cls"]), Gc=_f(r["G_eval_cls"]),
                  nk=_f(r["NMI_vs_kmeans"], 3), nl=_f(r["NMI_vs_labels"], 3),
                  cc=_f(r["class_cond_mode_perplexity"], 1)))
        A("")
        pv = s4.get("plain_vae_arm", {})
        if pv.get("available"):
            A("### Plain-VAE latent (same FIT/EVAL rows, method A + random)")
            A("")
            A("| key | M | R_fit | G_eval | T_eval |")
            A("|---|---|---|---|---|")
            for r in pv["rows"]:
                A(f"| {r['key']} | {r['M']} | {_f(r['R_fit'])} | {_f(r['G_eval'])} | {_f(r['T_eval'])} |")
            A("")
        else:
            A(f"_Plain-VAE arm skipped: {pv.get('reason', 'unavailable')}._")
            A("")
        A(_reading_guide(rows))
        A("")

    # ---- step 5
    s5 = res.get("step5")
    if s5:
        A("## Step 5 — coherence of per-patch sampling")
        A("")
        for i, r in enumerate(s5["rows"], 1):
            A(f"{i}. `{r}`")
        A("")
        A(f"Real code map: {s5['real_code_map_unique_codes']} distinct codes over "
          f"{s5['n_images']} images; i.i.d. code map: {s5['iid_code_map_unique_codes']}.")
        A("")
        A(f"![code-map sampling]({os.path.basename(s5['png'])})")
        A("")

    path = os.path.join(args.out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] wrote {path}")


def _reading_guide(rows):
    """The plan's heuristic thresholds, applied to the numbers actually measured."""
    out = ["### Reading guide (heuristic thresholds from the plan, not established ones)", ""]
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["M"]), []).append(r)
    Ms = sorted({r["M"] for r in rows if r["method"] == "A_kmeans_cls"})
    n_fit = None
    for M in Ms:
        a = by.get(("A_kmeans_cls", M), [None])[0]
        if a is None:
            continue
        rnd = by.get(("D_random", M), [None])[0]
        if rnd is not None:
            out.append(f"- **M = {M} chance baselines** (random assignment at the same M): "
                       f"NMI vs k-means {_f(rnd.get('NMI_vs_kmeans'), 3)}, NMI vs labels "
                       f"{_f(rnd.get('NMI_vs_labels'), 3)}, modes/class "
                       f"{_f(rnd.get('class_cond_mode_perplexity'), 1)}. Any method scoring "
                       f"near these is agreeing with the reference no better than chance.")
        gr = a.get("G_eval_random")
        verdict = ("worth pursuing" if (a["T_eval"] <= 0.8 and gr is not None
                                        and a["G_eval"] < 0.95 * gr) else "not met")
        out.append(f"- **M = {M}, image-level source (A):** T_eval = {_f(a['T_eval'])} "
                   f"(threshold ≤ 0.80), G_eval = {_f(a['G_eval'])} vs random "
                   f"{_f(a['G_eval_random'])} → **{verdict}**.")
        # Pick the best variant AMONG THOSE THAT CLEAR THE USAGE GATE. Ranking by G_eval
        # alone rewards exactly the failure this study has to exclude: a method that
        # collapses to a handful of modes gets a better held-out G (few, well-estimated
        # means) while answering none of the question. Only if nothing clears the gate do we
        # fall back to the overall best, and then the line says so.
        def _pick(key):
            rows_k = by.get((key, M), [])
            if not rows_k:
                return None, False
            gated = [r for r in rows_k if r["usage_perplexity_over_M"] >= 0.5]
            pool = gated or rows_k
            return min(pool, key=lambda r: r["G_eval"]), bool(gated)

        best_b, b_gated = _pick("B_emavq_cls")
        if best_b:
            rel = (best_b["G_eval"] - a["G_eval"]) / max(a["G_eval"], 1e-12)
            ok = rel <= 0.05 and best_b["usage_perplexity_over_M"] >= 0.5
            note = "" if b_gated else " (no variant cleared perp/M ≥ 0.5; showing overall best)"
            out.append(f"  - online EMA-VQ (B, best `{best_b['variant']}`): G_eval "
                       f"{_f(best_b['G_eval'])} = {rel * 100:+.2f}% vs A, perp/M "
                       f"{_f(best_b['usage_perplexity_over_M'], 3)} → "
                       f"**{'free in practice' if ok else 'not free'}** "
                       f"(needs ≤ +5% and perp/M ≥ 0.5){note}.")
        best_c, c_gated = _pick("C_fsq_pca_cls")
        if best_c:
            ok = best_c["usage_perplexity_over_M"] >= 0.5
            note = "" if c_gated else " (no variant cleared perp/M ≥ 0.5; showing overall best)"
            out.append(f"  - FSQ-on-CLS (C, best `{best_c['variant']}`): perp/M "
                       f"{_f(best_c['usage_perplexity_over_M'], 3)}, G_eval "
                       f"{_f(best_c['G_eval'])} vs A {_f(a['G_eval'])} → "
                       f"**{'viable' if ok else 'not viable'}** (needs perp/M ≥ 0.5){note}.")
        best_e, _ = _pick("E_kmeans_latent")
        if best_e:
            out.append(f"  - own-latent reference (E): G_eval {_f(best_e['G_eval'])}, T_eval "
                       f"{_f(best_e['T_eval'])}, NMI vs [CLS] k-means "
                       f"{_f(best_e['NMI_vs_kmeans'], 3)} — modes taken from the latent's own "
                       f"principal structure rather than from DINOv2.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# encode_vae: the optional plain-VAE latent cache (step 4's second latent space)
# --------------------------------------------------------------------------- #
def cmd_encode_vae(args):
    """Encode the plain-VAE latent cache, reusing the flow trainer's own writer.

    This exists only to feed step 4's optional arm, which scores the SAME DINOv2 mode
    assignment against a second latent space -- the control that separates "this partition
    makes a latent look mixture-shaped" from "the FSQ latent in particular is mixture-shaped".

    encode_split_memmap is imported from experiments/train_latent_flow.py rather than
    reimplemented, so the file this writes is the one that trainer would write and later map:
    same stem, same shuffle=False row order, same posterior-MEAN latent (latent_sample=False),
    same bf16-autocast-then-fp16 storage. Nothing in that module is modified.
    """
    from types import SimpleNamespace

    from experiments.train_latent_flow import encode_split_memmap, memmap_cache_exists

    t0 = time.time()
    verify_inputs(args, need_cache=False)
    if not os.path.isdir(VAE_RUN_DIR):
        raise SystemExit(f"plain-VAE run directory not found: {VAE_RUN_DIR}")

    cache_dir = args.vae_cache_dir or VAE_CACHE_DIRS[0]
    os.makedirs(cache_dir, exist_ok=True)
    stem_tpl = os.path.join(cache_dir, VAE_CACHE_NAME)

    # select_device returns the STRING "cuda"; encode_split_memmap gates its bf16 autocast on
    # `device == "cuda"`, so passing a torch.device here would silently encode in fp32 and the
    # two caches would not be numerically comparable.
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    ae = load_frozen_ae(VAE_RUN_DIR, device)

    eargs = SimpleNamespace(
        encode_batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
        latent_sample=False,          # cache the posterior MEAN, as configs/flow_in_vae.yaml does
        resize_img=256,
        use_amp=True,
        dataset_name="imagenet",
    )

    written = {}
    for split in ("train", "val"):
        stem = stem_tpl.format(split=split)
        if memmap_cache_exists(stem, False) and not args.force:
            print(f"[encode_vae] {split}: cache already present, skipping ({stem}.latents.npy)")
            written[split] = {"stem": stem, "skipped": True}
            continue
        ds = HDF5ImageDataset(H5_PATH, split,
                              transform=build_image_transform("imagenet", 256), labeled=True)
        c, h, w = ae.latent_shape(256)
        gib = len(ds) * c * h * w * 2 / 2 ** 30
        print(f"[encode_vae] {split}: {len(ds):,} images -> ({c}, {h}, {w}) fp16 = {gib:.2f} GiB "
              f"at {stem}.latents.npy", flush=True)
        t1 = time.time()
        encode_split_memmap(ae=ae, dataset=ds, args=eargs, device=device, stem=stem,
                            flip=False, desc=f"encode vae {split}")
        written[split] = {"stem": stem, "skipped": False, "n": len(ds),
                          "gib": round(gib, 2), "seconds": round(time.time() - t1, 1)}
        print(f"[encode_vae] {split}: done in {(time.time() - t1) / 60:.1f} min", flush=True)

    ok = _stem_complete(stem_tpl)
    payload = {"cache_dir": cache_dir, "stem": stem_tpl, "complete": bool(ok),
               "ae_checkpoint": ae.ae_checkpoint, "splits": written}
    record(args.out, "encode_vae", payload, time.time() - t0)
    print(json.dumps(payload, indent=2))
    if not ok:
        raise SystemExit("encode_vae finished but the cache is still incomplete")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_TORCH_HOME = "/data/image-models-project/manfred/.cache/torch"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="reports/mmfm_mode_tests",
                        help="output directory (results.json, PNGs, summary.md)")
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--num-workers", type=int, default=12)
    common.add_argument("--fit-n", type=int, default=256000)
    common.add_argument("--eval-n", type=int, default=0, help="0 = the whole val split")
    common.add_argument("--unit-n", type=int, default=20000,
                        help="images used by the step 1 / step 2 data checks")
    common.add_argument("--min-hits", type=int, default=30,
                        help="step 1b: minimum per-code hits for a code to be reported")
    common.add_argument("--modes", type=int, nargs="+", default=[1024, 4096, 8192])
    common.add_argument("--kmeans-iters", type=int, default=30)
    common.add_argument("--kmeans-init", default="kmeans++", choices=["kmeans++", "random"])
    common.add_argument("--vq-epochs", type=int, nargs="+", default=[1, 3])
    common.add_argument("--vq-decays", type=float, nargs="+", default=[0.99, 0.999])
    common.add_argument("--vq-batch", type=int, default=256)
    common.add_argument("--vq-dead-modes", nargs="+", default=["default", "scaled"],
                        choices=["default", "scaled"],
                        help="ema_dead_threshold regimes for method B; see dead_threshold_for")
    common.add_argument("--vq-dead-scale", type=float, default=0.1,
                        help="'scaled' threshold = this x (vq_batch / M), i.e. flag a code "
                             "dead only when it is used this fraction of uniform usage")
    common.add_argument("--vq-init-batch", type=int, default=256,
                        help="EMA-VQ init batch; grown to M when M exceeds it "
                             "(one [CLS] token is one vector, unlike the 1024 per image "
                             "that train_dualvae.py's k-means init sees)")
    common.add_argument("--fsq-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    common.add_argument("--own-latent-pca", type=int, default=256)
    common.add_argument("--skip-own-latent", action="store_true",
                        help="skip method E (the optional own-latent reference)")
    common.add_argument("--codemap-n", type=int, default=16)
    common.add_argument("--vae-cache-dir", default=None,
                        help="where the plain-VAE latent cache lives / is written "
                             f"(default {VAE_CACHE_DIRS[0]})")
    common.add_argument("--encode-batch-size", type=int, default=64,
                        help="encode_vae only; matches configs/flow_in_vae.yaml")
    common.add_argument("--force", action="store_true",
                        help="encode_vae only: re-encode even if the cache exists")
    common.add_argument("--live-count-min", type=float, default=1.0,
                        help="step 2: minimum ema_cluster_size for a seen code to count as "
                             "LIVE (its EMA mass has not decayed below one effective hit)")
    common.add_argument("--torch-home", default=os.environ.get("TORCH_HOME", DEFAULT_TORCH_HOME),
                        help="torch hub root holding facebookresearch_dinov2_main and "
                             "hub/checkpoints/dinov2_vitb14_pretrain.pth")

    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptxt in (
            ("verify", lambda a: verify_inputs(a), "check every input path and stop"),
            ("step0", cmd_step0, "FIT/EVAL subsets, DINOv2 [CLS], flow-space normalization"),
            ("step1", cmd_step1, "unit checks on the autoencoder"),
            ("step2", cmd_step2, "per-patch FSQ mixture statistics"),
            ("step3", cmd_step3, "image-level mode assignments"),
            ("step4", cmd_step4, "metrics per (method, M, variant)"),
            ("step5", cmd_step5, "coherence of per-patch sampling"),
            ("report", cmd_report, "write summary.md from results.json"),
            ("encode_vae", cmd_encode_vae,
             "encode the optional plain-VAE latent cache for step 4's second arm"),
    ):
        sp = sub.add_parser(name, parents=[common], help=helptxt)
        sp.set_defaults(func=fn)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU.")
        args.device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    args.func(args)


if __name__ == "__main__":
    main()
