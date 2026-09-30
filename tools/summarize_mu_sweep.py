"""Table of the residual-mean sweep (Delta = alpha * mu + sigma * eps) for one checkpoint.

    python tools/summarize_mu_sweep.py --step 0200000

alpha = 1 comes from the plain evaluation files; every other alpha from mu_sweep_<step>/.
Missing runs show as '-', so this can be run while the sweep is still going.
"""
import argparse
import csv
import glob
import json
import os
import re

E = "/data/image-models-project/manfred/eval_export"
COLS = ["alpha", "rfid", "kid", "lpips", "psnr", "ssim", "mse",
        "repa_cos", "cknna_patch_t0.25", "cknna_patch_t0.5", "cknna_patch_t0.75",
        "cknna_cls_t0.5"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def row(alpha, rfid, cknna):
    r = {"alpha": alpha}
    if rfid:
        m = rfid["results"]["imagenet-val-h5"]
        r.update({k: m[k] for k in ("rfid", "kid", "lpips", "psnr", "ssim", "mse")})
    if cknna:
        c = cknna["cknna"]
        r["repa_cos"] = cknna["repa_cosine_projected"]
        for t in ("0.25", "0.5", "0.75"):
            r[f"cknna_patch_t{t}"] = c[f"t={t}"]["vs_dino_patch_mean"]
        r["cknna_cls_t0.5"] = c["t=0.5"]["vs_dino_cls"]
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", default="0200000")
    args = p.parse_args()
    d = os.path.join(E, f"mu_sweep_{args.step}")

    rows = [row(1.0, load(f"{E}/vae_step_{args.step}/rfid_imagenet.json"),
                load(f"{E}/cknna_eval_step_{args.step}.json"))]
    alphas = sorted({float(re.search(r"mu([\d.]+)\.json", f).group(1))
                     for f in glob.glob(f"{d}/*_mu*.json")}, reverse=True)
    for a in alphas:
        rows.append(row(a, load(f"{d}/rfid_mu{a:.3f}.json"), load(f"{d}/cknna_mu{a:.3f}.json")))

    print(f"step {args.step}   Delta = alpha * mu + sigma * eps   (50k ImageNet val)\n")
    print("".join(f"{c:>18}" if c.startswith("cknna") else f"{c:>9}" for c in COLS))
    for r in rows:
        cells = []
        for c in COLS:
            v, w = r.get(c), 18 if c.startswith("cknna") else 9
            fmt = ".3f" if c == "alpha" else ".2e" if c == "kid" else ".4f"
            cells.append(f"{v:>{w}{fmt}}" if v is not None else f"{'-':>{w}}")
        print("".join(cells))

    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {d}/summary.csv")


if __name__ == "__main__":
    main()
