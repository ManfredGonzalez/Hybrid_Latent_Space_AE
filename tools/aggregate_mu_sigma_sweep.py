"""Combine the per-checkpoint rfid_grid.csv files from tools/mu_sigma_sweep.py into one
CSV and print an a x b rFID pivot table per model, so the 4 sweeps can be compared side by
side.

Usage:
    python tools/aggregate_mu_sigma_sweep.py --glob 'results/mu_sigma_sweep/*/rfid_grid.csv'
"""
import argparse
import csv
import glob
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="results/mu_sigma_sweep/*/rfid_grid.csv")
    p.add_argument("--out", default="results/mu_sigma_sweep/combined.csv")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {args.glob!r}")

    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows from {len(paths)} files)")

    by_model = {}
    for r in rows:
        key = (r["checkpoint_dir"], r["quantizer"])
        by_model.setdefault(key, []).append(r)

    b_values = sorted({r["b"] for r in rows}, key=float, reverse=True)
    a_values = sorted({r["a"] for r in rows}, key=float, reverse=True)

    for (ckpt, quantizer), model_rows in by_model.items():
        grid = {(r["a"], r["b"]): float(r["rfid"]) for r in model_rows}
        print(f"\n=== {ckpt}  (quantizer={quantizer}) ===")
        header = "a\\b".ljust(8) + "".join(b.ljust(9) for b in b_values)
        print(header)
        for a in a_values:
            line = a.ljust(8)
            for b in b_values:
                v = grid.get((a, b))
                line += (f"{v:8.3f} " if v is not None else "   n/a   ")
            print(line)


if __name__ == "__main__":
    main()
