"""Figures + LaTeX table for the matched flow evaluation produced by tools/eval_flow_runs.py.

Three panels:

  (a) Gen/FID vs NFE, log-x. The headline of the step sweep. All runs are scored against the
      same reals with the same number of fakes, so the vertical gaps are comparable and the
      SLOPE is the quantity of interest: a latent space whose FID degrades less as the solver
      budget shrinks is the one the learned velocity field is straighter in.

  (b) FID normalized to each run's own 50-step value. Panel (a) confounds "better at every
      budget" with "degrades more slowly"; dividing out each run's own best value separates
      them. A flat curve means the latent space is cheap to sample; a steep one means the
      run only looks good because it was given 100 network evaluations.

  (c) Held-out flow MSE per t bin, uniform t. Diagnostic for the loss decomposition: the CFM
      objective is model error plus the irreducible Var[x1 - x0 | x_t, t], and the second
      term rises with the multimodality of the latent. If a run's MSE is higher only in the
      mid-t bins, that is structure in p(x1), not a worse fit.

Also emits the rFID ceiling alongside Gen/FID when --rfid is given: generative FID is lower
bounded by how well the autoencoder can reconstruct at all, so the honest quantity for the
"easier to generate in" claim is the GAP between them, not Gen/FID alone.

Usage:
    python -m tools.plot_flow_eval --json reports/flow_eval_matched.json \
        --labels "VAE (std, beta=.001)" "DUALVAE D+ (code, beta=.001)" "DUALVAE B+ (std, beta=.1)" \
        --rfid 4.08 3.97 3.70 --out reports/flow_eval_matched
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-safe, distinguishable in greyscale by marker as well as hue.
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
MARKERS = ["o", "s", "^", "D", "v"]


def load(path):
    with open(path) as f:
        return json.load(f)


def make_figure(data, labels, rfid, out_prefix):
    runs = data["runs"]
    labels = labels or [r["run"] for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- (a) FID vs NFE -------------------------------------------------------------
    ax = axes[0]
    for i, (r, lab) in enumerate(zip(runs, labels)):
        nfe = [m["nfe"] for m in r["fid_sweep"]]
        fid = [m["gen_fid"] for m in r["fid_sweep"]]
        ax.plot(nfe, fid, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                label=lab, lw=2, ms=6)
        if rfid:
            ax.axhline(rfid[i], color=COLORS[i % len(COLORS)], ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE (network evaluations per sample)")
    ax.set_ylabel("Gen/FID  (lower is better)")
    ax.set_title(f"(a) Gen/FID vs sampling budget\n{data['n_real']} reals, "
                 f"{runs[0]['n_fake']} fakes, identical protocol")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if rfid:
        ax.text(0.02, 0.03, "dotted = each AE's rFID ceiling", transform=ax.transAxes,
                fontsize=7, alpha=0.7)

    # ---- (b) FID relative to each run's own best budget ------------------------------
    ax = axes[1]
    for i, (r, lab) in enumerate(zip(runs, labels)):
        sweep = sorted(r["fid_sweep"], key=lambda m: m["steps"])
        nfe = [m["nfe"] for m in sweep]
        fid = np.array([m["gen_fid"] for m in sweep])
        ax.plot(nfe, fid / fid[-1], marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], label=lab, lw=2, ms=6)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE")
    ax.set_ylabel("FID / FID at full budget")
    ax.set_title("(b) Degradation as the budget shrinks\n(flatter = straighter field)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (c) flow MSE by t bin -------------------------------------------------------
    ax = axes[2]
    for i, (r, lab) in enumerate(zip(runs, labels)):
        curve = r["flow_mse_by_t"]["mse"]
        nb = r["flow_mse_by_t"]["bins"]
        centers = (np.arange(nb) + 0.5) / nb
        ax.plot(centers, curve, marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], label=lab, lw=2, ms=5)
    ax.set_xlabel("t   (0 = noise, 1 = data)")
    ax.set_ylabel("held-out flow MSE")
    ax.set_title("(c) Where the CFM loss lives\n(mid-t = irreducible mixture variance)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=160, bbox_inches="tight")
    print(f"Wrote {out_prefix}.pdf / .png")


def make_table(data, labels, rfid, out_prefix):
    """LaTeX table: matched Gen/FID at every budget, the rFID ceiling and the gap."""
    runs = data["runs"]
    labels = labels or [r["run"] for r in runs]
    steps = sorted(m["steps"] for m in runs[0]["fid_sweep"])

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Matched generative evaluation of the three latent-flow runs. All rows are "
        rf"scored against the SAME {data['n_real']} real validation images with the SAME "
        rf"{runs[0]['n_fake']} generated samples, fixing the unequal \texttt{{gen\_fid\_n\_samples}} "
        r"(1000 vs 2000) used during training. Solver, guidance scale and sample seed are each "
        r"run's own trained values; only the ODE step count varies. rFID is the autoencoder's "
        r"reconstruction FID on the same split (a floor Gen/FID cannot beat); the gap is the part "
        r"attributable to the generator rather than the autoencoder.}",
        r"\label{tab:flow-matched}",
        r"\begin{tabular}{l" + "r" * (len(steps) + 3) + "}",
        r"\toprule",
        r"Run & " + " & ".join(rf"FID@{s}" for s in steps) +
        r" & rFID & gap & val MSE \\",
        r"\midrule",
    ]
    for i, (r, lab) in enumerate(zip(runs, labels)):
        by = {m["steps"]: m["gen_fid"] for m in r["fid_sweep"]}
        cells = [f"{by[s]:.2f}" for s in steps]
        ceil = rfid[i] if rfid else None
        ceil_cell = f"{ceil:.2f}" if ceil is not None else "--"
        gap_cell = f"{by[steps[-1]] - ceil:.2f}" if ceil is not None else "--"
        lines.append(f"{lab} & " + " & ".join(cells) +
                     f" & {ceil_cell} & {gap_cell} & {r['val_flow_mse_raw']:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = f"{out_prefix}_table.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


# The compact t table shows every other bin: enough to read the shape (rise, hump, endpoint)
# without printing all ten. The full curve stays in the JSON and in panel (c).
T_SHOW = (0.05, 0.25, 0.45, 0.65, 0.85, 0.95)


def t_cells(run, bins=T_SHOW):
    """Pick the requested bin centres out of a run's t-binned MSE curve."""
    nb = run["flow_mse_by_t"]["bins"]
    curve = run["flow_mse_by_t"]["mse"]
    out = []
    for t in bins:
        idx = min(int(t * nb), nb - 1)
        out.append(curve[idx])
    return out


def make_t_table(data, labels, out_prefix):
    """LaTeX table of the held-out flow MSE per t bin (uniform t)."""
    runs = data["runs"]
    labels = labels or [r["run"] for r in runs]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Held-out flow-matching MSE by trajectory position $t$ (uniform $t$, full "
        r"validation split, each model in its OWN standardized latent space). $t{=}0$ is pure "
        r"noise, $t{=}1$ is data. The CFM loss is model error plus the irreducible "
        r"$\mathrm{Var}[x_1-x_0\mid x_t,t]$, and the second term grows with the multimodality of "
        r"the latent -- so a higher mid-$t$ value is a property of the latent distribution, not a "
        r"worse fit. All runs converge at $t\to1$, where the target variance is $\mathrm{Var}(x_0)=1$ "
        r"per dimension for everyone.}",
        r"\label{tab:flow-tbins}",
        r"\begin{tabular}{l" + "r" * len(T_SHOW) + "}",
        r"\toprule",
        r"$t$ & " + " & ".join(f"${t:.2f}$" for t in T_SHOW) + r" \\",
        r"\midrule",
    ]
    for r, lab in zip(runs, labels):
        lines.append(f"{lab} & " + " & ".join(f"{v:.3f}" for v in t_cells(r)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = f"{out_prefix}_tbins.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def print_t_summary(data, labels):
    runs = data["runs"]
    labels = labels or [r["run"] for r in runs]
    w = max(len(l) for l in labels) + 2
    hdr = f"{'t':<{w}}" + "".join(f"{t:>9.2f}" for t in T_SHOW)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r, lab in zip(runs, labels):
        print(f"{lab:<{w}}" + "".join(f"{v:>9.3f}" for v in t_cells(r)))


def print_summary(data, labels, rfid):
    runs = data["runs"]
    labels = labels or [r["run"] for r in runs]
    steps = sorted(m["steps"] for m in runs[0]["fid_sweep"])
    w = max(len(l) for l in labels) + 2
    hdr = f"{'run':<{w}}" + "".join(f"{'FID@' + str(s):>10}" for s in steps)
    hdr += f"{'rFID':>8}{'gap':>9}{'val MSE':>10}{'FID@4/FID@50':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, (r, lab) in enumerate(zip(runs, labels)):
        by = {m["steps"]: m["gen_fid"] for m in r["fid_sweep"]}
        row = f"{lab:<{w}}" + "".join(f"{by[s]:>10.2f}" for s in steps)
        ceil = rfid[i] if rfid else float("nan")
        row += f"{ceil:>8.2f}{by[steps[-1]] - ceil:>9.2f}"
        row += f"{r['val_flow_mse_raw']:>10.4f}{by[steps[0]] / by[steps[-1]]:>14.2f}x"
        print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default="reports/flow_eval_matched.json")
    p.add_argument("--labels", nargs="*", default=None, help="Display names, in --runs order.")
    p.add_argument("--rfid", nargs="*", type=float, default=None,
                   help="Each run's autoencoder rFID, in the same order.")
    p.add_argument("--out", default="reports/flow_eval_matched")
    args = p.parse_args()

    data = load(args.json)
    labels = args.labels or None
    rfid = args.rfid or None
    if labels and len(labels) != len(data["runs"]):
        raise SystemExit(f"--labels has {len(labels)} entries, JSON has {len(data['runs'])} runs.")
    if rfid and len(rfid) != len(data["runs"]):
        raise SystemExit(f"--rfid has {len(rfid)} entries, JSON has {len(data['runs'])} runs.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(data, labels, rfid, args.out)
    make_table(data, labels, rfid, args.out)
    make_t_table(data, labels, args.out)
    print_summary(data, labels, rfid)
    print_t_summary(data, labels)


if __name__ == "__main__":
    main()
