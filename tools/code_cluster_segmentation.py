"""One combined code-cluster-segmentation figure across the six DUALVAE runs that span the
standard-vs-code prior x MSE-vs-GAN design:

    A   (std prior, beta=.001, MSE)      A+  (std prior, beta=.001, LPIPS+GAN)
    B   (std prior, beta=.1,   MSE)      B+  (std prior, beta=.1,   LPIPS+GAN)
                                         C+  (code prior, beta=.1,  LPIPS+GAN)
                                         D+  (code prior, beta=.001, LPIPS+GAN)

Every patch of the 32x32 latent grid is painted with the colour of the *super-cluster* its
codeword belongs to: the 256 codewords are grouped by k-means into K=10 groups (one per
Imagenette class, as a neutral choice) and each group gets a categorical colour. Grouping is
what makes the picture readable -- 256 raw code colours look like noise -- and it is exactly
the question the mixture claim poses: do nearby codewords (= nearby mixture components) land
on the same *stuff* in the image?

Colours are per model (the codebooks and their k-means differ), so compare *within-image
region structure*, never colours across columns. Per-panel numbers:
  codes   = distinct codewords used in that image (lower = less fragmented)
  coh.    = super-cluster neighbour agreement (fraction of adjacent patch pairs in the same
            group; higher = more spatially coherent regions)
An aggregate over 60 val images is printed and written to JSON for the report table.

Run:  python tools/code_cluster_segmentation.py
Outputs: reports/code_cluster_seg_all_runs.{pdf,json}  (Fig. "Code-cluster segmentation across
the prior x objective grid" and its statistics table in reports/latent_analysis_report.tex).
"""
import os, sys, json, gc
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tools.dualvae_latent_analysis import (load_config, find_weights, build_dualvae,
                                           build_transform, sample_image_paths)

MODELS = [
    ("A\n(std, $\\beta$=.001, MSE)",   "checkpoints/dualvae/dualvae_20260724-121508_392b5b"),
    ("B\n(std, $\\beta$=.1, MSE)",     "checkpoints/dualvae/dualvae_20260724-133147_b03f09"),
    ("A+\n(std, $\\beta$=.001, GAN)",  "checkpoints/dualvae/dualvae_20260801-201243_0b2ee6"),
    ("B+\n(std, $\\beta$=.1, GAN)",    "checkpoints/dualvae/dualvae_20260801-201309_1e27ab"),
    # E+ extends the standard-prior beta ladder one decade past B+ (.001 -> .1 -> 1.0). It is
    # the strongest KL pressure in the grid, so it is the run that tests whether "more KL"
    # eventually buys the mixture that the standard prior never produced at beta <= .1.
    ("E+\n(std, $\\beta$=1.0, GAN)",   "checkpoints/dualvae/dualvae_20260802-163932_ca522b"),
    ("C+\n(code, $\\beta$=.1, GAN)",   "checkpoints/dualvae/dualvae_20260726-230719_5e588d"),
    ("D+\n(code, $\\beta$=.001, GAN)", "checkpoints/dualvae/dualvae_20260726-230704_866e33"),
]
DATASET = "/media/tico/BACKUP-DIDI/imagenette/imagenette2-320"
N_IMAGES, N_AGG, SEED, ALPHA, K = 4, 60, 7, 0.78, 10
OUT_PDF = "reports/code_cluster_seg_all_runs.pdf"
OUT_JSON = "reports/code_cluster_seg_all_runs.json"
dev = "cuda" if torch.cuda.is_available() else "cpu"
PALETTE = np.array(plt.get_cmap("tab10").colors)


def load(dir_):
    cfg, _ = load_config(dir_)
    m = build_dualvae(cfg, dev)
    sd = torch.load(find_weights(dir_), map_location=dev)
    m.load_state_dict(sd if "encoder.0.weight" in sd else sd.get("model_state_dict", sd))
    m.eval()
    cb = m.vq_layer.embedding.weight.detach().float().cpu().numpy()
    # k-means groups the 256 codewords into K super-clusters; order the groups along the
    # codebook's 1st PC so the colour assignment is at least deterministic per model.
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(cb)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(km.cluster_centers_).ravel()
    remap = np.empty(K, dtype=int); remap[np.argsort(pc1)] = np.arange(K)
    return m, cfg, remap[km.labels_]


@torch.no_grad()
def codes_of(m, img):
    z_e_vq = m.bottle_neck_VQ(m.encoder(img))
    _, _, code_idx, _, _ = m.vq_layer(z_e_vq)
    g = img.shape[-1] // m.downsample_factor
    return code_idx.view(g, g).cpu().numpy()


def coherence(C):  # fraction of 4-neighbour patch pairs with the same label
    h = (C[:, :-1] == C[:, 1:]).sum(); v = (C[:-1, :] == C[1:, :]).sum()
    tot = C.shape[0] * (C.shape[1] - 1) + (C.shape[0] - 1) * C.shape[1]
    return (h + v) / tot


cfg0, _ = load_config(MODELS[0][1])
tf = build_transform(cfg0.get("resize_img", 256), cfg0.get("dataset_name", "imagenette"))
paths_agg = sample_image_paths(os.path.join(DATASET, "val"), N_AGG, SEED)
paths = paths_agg[:N_IMAGES]


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


segs, panel, agg = {}, {}, {}
origs, grays = [], []
for p in paths:
    img = tf(Image.open(p).convert("RGB")).unsqueeze(0)
    o = denorm(img)[0].permute(1, 2, 0).numpy()
    origs.append(o)
    grays.append(np.repeat((o @ [0.299, 0.587, 0.114])[:, :, None], 3, axis=2))

for name, d in MODELS:
    m, cfg, group = load(d)
    ds = m.downsample_factor
    segs[name], panel[name] = [], []
    with torch.no_grad():
        for p in paths:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            C = codes_of(m, img); G = group[C]
            segs[name].append(np.kron(PALETTE[G], np.ones((ds, ds, 1))))
            panel[name].append((int(len(np.unique(C))), float(coherence(G))))
        nc, ch, chc, ng = [], [], [], []
        for p in paths_agg:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            C = codes_of(m, img); G = group[C]
            nc.append(len(np.unique(C))); ng.append(len(np.unique(G)))
            ch.append(coherence(C)); chc.append(coherence(G))
    agg[name] = {"codes_per_img": float(np.mean(nc)),
                 "groups_per_img": float(np.mean(ng)),
                 "code_coherence": float(np.mean(ch)),
                 "cluster_coherence": float(np.mean(chc)),
                 "checkpoint": d}
    print(f"[done] {name.splitlines()[0]:4} codes/img={np.mean(nc):5.1f}  "
          f"code-coh={np.mean(ch)*100:4.1f}%  cluster-coh={np.mean(chc)*100:4.1f}%", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

ncol = 1 + len(MODELS)
fig, ax = plt.subplots(N_IMAGES, ncol, figsize=(ncol * 2.35, N_IMAGES * 2.55))
for r in range(N_IMAGES):
    ax[r, 0].imshow(origs[r]); ax[r, 0].axis("off")
    for c, (name, _) in enumerate(MODELS, start=1):
        ax[r, c].imshow((1 - ALPHA) * grays[r] + ALPHA * segs[name][r]); ax[r, c].axis("off")
        n, co = panel[name][r]
        ax[r, c].set_title(f"{n} codes | {co*100:.0f}% coh.", fontsize=7, pad=2)
ax[0, 0].set_title("original", fontsize=10, fontweight="bold", pad=10)
for c, (name, _) in enumerate(MODELS, start=1):
    a = agg[name]
    ax[0, c].set_title(f"{name}\n{a['codes_per_img']:.0f} codes/img, "
                       f"{a['cluster_coherence']*100:.0f}% coh.",
                       fontsize=9, fontweight="bold", pad=6)
fig.tight_layout()
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", dpi=220)
fig.savefig(OUT_PDF.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=140)
print("[saved]", OUT_PDF)

json.dump({"n_agg_images": N_AGG, "seed": SEED, "n_super_clusters": K, "runs": agg},
          open(OUT_JSON, "w"), indent=2)
print("[saved]", OUT_JSON)
print(f"\n=== aggregate over {N_AGG} val images (K={K} super-clusters) ===")
for name in agg:
    a = agg[name]
    print(f"  {name.splitlines()[0]:4}  codes/img = {a['codes_per_img']:5.1f}   "
          f"code-coh = {a['code_coherence']*100:4.1f}%   "
          f"cluster-coh = {a['cluster_coherence']*100:4.1f}%   "
          f"groups/img = {a['groups_per_img']:.1f}")
