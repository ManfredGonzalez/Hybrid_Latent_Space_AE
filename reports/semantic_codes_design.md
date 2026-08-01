# Semantic codes for DUALVAE via cross-view code-invariance (CVI)

**Goal.** Make the DUALVAE discrete codes *augmentation-invariant* (semantic content) so the
bag-of-codes becomes a better linear-probe feature — **without touching the reconstruction path**
(so FID/LPIPS are unaffected). This turns the dual-branch GMM into a content/style disentangler:
the **codes carry invariant content**, the **continuous/wavelet branch absorbs the style/nuisance**.

This is a lightweight auxiliary head, not a new model. It reuses the existing encoder, codebook,
and (conceptually) the SwAV assignment code already in `models/dual_ssl.py`.

---

## 1. Principle

- Linear probing rewards **invariance**; reconstruction rewards **preserving appearance**. Add a
  small, targeted invariance pressure to the *codes only*, and let the continuous branch keep the
  pixels — reconstruction still works, but the codes become semantic.
- **Reconstruction gives the anti-collapse for free.** Pure SSL needs Sinkhorn/negatives to stop
  the assignment collapsing to one prototype. Here the reconstruction loss already forces the codes
  to be *informative* (they must reconstruct the image), so a plain symmetric matching loss cannot
  collapse them — the CVI term only *reshapes* an already-informative code distribution to be
  augmentation-consistent. (Sinkhorn is available as a safety valve; see §7.)
- We match the **image-level soft bag-of-codes** (a distribution over the codebook, averaged over
  locations), which is **position-invariant** — so crops/flips don't misalign it, and it is exactly
  the feature that probed best.

---

## 2. The method (minimal, FID-safe version)

Per training step, on a clean batch `x`:

1. **Reconstruction — unchanged.** `recon, vq, van = model(x)`; `dualvae_loss(...)` exactly as now.
   (Clean → clean, so the FID path is untouched.)
2. **One augmented encode.** `x_aug = augment(x)` with **appearance** transforms (color jitter,
   grayscale, blur, moderate crop, flip). Encode only (no decode).
3. **Code-invariance loss.** Pull the soft bag-of-codes of `x` and `x_aug` together:
   `L = L_recon + λ_inv · L_CVI`.

Cost: **+1 encoder pass** (the augmented view). No extra decode, no extra GAN pass.

```
                 ┌── decode ── L_recon (UNCHANGED, clean→clean, FID-safe)
  x ── encoder ──┤ z_e_vq(x) ─────────────► soft hist h_clean ┐
                 └── (codes carry content)                    ├── L_CVI  (pull together)
  x_aug ── encoder ── z_e_vq(x_aug) ──────► soft hist h_aug   ┘
```

---

## 3. The loss (math + code)

Soft per-location assignment to the codebook `E ∈ R^{K×C}` (depth-1 code = the "component
identity"), then average over locations to an image-level distribution:

$$a_{b,\ell,k} = \mathrm{softmax}_k\!\big(-\lVert z^{vq}_{b,\ell}-e_k\rVert^2/\tau\big),\qquad
  h_b = \tfrac{1}{HW}\sum_\ell a_{b,\ell}\in\Delta^{K-1}.$$

CVI = symmetric cross-entropy between the clean and augmented histograms (stop-grad targets):

$$\mathcal{L}_{\text{CVI}}=\tfrac12\Big(H(\mathrm{sg}(h^{aug}),\,h^{clean})+H(\mathrm{sg}(h^{clean}),\,h^{aug})\Big).$$

```python
# --- model helpers (add to models/dual_vae.py) ---
import torch.nn.functional as F

def code_histogram(self, z_e_vq, tau=0.5):
    """(B,C,h,w) -> (B,K) image-level soft bag-of-codes over the depth-1 codebook."""
    b, c, h, w = z_e_vq.shape
    zf = z_e_vq.permute(0, 2, 3, 1).reshape(b * h * w, c)
    cb = self.vq_layer.embedding.weight                      # (K, C)
    if self.vq_layer.l2_normalize:                           # cosine, matches VQ lookup
        logits = (F.normalize(zf, dim=-1) @ F.normalize(cb, dim=-1).t()) / tau
    else:                                                    # -||.||^2 / tau
        d2 = (zf ** 2).sum(-1, keepdim=True) + (cb ** 2).sum(-1) - 2 * zf @ cb.t()
        logits = -d2 / tau
    a = F.softmax(logits, dim=-1).reshape(b, h * w, -1)
    return a.mean(dim=1)                                     # (B, K)

def encode_zevq(self, x):
    """Encoder + VQ bottleneck only (for the augmented view; no decode)."""
    return self.bottle_neck_VQ(self.encoder(x))             # (B, C, h, w)

# --- loss (in train script) ---
def code_invariance_loss(h_clean, h_aug, eps=1e-8):
    lc, la = torch.log(h_clean + eps), torch.log(h_aug + eps)
    return 0.5 * (-(h_aug.detach() * lc).sum(1).mean()
                  - (h_clean.detach() * la).sum(1).mean())
```

Also expose `z_e_vq` from `forward()` so the clean histogram is free (no re-encode): add
`"z_e_vq": z_e_vq` to the returned `vq_related_losses` dict (backward-compatible extra key).

> **RQ note.** With `rq_depth=2`, `z_e_vq` is the first residual, so the softmax over the codebook
> *is* the depth-1 (coarse) assignment — the most semantic level. Good default. (A `use_depth1`
> flag is kept for symmetry but depth-1 is what we want.)

---

## 4. Training-loop integration (`experiments/train_dualvae.py`)

Inside `train_one_epoch`, after the existing `loss` is computed and **before** `loss.backward()`:

```python
lam = code_invariance_weight(args, epoch)          # ramped schedule, §6
if lam > 0.0:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        x_aug = gpu_augment(images, args)          # appearance augs on GPU, §5
        h_clean = model.code_histogram(vq_related_losses["z_e_vq"].float(),
                                       tau=getattr(args, "code_invariance_tau", 0.5))
        h_aug   = model.code_histogram(model.encode_zevq(x_aug).float(),
                                       tau=getattr(args, "code_invariance_tau", 0.5))
    ci = code_invariance_loss(h_clean, h_aug)
    loss = loss + lam * ci
    running["code_invariance"] += ci.item()        # -> wandb Train/Code Invariance
```

No change to `dualvae_loss`, the GAN step, or validation. Everything else is untouched.

---

## 5. Augmentations (GPU, appearance-focused)

The codes should be invariant to **appearance/nuisance**, so weight toward color/blur/grayscale
plus a *moderate* crop (aggressive SSL crops break the "same content" assumption and aren't needed
since we don't decode this view). Batched on GPU via **kornia** (`pip install kornia`) or
`torchvision.transforms.v2`:

```python
import kornia.augmentation as Kaug
def build_ci_aug(args, size):
    return torch.nn.Sequential(
        Kaug.RandomResizedCrop((size, size), scale=(getattr(args,'ci_crop_scale_min',0.5), 1.0)),
        Kaug.RandomHorizontalFlip(),
        Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        Kaug.RandomGrayscale(p=getattr(args,'ci_grayscale_p',0.2)),
        Kaug.RandomGaussianBlur((23, 23), (0.1, 2.0), p=getattr(args,'ci_blur_p',0.5)),
    )
# images are Normalize(0.5,0.5,0.5); augment in [0,1] then renormalize:
def gpu_augment(images, args):
    x01 = (images * 0.5 + 0.5).clamp(0, 1)
    x01 = _CI_AUG(x01)                     # module built once, moved to device
    return (x01 - 0.5) / 0.5
```

---

## 6. Schedule (let the codebook settle first)

Ramp `λ_inv` from 0 so the k-means-seeded EMA codebook stabilizes before invariance pressure
(same rationale as `gan_start_epoch`):

```python
def code_invariance_weight(args, epoch):
    w = getattr(args, "code_invariance_weight", 0.5)
    s = getattr(args, "code_invariance_start_epoch", 8)
    r = max(1, getattr(args, "code_invariance_ramp_epochs", 5))
    return w * min(max((epoch - s) / r, 0.0), 1.0)
```

---

## 7. Config flags (drop into the C+/D+/wavelet yaml)

```yaml
# --- Cross-view code-invariance (semantic codes; default OFF) ---
code_invariance: true
code_invariance_weight: 0.5        # λ_inv; keep small so it nudges, not dominates. TUNE 0.1–1.0.
code_invariance_tau: 0.5           # soft-assignment temperature (lower = sharper)
code_invariance_start_epoch: 8     # ramp start (after codebook/EMA settle)
code_invariance_ramp_epochs: 5
code_invariance_use_depth1: true   # match on the coarse (component-identity) code
# augmentation strength (appearance-focused)
ci_crop_scale_min: 0.5
ci_grayscale_p: 0.2
ci_blur_p: 0.5
```

Gate everything on `getattr(args, "code_invariance", False)` so existing runs are unaffected.

---

## 8. Evaluation plan (the whole point is measurable)

Train the **same** recipe (e.g. D+ or D++) **with vs without** CVI, then:

1. **Probe (target metric):** `tools/linear_probe_retrieval.py` — does **bag-of-codes accuracy**
   rise above the ~0.37 ceiling? (Also `z_vq` accuracy and retrieval mAP.)
2. **FID unchanged?** `tools/reconstruction_ablation.py` (rFID) — should stay ≈ the CVI-off run,
   since the reconstruction path is untouched. A big FID regression means `λ_inv` is too high.
3. **No code collapse:** watch `Codebook/Perplexity`, `Codebook/Usage`, `Sigma2` in wandb — the
   effective #components should stay high (reconstruction anchors it).
4. **Mixture intact:** `tools/dualvae_latent_analysis.py` — separability/silhouette shouldn't drop.

**Success = probe up, FID flat, perplexity flat.**

---

## 9. Risks and mitigations

| Risk | Why | Mitigation |
|---|---|---|
| Code collapse (all → 1 code) | invariance can trivialize codes | reconstruction anchors informativeness; small `λ_inv`; ramp; watch perplexity; §7 Sinkhorn valve |
| FID regression | `λ_inv` too high perturbs the encoder | keep `λ_inv` small, ramp late; reconstruction path is otherwise untouched |
| Crop breaks "same content" | different crops show different objects | appearance-weighted augs + *moderate* crop (scale ≥ 0.5); histogram is position-invariant |
| No semantic gain | augs too weak / codes already saturated | ensure color+blur+grayscale on; try lower `tau`; probe depth-1 vs summed codes |

**Sinkhorn safety valve.** If perplexity sags, replace the raw target `sg(h_aug)` with a
Sinkhorn-balanced version (reuse `DualSSL.sinkhorn` from `models/dual_ssl.py`) so usage stays
balanced — turns CVI into a SwAV swapped-prediction on the codes.

---

## 10. Variants (if the minimal version underperforms)

- **Two decoded views (symmetric).** Reconstruct *both* views and match codes both ways — stronger
  invariance signal, ~2× cost, small FID risk from reconstructing augmented views.
- **Dense per-location match.** If you disable cropping (aligned views), match per-location codes
  instead of the histogram — a stronger, spatially-precise signal.
- **Masked-code prediction (BEiT/iBOT-style).** Mask input patches, predict the *codes* of the
  masked patches from context. No augmentation-invariance assumption; reuses the codebook as the
  MIM target. Heavier change but a very strong semantic prior.
- **Teacher distillation.** Align `h_clean` (or the pooled feature) to a frozen DINO/CLIP feature —
  the biggest probe gain, but imports external supervision (changes the "reconstruction-only" claim).

---

## 11. Why this is a clean story

A reconstruction VQ-GMM whose **codes are made semantic by a lightweight cross-view invariance
head**, with the **continuous/wavelet branch absorbing style** and the **reconstruction objective
supplying the anti-collapse** that pure SSL needs Sinkhorn for. It's the content/style reading of
your dual branch, realized with ~30 lines and no change to the reconstruction/FID path — and it
connects directly to your SwAV/`dual_ssl.py` work (the same assignment matching, used as an
auxiliary term instead of the whole objective).

---

## 12. Post-mortem + fix: the pooled loss was flat at ln(K); go per-location + me-max

**What happened (run `dualvae_20260728-211259_bb2818`, pooled + Sinkhorn, weight 0.1).**
The CVI loss sat pinned at **5.55 ≈ ln(256)=5.545** and was flat all training, even though the
*effective* weight (ratio-anchor) was **200–600**. Flat loss + huge weight = **no gradient**, not
under-weighting. Downstream barely moved (bagcodes probe 0.406→0.443 vs D+ 0.438).

**Two root causes.**
1. **Image-pooling blurs the target.** Averaging soft assignments over all 32×32 locations pushes
   each image's histogram toward uniform, so the two views trivially match — nothing to learn.
2. **Sinkhorn flattens the target to equipartition**, deleting the per-image structure CVI needs.
   The anti-collapse valve and the anti-signal were the same knob. CE between two near-uniform
   distributions is ln(K), by definition — exactly what we observed.

**The fix (implemented; `code_invariance_mode: aligned`).**
- **Per-location match** (`DUALVAE.code_soft_assign` → `(B,HW,K)`): compare *which code each patch
  picked* between the clean view and a **photometric-only** augmented view (no crop/flip, so grids
  align). Real gradient (verified: grad-norm ~0.27 with a settled codebook vs ~0 pooled).
- **me-max anti-collapse** replaces Sinkhorn: penalize `ln K − H(batch-marginal usage)` — keeps all
  codes used **without** flattening any single patch. Target may be sharpened (`target_tau=0.25`).
- **Monitoring** (wandb): `Train/CVI Agreement` (↑ toward 1 = same code both views),
  `Train/CVI Patch Entropy` (↓ = confident patches), `Train/CVI Marginal Entropy` (~5.5 = no
  collapse). If Agreement climbs while Marginal Entropy holds ~5.5, CVI is finally biting; then
  raising `code_invariance_weight` (0.1→0.2→0.5→1.0) is meaningful.

Code: `models/dual_vae.py::code_soft_assign`, `experiments/train_dualvae.py::code_invariance_aligned`
(+ aligned branch in `build_ci_augment`/`train_one_epoch`). Config
`configs/dualvae_code_prior_gan_cvi_aligned_D.yaml`. Legacy pooled path kept, gated (default `pooled`).

---

## 13. Switching families: masked latent modeling (two experiments, replaces CVI)

CVI kept failing because it's in the invariance/clustering family (all signal from augmentation +
an anti-collapse crutch that fights the codebook). We switched to the **masked/generative family**
(BEiT/iBOT/data2vec/MaskGIT): predict HIDDEN latent locations from their visible neighbours.
Augmentation-free, grounded (no collapse shortcut -> no Sinkhorn/me-max), and it *uses* the codebook
as the tokenizer instead of policing it. Reconstruction/GAN path untouched; pure aux head; reuses the
main forward's `z_e_vq` (no extra encoder pass). Two experiments share one predictor
(`models/masked_predictor.py`, a small ViT over the 32x32 grid with a learned mask token):

- **Experiment A -- Masked Code Modeling (MCM),** `configs/dualvae_msm_code_D.yaml`.
  Predict the GMM soft assignment (responsibilities) of hidden patches; cross-entropy at masked
  locations. Makes codes context-predictable = semantic. Watch **Train/MSM Accuracy** (rises).

- **Experiment B -- Masked Latent GMM Modeling (MLM),** `configs/dualvae_msm_latent_D.yaml`.
  Predictor outputs context mixing weights `r_k(ctx)`; loss = NLL of the TRUE hidden `z_e_vq` under
  `p(z|ctx)=sum_k r_k(ctx) N(e_k, sigma_k^2 I)` -- a *conditional* version of the code-centered GMM
  prior. So the head is a **native generative prior**: sample `k~r_k`, `z~N(e_k,sigma_k^2)`, fill the
  grid MaskGIT-style, decode -> generate. Watch **Train/MSM Bits-Per-Dim** (falls) and **Accuracy**.
  Realises "reconstructs better than RAE AND carries a GMM latent AND has a native generative prior."

Both use a **bounded** effective weight (`msm_balance` ratio-anchor + `msm_max_scale=50` cap) -- the
lesson from CVI's runaway 200-600 weight. Means/vars/weights come straight from the codebook
(`vq_layer.embedding.weight`, `.sigma2`, `.pi`). Predictor saved as `masked_predictor_{best,last,final}.pt`.
Gated on `masked_modeling` (default False). Verified: gradients reach the encoder + predictor, both
losses/optimizers step, legacy path returns (None,None). **Next:** a MaskGIT sampler tool to draw from
the Experiment-B prior once a checkpoint trains, + downstream (bag-of-codes vs D+) for both.
