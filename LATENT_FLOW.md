# Class-conditional latent flow matching — DUALVAE vs. vanilla VAE

Trains the **same** conditional generator in two different frozen latent spaces and compares
generative quality. The hypothesis under test: the hybrid discrete+continuous DUALVAE latent
space is a better space to generate in than a vanilla VAE's.

| | latent space | config | job |
|---|---|---|---|
| A | `checkpoints/dualvae/dualvae_20260726-230704_866e33` | `configs/flow_dualvae.yaml` | `train_flow_dualvae.slurm` |
| B | `checkpoints/vae/vae_20260729-110353_138ede` | `configs/flow_vae.yaml` | `train_flow_vae.slurm` |

`ae_checkpoint` names the **run directory**; `resolve_ae_checkpoint` picks `final_epoch.pt`
(last epoch) → `last.pt` → `best.pt`, and raises with the directory listing if none exist.
Last-epoch weights come first deliberately: the autoencoder trainers select `best.pt` on the
**train** loss (`save_if_best_val`), which stops being a meaningful signal once the GAN is
active, whereas the last epoch is the one whose codebook EMA statistics, discriminator balance
and LR schedule all finished. Name a specific `.pt` in the config to override. The resolved
path is what gets recorded in the latent cache key, the flow checkpoint and wandb — so you can
always tell which weights a run actually used.

The two configs differ in exactly three lines (`ae_checkpoint`, `checkpoints`, `run_prefix`).
Everything about the generator — UNet size, objective, batch size, LR schedule, epoch budget,
sampler, guidance scale, FID protocol — is identical, so a gap in `Gen/FID` is attributable to
the latent space rather than to the generator.

## Run

```bash
python main.py --model latent_flow --config ./configs/flow_dualvae.yaml
python main.py --model latent_flow --config ./configs/flow_vae.yaml
# or: sbatch train_flow_dualvae.slurm ; sbatch train_flow_vae.slurm   (different nodes)
```

## Where the labels come from

Class-conditional training needs real labels, and the autoencoder pipeline never had them:
`FlatImageDataset` returns a constant `label = 0`. `LabeledImageDataset` resolves the class
from the most trustworthy source available:

1. **Nested directories** — `train/<wnid>/xxx.JPEG`; class = subdirectory.
2. **`noisy_imagenette.csv`** — fastai's label file, which ships in the imagenette2 tarball.
   Auto-discovered in the split directory or its parent, keyed by image basename. Column
   `noisy_labels_0` is the clean ground truth (the `noisy_labels_1/5/25/50` columns are
   deliberately corrupted variants — `label_column` selects them if you ever want them).
3. **Filename prefix** — `n01440764_10026.JPEG` → `n01440764`.

**The CSV outranks the filename prefix, and that matters here.** imagenette2 is not uniformly
named: about 4% of the images in *both* splits (366 of 9469 train, 134 of 3925 val) come from
ImageNet's validation set and are named `ILSVRC2012_val_00000293.JPEG`, which carries no class
at all. On a flattened tree, prefix parsing would invent a bogus `ILSVRC2012` "class" holding
~500 images of ten different real classes. The CSV covers every file exactly once, so it wins
whenever present. Verified on `imagenette2-320`: 9469/3925 images, 10 classes, and the
ILSVRC-named files land on their true labels (`ILSVRC2012_val_00000293.JPEG` → tench).

If neither the CSV nor a usable layout is found, it raises with an explanation rather than
silently training an unconditional model. Check on the server before launching:

```bash
python -c "from data.datasets import get_labeled_datasets as G; \
tr,va=G('imagenette', path='./datasets/imagenette2-320'); \
print(tr.layout, len(tr), len(va), len(tr.classes), tr.labels_csv)"
```

Want: `flat+csv 9469 3925 10 <path to noisy_imagenette.csv>`. If the CSV is missing on the
cluster, copy it next to `train/` and `val/`, or set `labels_csv:` in the config.

## What it does

1. Loads the autoencoder **frozen** (`tools/latent_ae.py`), rebuilding its architecture from the
   `config_used.yaml` sitting next to the checkpoint — no flags to keep in sync by hand.
2. Encodes the dataset **once** into `./latent_cache/` (`8 x 32 x 32` per image, fp16, plus the
   horizontally-mirrored encoding as free augmentation). The 256px encoder then never runs in
   the training loop.
3. Standardizes latents per channel with that autoencoder's **own** train-set statistics. The
   two spaces really are on different scales — measured on imagenette, the DUALVAE's per-channel
   std runs 0.83–1.49 while the VAE's runs 0.32–0.49 (it is pre-multiplied by 0.18215, and its
   channels are far more uniform). Flow matching couples the data to a standard normal, so
   without this the two runs would be solving differently-conditioned problems.
4. Trains `LatentFlowUNet` on the rectified-flow objective
   `L = E || v(x_t, t, y) - (x1 - x0) ||²`, `x_t = (1-t)·x0 + t·x1`, with label dropout for
   classifier-free guidance and an EMA copy of the weights for sampling.
5. Logs, per epoch: train/val flow loss; every 50 epochs a 10×4 class-conditional sample grid;
   every 100 epochs **Gen/FID** and **Gen/KID** of 1000 class-balanced samples against 2000 real
   validation images.

## Reading the results

* **`Gen/FID` (wandb, `latent_flow` project)** — the headline number. Compare the curves, not
  just the final value: reaching a given FID in fewer epochs is the "efficiency" claim.
* **`Val/Flow Loss`** — comparable *within* a run (epoch selection), **not across the two runs**:
  each is a loss in its own standardized latent space, so the two numbers are not on the same
  scale. Do not rank the latent spaces with it. (Concretely: after one epoch the DUALVAE run
  sits at 1.78 and the VAE run at 1.63 — that gap is latent geometry, not model quality.)
* **The rFID floor.** A latent generator can never beat its autoencoder's own reconstruction
  FID. If the two autoencoders' rFIDs differ, subtract that off before crediting the latent
  space — the fair statement is `Gen/FID - rFID`, the cost of generating in the space, as
  distinct from the cost of the space itself.
* **Guidance.** FID depends strongly on `guidance_scale`; a single scale can rank two models
  backwards. Before concluding, sweep it on both final checkpoints with `sample_latent_flow.py`.

## Sampling from a trained model

```bash
# grid, EMA weights
python sample_latent_flow.py --checkpoint ./checkpoints/flow_dualvae/<run>/best.pt \
    --n_per_class 8 --guidance_scale 2.0 --out ./samples/flow_dualvae

# generative FID/KID on more samples than the training loop can afford
python sample_latent_flow.py --checkpoint ./checkpoints/flow_dualvae/<run>/best.pt \
    --fid --n_samples 5000

# guidance sweep
for w in 1.0 1.5 2.0 3.0 4.0; do
  python sample_latent_flow.py --checkpoint <run>/best.pt --fid --n_samples 2000 \
      --guidance_scale $w --out ./samples/sweep_w$w
done

# identical noise in both latent spaces, for a side-by-side figure
python sample_latent_flow.py --checkpoint <dualvae_run>/best.pt --noise_seed 7 --out ./samples/A
python sample_latent_flow.py --checkpoint <vae_run>/best.pt     --noise_seed 7 --out ./samples/B
```

Checkpoints are self-contained (weights, EMA, latent statistics, class names, and the
autoencoder they belong to), so a run reproduces from its checkpoint path alone.

## Files

| file | what |
|---|---|
| `models/flow_unet.py` | `LatentFlowUNet` — ADM-style UNet velocity field, FiLM time+class conditioning, NULL class for CFG |
| `losses/flow_matching.py` | rectified-flow objective, timestep sampling, Euler/Heun ODE samplers with CFG |
| `tools/latent_ae.py` | frozen `encode`/`decode` over DUALVAE and VAE + latent standardization |
| `experiments/train_latent_flow.py` | the trainer (latent cache, EMA, sampling, generative FID) |
| `data/datasets.py` | `LabeledImageDataset`, `get_labeled_datasets`, `build_image_transform` |
| `sample_latent_flow.py` | sampling / FID / guidance sweeps from a checkpoint |
| `configs/flow_{dualvae,vae}.yaml` | the two runs |

## References for the design choices

* **Objective** — Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023
  (arXiv:2210.02747); Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer
  Data with Rectified Flow*, ICLR 2023 (arXiv:2209.03003).
* **Latent (two-stage) generation** — Rombach et al., *High-Resolution Image Synthesis with
  Latent Diffusion Models*, CVPR 2022 (arXiv:2112.10752).
* **Logit-normal timestep sampling** (`t_sampling: logit_normal`, m=0, s=1) — Esser et al.,
  *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024
  (arXiv:2403.03206), Sec. 3.1. The velocity target is trivial at both endpoints (the optimal
  prediction is the mean of p1 at t=0 and of p0 at t=1), so uniform t spends capacity where
  there is nothing to learn; their formulation sweep ranks `rf/lognorm(0.00, 1.00)` at the top.
  Set `t_sampling: uniform` for the baseline arm of that ablation.
* **Classifier-free guidance** (`cfg_dropout_prob`, `guidance_scale`) — Ho & Salimans,
  *Classifier-Free Diffusion Guidance*, NeurIPS 2021 Workshop (arXiv:2207.12598).
* **UNet architecture** (FiLM time/class conditioning, attention at the low-resolution levels)
  — Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021
  (arXiv:2105.05233).
* **EMA of the weights** — Song & Ermon, *Improved Techniques for Training Score-Based
  Generative Models*, NeurIPS 2020 (arXiv:2006.09011), Technique 5: without EMA, FID
  oscillates substantially between nearby checkpoints; with EMA it is stable and usually
  better. Theoretical root: Polyak & Juditsky 1992 (iterate averaging). Note this evidence is
  from score-based diffusion — the flow-matching papers use EMA but do not ablate it — and
  that Karras et al. 2024 (EDM2, arXiv:2312.02696) show FID depends strongly and
  non-monotonically on EMA *length*, so `ema_decay: 0.9999` is a convention, not a tuned
  value. Worth sweeping if the two runs finish close.

## Knobs worth knowing

* `latent_sample: false` caches the posterior **mean**. On these checkpoints (`kl_beta = 0.001`)
  the posterior is nearly deterministic, so this costs essentially nothing; set
  `latent_sample: true` + `latent_cache: false` for LDM-style per-step posterior samples.
* `unet_channel_mults: [1,2,2,2]` at `base_channels: 128` is 39.6M parameters (~53 ms/step at
  batch 64 on a 4090, 3.6 GB). Scale both configs together or not at all.
* `sample_solver: heun` costs 2 network evaluations per step; report the step count alongside
  any FID.
* Delete `./latent_cache/*.pt` after retraining an autoencoder — the cache key includes the
  autoencoder's run id and checkpoint filename, but not its mtime.
