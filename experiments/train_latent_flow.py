"""Class-conditional latent flow matching on top of a FROZEN autoencoder.

The experiment this file exists for: take two autoencoders trained on the same data with the
same reconstruction objective -- the DUALVAE (discrete codes + continuous offset) and the
vanilla VAE baseline -- and train the IDENTICAL conditional generator in each one's latent
space. Whichever latent space reaches a better generative FID (and reaches it in fewer
steps) is the better space for generation. Nothing here is model-specific: the autoencoder
is loaded through tools/latent_ae.py, which hides the wiring differences behind
encode()/decode().

Protocol notes that matter for the comparison being fair:
  * The autoencoder is frozen and in eval() for the whole run.
  * Latents are standardized with statistics estimated from THAT autoencoder's own training
    latents (per-channel by default). The two latent spaces have different natural scales;
    without this the flow model would be solving a differently-conditioned problem in each.
  * Both runs must use the same UNet config, batch size, LR schedule, epoch budget, sampler
    and guidance scale. configs/flow_dualvae.yaml and configs/flow_vae.yaml differ ONLY in
    ae_checkpoint / checkpoints / wandb naming.
  * Generative FID is computed against the same real validation images in both runs.
"""

import copy
import os

import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.datasets import get_labeled_datasets
from losses.flow_matching import flow_matching_loss, sample_flow
from models.flow_unet import build_flow_model
from tools.latent_ae import LatentStats, load_frozen_ae
from tools.normalization import denormalize
from tools.utils import (build_lr_scheduler, create_directory, make_run_id, save_config_copy,
                         seed_worker, select_device, set_seed, setup_wandb)


# ---------------------------------------------------------------------------------------
# Latent cache
# ---------------------------------------------------------------------------------------

class LatentTensorDataset(Dataset):
    """Pre-encoded latents + labels held in memory.

    Encoding the whole dataset once and reusing it every epoch removes the frozen encoder
    from the training loop entirely -- at 256px that encoder costs more per step than the
    latent UNet does, so caching is most of the difference between a day's training and
    three days'. Latents are stored fp16 (they are the input to a bf16 network anyway) and
    cast to fp32 on read.

    With flip augmentation the cache holds TWO latents per image: the image's and its
    horizontally mirrored version's, each encoded properly through the AE (mirroring the
    latent directly is not the same operation). One of the two is drawn at random per read.
    """

    def __init__(self, latents, labels, flipped=None):
        self.latents = latents            # (N, C, H, W) fp16
        self.labels = labels              # (N,) int64
        self.flipped = flipped            # (N, C, H, W) fp16 or None

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        if self.flipped is not None and torch.rand(()) < 0.5:
            z = self.flipped[idx]
        else:
            z = self.latents[idx]
        return {"latent": z.float(), "label": self.labels[idx]}


def _cache_path(args, ae, split, flip):
    run_id = os.path.basename(os.path.dirname(os.path.abspath(ae.ae_checkpoint)))
    ckpt_tag = os.path.splitext(os.path.basename(ae.ae_checkpoint))[0]
    flip_tag = "_flip" if flip else ""
    name = f"{ae.kind}_{run_id}_{ckpt_tag}_{args.dataset_name}_{split}_{args.resize_img}{flip_tag}.pt"
    return os.path.join(getattr(args, "latent_cache_dir", "./latent_cache"), name)


@torch.no_grad()
def encode_split(ae, dataset, args, device, flip=False, desc="encode"):
    """Run the frozen encoder over a whole split -> (latents fp16, labels, flipped or None)."""
    loader = DataLoader(dataset, batch_size=getattr(args, "encode_batch_size", 32), shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=seed_worker)
    lat, flat, labels = [], [], []
    sample = getattr(args, "latent_sample", False)
    for batch in tqdm(loader, desc=desc, unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp and device == "cuda"):
            z = ae.encode(images, sample=sample)
            zf = ae.encode(torch.flip(images, dims=[3]), sample=sample) if flip else None
        lat.append(z.float().half().cpu())
        if flip:
            flat.append(zf.float().half().cpu())
        labels.append(batch["label"].long())
    return (torch.cat(lat), torch.cat(labels), torch.cat(flat) if flip else None)


def build_latent_datasets(args, ae, trainset, valset, device):
    """Encode (or load from cache) the train/val latents.

    The cache key includes the autoencoder run id and checkpoint file, so the DUALVAE and the
    VAE runs can never read each other's latents.
    """
    flip = getattr(args, "flip_augment", True)
    cache_dir = getattr(args, "latent_cache_dir", "./latent_cache")
    use_cache = getattr(args, "latent_cache", True)
    create_directory(cache_dir)

    out = {}
    for split, ds, do_flip in (("train", trainset, flip), ("val", valset, False)):
        path = _cache_path(args, ae, split, do_flip)
        if use_cache and os.path.exists(path):
            print(f"[latents] loading cache {path}")
            blob = torch.load(path, map_location="cpu")
            latents, labels, flipped = blob["latents"], blob["labels"], blob.get("flipped")
        else:
            latents, labels, flipped = encode_split(args=args, ae=ae, dataset=ds, device=device,
                                                    flip=do_flip, desc=f"encode {split}")
            if use_cache:
                torch.save({"latents": latents, "labels": labels, "flipped": flipped}, path)
                print(f"[latents] wrote cache {path}")
        out[split] = LatentTensorDataset(latents, labels, flipped)
        print(f"[latents] {split}: {tuple(latents.shape)} (flip_augment={flipped is not None})")
    return out["train"], out["val"]


# ---------------------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of the flow weights -- sampled from instead of the raw weights.

    Justification: Song & Ermon 2020 ("Improved Techniques for Training Score-Based Generative
    Models", NeurIPS, arXiv:2006.09011) make this Technique 5 and show it directly -- without
    EMA, FID and sample quality oscillate substantially between nearby checkpoints; with EMA
    (momentum ~0.999) the curves are stable and usually better. The underlying reason is
    generic to SGD on a high-variance stochastic target (Polyak & Juditsky 1992, iterate
    averaging), which is why it carries over from score matching to flow matching unchanged --
    the flow-matching papers (Lipman et al. 2023; Esser et al. 2024) all use EMA, though
    neither ablates it.

    CAVEAT on the decay: Karras et al. 2024 (EDM2, arXiv:2312.02696) show FID depends strongly
    and non-monotonically on EMA LENGTH, with an optimum that shifts with model size and
    training duration. 0.9999 here is a convention, not a tuned value -- if the two latent
    spaces finish close on Gen/FID, sweep this before concluding anything.

    The warmup term keeps the average from being dominated by the random initialization
    during the first steps.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.step_count = 0

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ---------------------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------------------

def train_step(model, loader, optimizer, stats, args, device, ema=None, epoch=0):
    model.train()
    total, n = 0.0, 0
    with tqdm(total=len(loader.dataset), desc=f"Epoch {epoch}/{args.epochs}", unit="lat") as pbar:
        for batch in loader:
            z = batch["latent"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            z = stats.normalize(z)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.use_amp and device == "cuda"):
                loss, _ = flow_matching_loss(
                    model, z, y,
                    t_sampling=getattr(args, "t_sampling", "logit_normal"),
                    logit_normal_mean=getattr(args, "logit_normal_mean", 0.0),
                    logit_normal_std=getattr(args, "logit_normal_std", 1.0),
                    cfg_dropout_prob=getattr(args, "cfg_dropout_prob", 0.1),
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       getattr(args, "grad_clip", 1.0))
            optimizer.step()
            if ema is not None:
                ema.update(model)

            total += loss.item()
            n += 1
            pbar.set_postfix(loss=loss.item(), gnorm=float(grad_norm))
            pbar.update(z.size(0))
    return {"loss": total / max(n, 1)}


@torch.no_grad()
def validation_step(model, loader, stats, args, device, num_repeats=4):
    """Held-out flow-matching loss.

    The Monte-Carlo estimate is noisy (t and the noise are resampled every time), so the val
    pass is repeated with a FIXED seed: the number is then comparable epoch-to-epoch and,
    more importantly, run-to-run -- but note it is a loss in each model's OWN standardized
    latent space, so it ranks epochs within a run, not the two latent spaces against each
    other. Use generative FID for that.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    total, n = 0.0, 0
    for _ in range(num_repeats):
        for batch in loader:
            z = stats.normalize(batch["latent"].to(device))
            y = batch["label"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.use_amp and device == "cuda"):
                loss, _ = flow_matching_loss(
                    model, z, y,
                    t_sampling=getattr(args, "t_sampling", "logit_normal"),
                    logit_normal_mean=getattr(args, "logit_normal_mean", 0.0),
                    logit_normal_std=getattr(args, "logit_normal_std", 1.0),
                    cfg_dropout_prob=0.0,   # evaluate the conditional model
                    generator=gen,
                )
            total += loss.item()
            n += 1
    return {"loss": total / max(n, 1)}


# ---------------------------------------------------------------------------------------
# Sampling / generative metrics
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def generate_images(model, ae, stats, args, device, labels, seed=None, batch_size=None):
    """labels: (N,) int64 -> (N, 3, H, W) images in [0, 1]."""
    batch_size = batch_size or getattr(args, "sample_batch_size", 16)
    c, h, w = ae.latent_shape(args.resize_img)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = []
    for i in range(0, labels.shape[0], batch_size):
        y = labels[i:i + batch_size].to(device)
        z = sample_flow(
            model, (y.shape[0], c, h, w), y,
            num_steps=getattr(args, "sample_steps", 50),
            guidance_scale=getattr(args, "guidance_scale", 2.0),
            device=device, solver=getattr(args, "sample_solver", "heun"), generator=gen,
        )
        images = ae.decode(stats.denormalize(z).float())
        images = denormalize(images.float(), args.dataset_name, device).clamp(0, 1)
        out.append(images.cpu())
    return torch.cat(out)


@torch.no_grad()
def sample_grid(model, ae, stats, args, device, num_classes, per_class=None):
    """One grid with `per_class` samples for every class (rows = classes), fixed seed so the
    grid tracks training progress rather than noise luck."""
    per_class = per_class or getattr(args, "sample_grid_per_class", 4)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    images = generate_images(model, ae, stats, args, device, labels,
                             seed=getattr(args, "sample_seed", 1234))
    return vutils.make_grid(images, nrow=per_class, normalize=False)


@torch.no_grad()
def generative_fid(model, ae, stats, args, device, valset_images_loader, num_classes):
    """FID/KID between generated samples and the real validation images.

    This is the headline number for the comparison. Both runs generate the same number of
    samples with the same class balance, the same solver, the same step count and the same
    guidance scale, and score against the same reals.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        print("WARNING: gen_fid requested but torchmetrics is unavailable - skipping.")
        return {}

    fid_device = getattr(args, "gen_fid_device", device)
    n = getattr(args, "gen_fid_n_samples", 1000)
    fid = FrechetInceptionDistance(normalize=True).to(fid_device)
    kid = KernelInceptionDistance(subset_size=getattr(args, "kid_subset_size", 100),
                                  normalize=True).to(fid_device)

    # Reals: the actual validation images, capped at gen_fid_n_real (the Inception pass over
    # the full split every time is the expensive half of this metric).
    max_real = getattr(args, "gen_fid_n_real", 0) or float("inf")
    n_real = 0
    for batch in valset_images_loader:
        imgs = denormalize(batch["image"].float().to(device), args.dataset_name, device).clamp(0, 1)
        fid.update(imgs.to(fid_device), real=True)
        kid.update(imgs.to(fid_device), real=True)
        n_real += imgs.shape[0]
        if n_real >= max_real:
            break

    # Fakes: class-balanced, fixed seed.
    labels = torch.arange(num_classes).repeat(int(np.ceil(n / num_classes)))[:n]
    for i in range(0, n, getattr(args, "sample_batch_size", 16)):
        chunk = labels[i:i + getattr(args, "sample_batch_size", 16)]
        imgs = generate_images(model, ae, stats, args, device, chunk,
                               seed=getattr(args, "sample_seed", 1234) + i)
        fid.update(imgs.to(fid_device), real=False)
        kid.update(imgs.to(fid_device), real=False)

    kid_mean, _ = kid.compute()
    out = {"gen_fid": fid.compute().item(), "gen_kid": kid_mean.item(),
           "gen_fid_n_real": n_real, "gen_fid_n_fake": n}
    fid.reset()
    kid.reset()
    return out


# ---------------------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------------------

def save_checkpoint(path, model, ema, optimizer, stats, args, epoch, num_classes, latent_shape,
                    class_names, include_optimizer=False):
    """Self-contained checkpoint: the sampler needs the latent stats and the AE identity, not
    just the weights.

    include_optimizer is off for best.pt/final_epoch.pt -- AdamW's two moment buffers are as
    large as the model twice over, and rewriting them on every val improvement across a
    2000-epoch run costs real wall-clock. Only last.pt (the resume point) carries them.
    """
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if include_optimizer else None,
        "latent_stats": stats.state_dict(),
        "epoch": epoch,
        "num_classes": num_classes,
        "class_names": class_names,
        "latent_shape": list(latent_shape),
        "ae_checkpoint": args.ae_checkpoint,
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
    }, path)


def train_latent_flow(args):
    device = select_device(args.device)
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)

    run_id = make_run_id(getattr(args, "run_prefix", "flow"))
    ckpt_dir = os.path.join(args.checkpoints, run_id)
    create_directory(ckpt_dir)
    save_config_copy(args, ckpt_dir)
    if args.do_wandb:
        setup_wandb(args, run_id)
        wandb.define_metric("Gen/*", step_metric="epoch")
        wandb.define_metric("Samples", step_metric="epoch")

    # ---- frozen autoencoder + labeled data ----
    # ae_checkpoint may name a run directory; the loader resolves it (last epoch first, then
    # best). Write the resolved file back onto args so the latent cache key, the saved flow
    # checkpoint and wandb all record which weights were actually used.
    ae = load_frozen_ae(args.ae_checkpoint, device, getattr(args, "ae_config", None))
    args.ae_checkpoint = ae.ae_checkpoint
    if args.do_wandb:
        wandb.config.update({"ae_checkpoint_resolved": ae.ae_checkpoint}, allow_val_change=True)
    trainset, valset = get_labeled_datasets(args.dataset_name, path=args.dataset_path,
                                            resize_img=args.resize_img,
                                            val_ratio=getattr(args, "val_ratio", 0.2),
                                            seed=args.seed,
                                            labels_csv=getattr(args, "labels_csv", None),
                                            label_column=getattr(args, "label_column", "noisy_labels_0"))
    num_classes = len(trainset.classes)
    class_names = list(getattr(trainset, "class_names", trainset.classes))
    print(f"[data] train={len(trainset)} ({getattr(trainset, 'layout', '?')}) "
          f"val={len(valset)} ({getattr(valset, 'layout', '?')}) classes={num_classes}")
    print(f"[data] labels from: {getattr(trainset, 'labels_csv', None) or 'directory/filename'}")
    print(f"[data] {class_names}")

    latent_train, latent_val = build_latent_datasets(args, ae, trainset, valset, device)

    # ---- latent standardization (estimated on TRAIN latents only) ----
    stats = LatentStats.from_latents(latent_train.latents,
                                     mode=getattr(args, "latent_norm", "per_channel"))
    stats = stats.to(device)
    print(f"[latents] {stats}")

    generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(latent_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=getattr(args, "latent_num_workers", 0),
                             worker_init_fn=seed_worker, generator=generator, drop_last=True)
    valloader = DataLoader(latent_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=getattr(args, "latent_num_workers", 0))
    # Real images for generative FID (kept separate from the latent loaders).
    val_image_loader = DataLoader(valset, batch_size=getattr(args, "encode_batch_size", 32),
                                  shuffle=False, num_workers=args.num_workers)

    # ---- flow model ----
    c, h, w = ae.latent_shape(args.resize_img)
    model = build_flow_model(args, in_channels=c, latent_size=h, num_classes=num_classes).to(device)
    print(f"[flow] UNet on {c}x{h}x{w} latents, {model.num_parameters() / 1e6:.1f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=getattr(args, "weight_decay", 0.0),
                                  betas=(0.9, getattr(args, "adam_beta2", 0.999)))
    lr_scheduler = build_lr_scheduler(optimizer, args)
    ema = ModelEMA(model, decay=getattr(args, "ema_decay", 0.9999)) if getattr(args, "use_ema", True) else None

    best_val = float("inf")
    start_epoch = 0

    # Optional resume from a `last.pt` (set resume_from in the config), for jobs that get
    # preempted or hit the wall clock. The latent stats are recomputed identically from the
    # same cache, so only the weights/optimizer/epoch need restoring.
    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch):
            if lr_scheduler is not None:
                lr_scheduler.step()
        print(f"[resume] {resume_from} -> continuing at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_step(model, trainloader, optimizer, stats, args, device,
                                   ema=ema, epoch=epoch)
        val_metrics = validation_step(model, valloader, stats, args, device,
                                      num_repeats=getattr(args, "val_repeats", 4))

        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch {epoch}: Train FM Loss={train_metrics['loss']:.5f} | "
              f"Val FM Loss={val_metrics['loss']:.5f} | lr={train_metrics['lr']:.2e}")

        log = {
            "epoch": epoch,
            "Train/Flow Loss": train_metrics["loss"],
            "Train/Learning Rate": train_metrics["lr"],
            "Val/Flow Loss": val_metrics["loss"],
        }

        sample_model = ema.ema if ema is not None else model
        every_sample = max(1, getattr(args, "sample_every_n_epochs", 10))
        if (epoch % every_sample == 0) or (epoch == args.epochs - 1):
            grid = sample_grid(sample_model, ae, stats, args, device, num_classes)
            if args.do_wandb:
                log["Samples"] = wandb.Image(grid, caption=f"Epoch {epoch} | rows = {class_names}")
            else:
                vutils.save_image(grid, os.path.join(ckpt_dir, f"samples_epoch{epoch:04d}.png"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        every_fid = max(1, getattr(args, "gen_fid_every_n_epochs", 25))
        if getattr(args, "gen_fid", True) and ((epoch % every_fid == 0 and epoch > 0)
                                               or epoch == args.epochs - 1):
            gen_metrics = generative_fid(sample_model, ae, stats, args, device,
                                         val_image_loader, num_classes)
            if gen_metrics:
                print(f"           Gen FID={gen_metrics['gen_fid']:.3f} "
                      f"KID={gen_metrics['gen_kid']:.5f} "
                      f"({gen_metrics['gen_fid_n_fake']} samples vs {gen_metrics['gen_fid_n_real']} reals)")
                log["Gen/FID"] = gen_metrics["gen_fid"]
                log["Gen/KID"] = gen_metrics["gen_kid"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.do_wandb:
            wandb.log(log, step=epoch)

        # best.pt tracks the held-out flow loss (FID is too expensive to run every epoch, and
        # too noisy at these sample counts to select on). It is BOOKKEEPING ONLY -- it never
        # stops or alters training. NOTE it is selected within a run; do not compare a best.pt
        # of one run against a final_epoch.pt of another.
        if val_metrics["loss"] < best_val - 1e-7:
            best_val = val_metrics["loss"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, ema, optimizer, stats,
                            args, epoch, num_classes, (c, h, w), class_names)
            print("Checkpoint saved (best val flow loss).")

        # `last.pt` is the resume point (carries the optimizer state); written periodically so
        # a preempted job loses at most save_every_n_epochs epochs.
        if epoch % max(1, getattr(args, "save_every_n_epochs", 25)) == 0 or epoch == args.epochs - 1:
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, ema, optimizer, stats, args,
                            epoch, num_classes, (c, h, w), class_names, include_optimizer=True)

        # NO EARLY STOPPING, deliberately. Every experiment must run the full `epochs` budget or
        # the comparison is not controlled: a run that halts at epoch 900 and one that runs to
        # 2000 have not been given the same compute, and the cosine LR schedule (built over
        # args.epochs) would be truncated mid-decay, leaving that run at a high LR it never
        # annealed from. If you change `epochs`, change it in BOTH configs.

    save_checkpoint(os.path.join(ckpt_dir, "final_epoch.pt"), model, ema, optimizer, stats, args,
                    args.epochs - 1, num_classes, (c, h, w), class_names)
    print(f"Final checkpoint saved in {ckpt_dir}")
    if args.do_wandb:
        wandb.finish()
    return model
