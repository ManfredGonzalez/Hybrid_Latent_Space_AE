#!/usr/bin/env python
"""
Self-supervised (SwAV) training for DualSSL -- the DualVAE codebook reused as SwAV prototypes.

This is a NEW, standalone experiment: it does not touch the DualVAE reconstruction code.
It demonstrates that with very few changes (drop the decoder + reconstruction, add two
augmented views + a swapped-prediction loss + Sinkhorn balancing) the same encoder/codebook
becomes an SSL learner. No reconstruction, no negatives, no decoder.

Validation is supervised online during training via a frozen-feature kNN classifier on the
Imagenette val set (labels from noisy_imagenette.csv), logged to Weights & Biases each few
epochs together with anti-collapse diagnostics (prototype-usage entropy, feature std).

Run:
    python train_dualSSL.py --config configs/dualssl_swav.yaml
"""
import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from tools.arguments import load_config_as_args
from tools.utils import (create_directory, set_seed, select_device, setup_wandb,
                         make_run_id, save_config_copy, build_lr_scheduler, seed_worker)
from models.dual_ssl import DualSSL

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENETTE_MEAN = (0.5, 0.5, 0.5)
IMAGENETTE_STD = (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_ssl_transform(size, scale_min, scale_max, cj_strength, blur_p, gray_p):
    """SimCLR/SwAV-style strong augmentation for one view."""
    s = cj_strength
    k = max(3, int(0.1 * size) // 2 * 2 + 1)   # odd Gaussian-blur kernel ~10% of the image
    return T.Compose([
        T.RandomResizedCrop(size, scale=(scale_min, scale_max), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        T.RandomGrayscale(p=gray_p),
        T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p),
        T.ToTensor(),
        T.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STD),
    ])


def build_eval_transform(size):
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STD),
    ])


def load_label_map(csv_path):
    m = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m[os.path.basename(row["path"])] = row["noisy_labels_0"]
    return m


class MultiViewDataset(Dataset):
    """Returns `num_views` independently-augmented crops of each image, stacked."""
    def __init__(self, root_dir, transform, num_views):
        self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))
                      if f.lower().endswith(VALID_EXTS)]
        self.transform = transform
        self.num_views = num_views

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        views = torch.stack([self.transform(img) for _ in range(self.num_views)])
        return views                                            # (num_views, 3, H, W)


class LabeledEvalDataset(Dataset):
    """Val images with integer class labels (from the CSV) for the frozen-feature monitor."""
    def __init__(self, root_dir, transform, label_map, class_to_idx):
        self.samples = [(os.path.join(root_dir, f), class_to_idx[label_map[f]])
                        for f in sorted(os.listdir(root_dir))
                        if f.lower().endswith(VALID_EXTS) and f in label_map]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), y


# --------------------------------------------------------------------------- #
# Online validation monitor (frozen-feature kNN + collapse diagnostics)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, eval_loader, device, k, test_frac, seed, num_prototypes):
    model.eval()
    reps, scores_all, labels = [], [], []
    for imgs, y in eval_loader:
        rep, scores = model.forward_eval(imgs.to(device))
        reps.append(rep.float().cpu()); scores_all.append(scores.float().cpu()); labels.append(y)
    reps = torch.cat(reps).numpy()
    scores_all = torch.cat(scores_all)
    y = torch.cat(labels).numpy()

    # kNN top-1 on a stratified val-internal split (gallery=fit, queries=test).
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    Xn = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-12)
    sim = Xn[te] @ Xn[tr].T                                     # (n_te, n_tr) cosine
    nn_idx = np.argsort(-sim, axis=1)[:, :k]
    votes = y[tr][nn_idx]                                       # (n_te, k)
    preds = np.array([np.bincount(v).argmax() for v in votes])
    knn_acc = float((preds == y[te]).mean())

    # Anti-collapse diagnostics.
    feat_std = float(Xn.std(axis=0).mean())
    p = F.softmax(scores_all / model.temperature, dim=1).mean(dim=0)   # mean assignment
    proto_entropy = float(-(p * (p + 1e-12).log()).sum() / np.log(num_prototypes))  # in [0,1]
    hard = scores_all.argmax(dim=1)
    protos_used = float(len(torch.unique(hard))) / num_prototypes
    return {"Val/kNN_Top1": knn_acc, "Val/Prototype_Entropy": proto_entropy,
            "Val/Prototypes_Used_Frac": protos_used, "Val/Feature_Std": feat_std}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args):
    device = select_device(getattr(args, "device", "cuda"))
    args.device = device
    set_seed(getattr(args, "seed", 42), getattr(args, "deterministic", False),
             getattr(args, "cudnn_benchmark", True))

    size = getattr(args, "resize_img", 224)
    num_views = getattr(args, "num_views", 2)
    train_tf = build_ssl_transform(size, getattr(args, "crop_scale_min", 0.4),
                                   getattr(args, "crop_scale_max", 1.0),
                                   getattr(args, "color_jitter_strength", 0.5),
                                   getattr(args, "gaussian_blur_p", 0.5),
                                   getattr(args, "grayscale_p", 0.2))
    eval_tf = build_eval_transform(size)

    train_dir = os.path.join(args.dataset_path, "train")
    val_dir = os.path.join(args.dataset_path, "val")
    csv_path = getattr(args, "labels_csv", None) or os.path.join(args.dataset_path, "noisy_imagenette.csv")
    label_map = load_label_map(csv_path)
    classes = sorted(set(label_map.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_set = MultiViewDataset(train_dir, train_tf, num_views)
    eval_set = LabeledEvalDataset(val_dir, eval_tf, label_map, class_to_idx)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              generator=gen, drop_last=True, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=getattr(args, "eval_batch_size", 128),
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[data] train={len(train_set)} imgs x {num_views} views | val(monitor)={len(eval_set)} imgs | {len(classes)} classes")

    num_prototypes = getattr(args, "num_prototypes", 256)
    model = DualSSL(
        latent_channels=getattr(args, "latent_channels", 8),
        downsample_factor=getattr(args, "downsample_factor", 8),
        num_prototypes=num_prototypes,
        embedding_mode=getattr(args, "embedding_mode", "global"),
        proj_dim=getattr(args, "proj_dim", None),
        proj_hidden_dim=getattr(args, "proj_hidden_dim", 256),
        l2_normalize_codes=getattr(args, "l2_normalize_codes", True),
        temperature=getattr(args, "temperature", 0.1),
        sinkhorn_eps=getattr(args, "sinkhorn_eps", 0.05),
        sinkhorn_iters=getattr(args, "sinkhorn_iters", 3),
        assignment=getattr(args, "assignment", "cosine"),
        ema_decay=getattr(args, "ema_decay", 0.99),
        sigma2_floor=getattr(args, "sigma2_floor", 0.1),
        sigma2_ceil=getattr(args, "sigma2_ceil", 10.0),
    ).to(device)
    # Duality / transfer: optionally seed the prototypes (and sigma_k^2) from a trained
    # DualVAE codebook -- the same GMM codebook serving both generative and SSL objectives.
    init_from = getattr(args, "init_codebook_from", None)
    if init_from:
        model.load_codebook_from_dualvae(init_from, device=device)
    print(f"[model] DualSSL | mode={model.embedding_mode} | assignment={model.assignment} | "
          f"prototypes={num_prototypes} x {model.feat_dim} | "
          f"projector={'yes' if model.projector is not None else 'no (codebook reused directly)'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=getattr(args, "weight_decay", 1e-6))
    scheduler = build_lr_scheduler(optimizer, args)
    use_amp = getattr(args, "use_amp", True)
    freeze_niters = getattr(args, "freeze_prototypes_niters", 300)
    eval_every = getattr(args, "eval_every_n_epochs", 5)

    run_id = make_run_id(prefix="dualssl")
    ckpt_dir = os.path.join(getattr(args, "checkpoints", "./checkpoints/dualssl/"), run_id)
    create_directory(ckpt_dir)
    save_config_copy(args, ckpt_dir)
    use_wandb = getattr(args, "do_wandb", True)
    if use_wandb:
        setup_wandb(args, run_id)

    global_step = 0
    best_knn = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        with tqdm(total=len(train_set), desc=f"Epoch {epoch}/{args.epochs}", unit="img") as pbar:
            for views in train_loader:
                views = views.to(device, non_blocking=True)     # (B, V, 3, H, W)
                view_list = [views[:, i] for i in range(views.shape[1])]
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    scores_list = [model.forward_scores(v) for v in view_list]
                loss = model.swav_loss([s.float() for s in scores_list],
                                       n_targets=getattr(args, "num_global_views", num_views))
                loss.backward()
                # SwAV: freeze prototypes for the first few hundred steps (stability).
                if global_step < freeze_niters:
                    for p in model.prototypes.parameters():
                        p.grad = None
                optimizer.step()

                running += loss.item(); nb += 1; global_step += 1
                if use_wandb and global_step % 20 == 0:
                    wandb_log({"Train/Loss_step": loss.item(),
                               "Train/LR": optimizer.param_groups[0]["lr"],
                               "train_step": global_step})
                pbar.set_postfix(loss=loss.item()); pbar.update(views.shape[0])

        metrics = {"epoch": epoch, "Train/Loss": running / max(nb, 1),
                   "Train/LR": optimizer.param_groups[0]["lr"]}
        if model.assignment == "gaussian":
            metrics.update({f"Codebook/{k}": v for k, v in model.sigma2_stats().items()})
        if epoch % eval_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, eval_loader, device,
                                   getattr(args, "knn_k", 20), getattr(args, "probe_test_frac", 0.4),
                                   args.seed, num_prototypes)
            metrics.update(val_metrics)
            print(f"[epoch {epoch}] " + "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in val_metrics.items()))
            if val_metrics["Val/kNN_Top1"] > best_knn:
                best_knn = val_metrics["Val/kNN_Top1"]
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        if use_wandb:
            wandb_log(metrics)
        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_epoch.pt"))
    print(f"[done] best Val/kNN_Top1={best_knn:.4f} | checkpoints in {ckpt_dir}")
    if use_wandb:
        import wandb
        wandb.finish()


def wandb_log(d):
    import wandb
    wandb.log(d)


def parse():
    p = argparse.ArgumentParser(description="SwAV SSL training for DualSSL.")
    p.add_argument("--config", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    cli = parse()
    args = load_config_as_args(cli.config)
    args.config = cli.config
    train(args)
