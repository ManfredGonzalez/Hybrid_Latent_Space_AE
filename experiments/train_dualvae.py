import os
import torch
import wandb
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.utils import create_directory, seed_worker, set_seed, setup_wandb, scale_ratio, build_lr_scheduler, build_val_fid, update_val_fid, compute_val_fid, should_run_val_fid, reset_val_fid, make_run_id, save_config_copy
from tools.distributed import (ddp_setup, ddp_cleanup, is_dist, is_main_process,
                               world_size, unwrap, all_reduce_metrics, broadcast_module_state_,
                               barrier)
from tools.normalization import denormalize
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.dual_vae import DUALVAE
from models.masked_predictor import sample_mask
from models.modules.cont_dropout import validate_cont_dropout_p
from losses.loss import dualvae_loss, kl_divergence_loss
from losses.reconstruction import build_reconstruction_criterion
from losses.gan import build_gan, generator_step_terms, discriminator_step
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.utils as vutils


# --- Cross-view code-invariance (CVI): make the codes augmentation-invariant (semantic
# content) WITHOUT touching the reconstruction path. See reports/semantic_codes_design.md.
# All gated on args.code_invariance (default False) -> zero effect on existing runs. ---
def build_ci_augment(args, device):
    """Per-image GPU appearance augmentation (kornia) for the CVI aux view. Returns a callable
    on Normalize(0.5,0.5,0.5) tensors, or None when CVI is off."""
    if not getattr(args, 'code_invariance', False):
        return None
    import kornia.augmentation as K
    size = args.resize_img
    mode = getattr(args, 'code_invariance_mode', 'pooled')
    if mode == 'aligned':
        # PHOTOMETRIC-ONLY: no crop/flip, so location i of the augmented view lines up with
        # location i of the clean view. That spatial correspondence is what makes the
        # per-location (aligned) CVI loss meaningful -- the invariance target is *appearance*
        # (color/blur/grayscale/noise), which is exactly what we want the codes to ignore.
        # (Crop/scale invariance is deliberately dropped here; it would need geometric
        # tracking to warp the code grid back into alignment.)
        aug = torch.nn.Sequential(
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=getattr(args, 'ci_grayscale_p', 0.2)),
            K.RandomGaussianBlur((23, 23), (0.1, 2.0), p=getattr(args, 'ci_blur_p', 0.5)),
            K.RandomGaussianNoise(mean=0.0, std=getattr(args, 'ci_noise_std', 0.05),
                                  p=getattr(args, 'ci_noise_p', 0.5)),
        ).to(device)
    else:
        # POOLED (legacy): geometric + photometric; safe only with the image-pooled loss,
        # which is position-invariant so crops don't misalign it (but pooling kills the signal).
        aug = torch.nn.Sequential(
            K.RandomResizedCrop((size, size), scale=(getattr(args, 'ci_crop_scale_min', 0.5), 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=getattr(args, 'ci_grayscale_p', 0.2)),
            K.RandomGaussianBlur((23, 23), (0.1, 2.0), p=getattr(args, 'ci_blur_p', 0.5)),
        ).to(device)

    def apply(images):
        x01 = (images * 0.5 + 0.5).clamp(0, 1)   # augment in [0,1], then renormalize
        return (aug(x01) - 0.5) / 0.5
    return apply


def code_invariance_weight(args, epoch):
    """Ramp lambda_inv from 0 (let the k-means/EMA codebook settle first)."""
    w = getattr(args, 'code_invariance_weight', 0.5)
    s = getattr(args, 'code_invariance_start_epoch', 8)
    r = max(1, getattr(args, 'code_invariance_ramp_epochs', 5))
    return w * min(max((epoch - s) / r, 0.0), 1.0)


@torch.no_grad()
def sinkhorn_balance(h, n_iters=3, eps=1e-8):
    """Sinkhorn-Knopp balancing of the (B, K) bag-of-codes TARGET: make code usage uniform
    ACROSS the batch (columns) while each image's row stays a distribution (sums to 1). This
    is the anti-collapse valve: the target now uses ALL codes, so the CVI loss can no longer
    be satisfied by shrinking the codebook to a few codes (the perplexity-collapse shortcut)."""
    Q = h.t().clone().float()                          # (K, B)
    K, B = Q.shape
    Q = Q / Q.sum().clamp(min=eps)
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=eps) / K   # codes -> uniform usage
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=eps) / B   # per-image -> uniform
    Q = Q * B                                          # each image's column sums to 1
    return Q.t()                                       # (B, K)


def code_invariance_loss(h_clean, h_aug, eps=1e-8):
    """Cross-entropy pulling the clean soft bag-of-codes toward the (detached) augmented
    target. Asymmetric so the augmented view is encoded under no_grad (no retained graph ->
    minimal extra memory). Pair with sinkhorn_balance on the target to prevent codebook
    collapse (otherwise a strong CVI weight collapses perplexity by shrinking the vocabulary).

    NOTE: this is the LEGACY image-pooled loss. Diagnosis showed it sits at ln(K) and is flat
    -- pooling over all HxW locations blurs each image's histogram toward uniform, so the two
    views trivially "match" with no gradient, and Sinkhorn makes it worse by flattening the
    target. Prefer code_invariance_aligned (per-location + me-max) below."""
    return -(h_aug.detach() * torch.log(h_clean + eps)).sum(1).mean()


def code_invariance_aligned(a_clean, a_aug, div_weight=1.0, eps=1e-8):
    """Spatially-ALIGNED cross-view code-invariance (the fix for the flat-at-ln(K) failure).

    a_clean, a_aug: (B, HW, K) PER-LOCATION soft assignments. The two views must be
    PHOTOMETRICALLY augmented only (no crop/flip) so location i corresponds across views.

    Two terms, both bounded and >= 0:
      * invariance  -- per-location cross-entropy CE(target=a_aug.detach(), pred=a_clean),
        averaged over locations. Unlike the pooled version this compares *which code each
        patch picked* across views, so there is a real gradient: "the ear patch must get the
        same code whether the photo is bluish or grayscale".
      * anti-collapse (me-max) -- penalize (ln K - H(batch-marginal usage)). This keeps ALL
        codes in use WITHOUT flattening any individual patch's distribution, so it does not
        destroy the per-patch signal the way Sinkhorn-on-the-target did. Zero iff the codes
        are used uniformly across the batch.

    Returns (loss, diagnostics dict) for monitoring.
    """
    B, HW, K = a_clean.shape
    tgt = a_aug.detach()
    ce = -(tgt * torch.log(a_clean + eps)).sum(-1).mean()          # invariance (want down)
    p_bar = a_clean.reshape(-1, K).mean(0)                         # batch-marginal code usage
    lnK = math.log(K)
    marg_H = -(p_bar * torch.log(p_bar + eps)).sum()
    div = lnK - marg_H                                             # 0 iff all codes used equally
    loss = ce + div_weight * div
    with torch.no_grad():
        # sharpness of each patch's assignment: ln(K) if uniform (bad), ~0 if confident (good).
        patch_H = -(a_clean * torch.log(a_clean + eps)).sum(-1).mean()
        # the interpretable gauge: fraction of patches that pick the SAME code in both views.
        agree = (a_clean.argmax(-1) == tgt.argmax(-1)).float().mean()
    diag = {"ce": ce.item(), "div": div.item(), "patch_entropy": patch_H.item(),
            "marginal_entropy": marg_H.item(), "agreement": agree.item()}
    return loss, diag


# --- Masked latent modeling (MSM): generative/self-supervised aux head. Predict HIDDEN latent
# locations from their visible neighbours -> makes the codes context-predictable (= semantic)
# and, in 'latent' mode, yields a native generative prior over the GMM latent. Augmentation-free,
# grounded (no collapse shortcut), reconstruction/GAN path untouched. Gated on args.masked_modeling
# (default False -> zero effect on existing runs). Two experiments share one predictor head:
#   * type 'code'   (MCM): CE vs the GMM soft assignment (responsibilities) at masked locations.
#   * type 'latent' (MLM): NLL of the true masked z_e_vq under the context-predicted mixture
#                          p(z|ctx) = sum_k r_k(ctx) N(e_k, sigma_k^2 I) -> a conditional GMM prior.
def build_masked_predictor(args, device):
    if not getattr(args, 'masked_modeling', False):
        return None, None
    from models.masked_predictor import MaskedLatentPredictor
    g = args.resize_img // args.downsample_factor
    pred = MaskedLatentPredictor(
        latent_channels=args.latent_channels, num_embeddings=args.num_embeddings, grid_hw=(g, g),
        dim=getattr(args, 'msm_dim', 256), depth=getattr(args, 'msm_depth', 4),
        heads=getattr(args, 'msm_heads', 8), dropout=getattr(args, 'msm_dropout', 0.0),
    ).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=getattr(args, 'msm_lr', args.lr),
                            weight_decay=getattr(args, 'msm_weight_decay', 0.0))
    n_params = sum(p.numel() for p in pred.parameters())
    print(f"[MSM] masked latent modeling ON [type={getattr(args,'masked_modeling_type','code')}]: "
          f"predictor {n_params/1e6:.1f}M params, mask_ratio={getattr(args,'msm_mask_ratio',0.5)}, "
          f"weight={getattr(args,'msm_weight',0.1)} start_epoch={getattr(args,'msm_start_epoch',8)}. "
          f"Watch Train/MSM Accuracy (up = codes predictable from context).")
    return pred, opt


def masked_modeling_weight(args, epoch):
    """Ramp the MSM weight from 0 (let the k-means/EMA codebook settle first)."""
    w = getattr(args, 'msm_weight', 0.1)
    s = getattr(args, 'msm_start_epoch', 8)
    r = max(1, getattr(args, 'msm_ramp_epochs', 5))
    return w * min(max((epoch - s) / r, 0.0), 1.0)


def masked_code_modeling_loss(logits, gamma_teacher, mask):
    """MCM: cross-entropy at MASKED locations between the predicted code distribution and the
    GMM's own soft assignment (responsibilities), which the caller has detached. gamma_teacher,
    logits: (B, HW, K); mask: (B, HW) bool. Returns (loss, diag)."""
    import torch.nn.functional as F
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(gamma_teacher * logp).sum(-1)                       # (B, HW)
    loss = ce[mask].mean()
    with torch.no_grad():
        acc = (logits.argmax(-1)[mask] == gamma_teacher.argmax(-1)[mask]).float().mean()
    return loss, {"acc": acc.item()}


def masked_latent_gmm_loss(logits, z_target, means, sigma2, mask, eps=1e-8):
    """MLM: negative log-likelihood of the TRUE masked latent z under the context-predicted
    mixture  p(z|ctx) = sum_k r_k(ctx) N(e_k, sigma_k^2 I),  r = softmax(logits). This is a
    conditional version of the code-centered GMM prior, so the head is also a generative prior
    over z_e_vq (sample k ~ r_k, then z ~ N(e_k, sigma_k^2)). logits (B,HW,K); z_target (B,HW,C);
    means (K,C); sigma2 (K,) -- means/sigma2/z_target all detached by the caller. Returns (loss, diag)."""
    import torch.nn.functional as F
    B, HW, K = logits.shape
    C = z_target.shape[-1]
    log_r = F.log_softmax(logits.float(), dim=-1)             # (B,HW,K) -- carries gradient
    z = z_target.float()
    # ||z - e_k||^2 via the expansion (no (B,HW,K,C) intermediate); constants, no graph retained.
    d2 = (z * z).sum(-1, keepdim=True) + (means * means).sum(-1).view(1, 1, K) - 2.0 * (z @ means.t())
    logN = -0.5 * (C * math.log(2 * math.pi) + C * torch.log(sigma2).view(1, 1, K)
                   + d2 / sigma2.view(1, 1, K))               # log N(z; e_k, sigma_k^2 I)
    log_mix = torch.logsumexp(log_r + logN, dim=-1)          # (B,HW) conditional log-likelihood
    nll = -log_mix[mask]
    loss = nll.mean()
    with torch.no_grad():
        bpd = loss / (C * math.log(2))                        # bits-per-dim (lower = better prior)
        acc = (logits.argmax(-1)[mask] == d2.argmin(-1)[mask]).float().mean()  # picks the true code?
        r = log_r.exp()
        comp_H = -(r * log_r).sum(-1)[mask].mean()            # component entropy (context sharpness)
    return loss, {"acc": acc.item(), "bpd": bpd.item(), "comp_entropy": comp_H.item()}


def _maybe_subset(dataset, n, seed, what="val"):
    """Deterministically cap a split at `n` images (`n` <= 0 / None -> untouched).

    Full-ImageNet validation is 50k images; running the whole thing -- plus InceptionV3 for
    FID/KID -- every single epoch costs more GPU-hours than it informs. A fixed 5-10k subset
    tracks the same curves at a fraction of the cost. The permutation is seeded, so the
    subset is identical across epochs, across ranks, and across restarts, which is what makes
    the epoch-to-epoch metric deltas meaningful.
    """
    if not n or n <= 0 or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    if is_main_process():
        print(f"[data] {what}: using a fixed {n}-image subset of {len(dataset)} (seed {seed}).")
    return Subset(dataset, idx)


def prepare_data(args):
    generator = torch.Generator().manual_seed(args.seed)
    dataset_name = getattr(args, 'dataset_name', 'pineapple').lower()
    if dataset_name == 'pineapple':
        trainset = PineappleDataset(
            path=args.dataset_path,
            split='train', test_txt=args.path_test_ids, augment=False, seed=args.seed
        )
        valset = PineappleDataset(
            path=args.dataset_path,
            split='val', test_txt=args.path_test_ids, augment=False, seed=args.seed
        )
        testset = PineappleDataset(
            path=args.dataset_path,
            split='test', test_txt=args.path_test_ids, augment=False, seed=args.seed
        )
    else:
        # Load CIFAR, MNIST, Imagenette or ImageNet
        if dataset_name in ("imagenette", "imagenet"):
            # Both return (train, val) directly and ignore the `split` argument. For imagenet
            # `dataset_path` is the single .h5 from tools/build_imagenet_subset.py, read by
            # data.datasets.HDF5ImageDataset.
            trainset, valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, resize_img=args.resize_img, seed=args.seed)
            testset = valset # Or a dedicated test split if available
        else:
            # Load CIFAR or MNIST
            trainset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='train', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='val', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            testset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='test', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)

    # Optional fixed val/test subsets (see _maybe_subset): the headline cost saver on ImageNet.
    val_n = getattr(args, 'val_subset_size', 0)
    valset = _maybe_subset(valset, val_n, args.seed, "val")
    testset = _maybe_subset(testset, val_n, args.seed, "test") if testset is not None else None

    # --- DDP: shard each split across ranks -------------------------------------------------
    # Without a DistributedSampler every rank would iterate the SAME images, so a 4-GPU run
    # would do 4x the work of a 1-GPU run for exactly the same gradient. The sampler also owns
    # the shuffling (hence shuffle=False on the loader), and needs set_epoch() each epoch or
    # every epoch replays one identical permutation.
    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True) if is_dist() else None
    val_sampler = DistributedSampler(valset, shuffle=False, drop_last=False) if is_dist() else None
    test_sampler = DistributedSampler(testset, shuffle=False, drop_last=False) if (is_dist() and testset is not None) else None

    nw = args.num_workers
    # persistent_workers keeps the pool alive between epochs -- with 1.28M-image epochs the
    # respawn is minor, but re-opening the HDF5 handle in every worker every epoch is not.
    # prefetch_factor + pin_memory keep the H100 fed while workers decode JPEGs.
    loader_kwargs = dict(num_workers=nw, worker_init_fn=seed_worker, generator=generator,
                         pin_memory=True)
    if nw > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=getattr(args, 'prefetch_factor', 4))

    # drop_last on TRAIN only: a short final batch makes ranks disagree on the number of
    # optimizer steps, which deadlocks DDP's gradient all-reduce.
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=(train_sampler is None), sampler=train_sampler,
                             drop_last=True, **loader_kwargs)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                           sampler=val_sampler, **loader_kwargs)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            sampler=test_sampler, **loader_kwargs)
    return trainset, valset, testset, trainloader, valloader, testloader

def check_device_and_vram(model, loader, device):
    print("\n" + "="*40)
    print("--- Pre-Training Device & VRAM Check ---")
    
    # 1. Check Model Device
    # We check the device of the first parameter in the model
    model_device = next(model.parameters()).device
    print(f"Model is currently on: {model_device}")
    
    # 2. Check Data Device
    # By default, DataLoaders keep data on the CPU until you explicitly move it.
    sample_batch = next(iter(loader))
    cpu_images = sample_batch["image"]
    print(f"Data straight from DataLoader is on: {cpu_images.device}")
    
    # Simulate moving it to the device like you do in your training loop
    gpu_images = cpu_images.to(device)
    print(f"Data successfully moved to target: {gpu_images.device}")
    
    # 3. Check GPU Memory (if using CUDA)
    if torch.cuda.is_available() and 'cuda' in str(device):
        # Convert bytes to Megabytes for readability
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        
        print(f"\nGPU VRAM Status on [{torch.cuda.get_device_name(device)}]:")
        print(f"  Allocated (Weights + 1 Batch): {allocated:.2f} MB")
        print(f"  Reserved (Cached by PyTorch):  {reserved:.2f} MB")
        print(f"  Total VRAM on Device:          {total_vram:.2f} MB")
    else:
        print("\nCUDA is not active or device is set to CPU. No VRAM to report.")
        
    print("="*40 + "\n")

def reconstruct_grid(model, dataset, args, n_samples=8):
    # Rank-0 only (the caller gates it): this indexes the dataset directly, outside the
    # DistributedSampler, so calling it on every rank would just render the same panel N times.
    model = unwrap(model)
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    # torch.stack, not torch.tensor(np.stack(...)): the transform already yields tensors, and
    # the numpy round-trip both copies and raises a UserWarning.
    imgs = torch.stack([dataset[i]["image"] for i in idxs]).to(args.device)

    with torch.no_grad():
        recon, _, _ = model(imgs)

    # Denormalize if needed (here assume already in [0,1])
    grid = vutils.make_grid(torch.cat([denormalize(imgs, args.dataset_name, args.device), denormalize(recon, args.dataset_name, args.device)], dim=0), nrow=n_samples, normalize=True, scale_each=True)
    return grid

def build_recon_criterion(args):
    return build_reconstruction_criterion(
        name=getattr(args, 'perceptual_loss', 'none'),
        device=args.device,
        perceptual_weight=getattr(args, 'perceptual_weight', 1.0),
        ffl_alpha=getattr(args, 'ffl_alpha', 1.0),
        dataset_name=args.dataset_name,
        perceptual_batch_fraction=getattr(args, 'perceptual_batch_fraction', 1.0),
    )

def initialize_model(args):
    model = DUALVAE(
        commitment_cost=args.commitment_cost,
        latent_channels=args.latent_channels,
        num_embeddings=args.num_embeddings,
        downsample_factor=getattr(args, 'downsample_factor', 8),
        l2_normalize_codes=getattr(args, 'l2_normalize_codes', False),
        cont_dropout_p=getattr(args, 'cont_dropout_p', 0.0),
        use_ema_codebook=getattr(args, 'use_ema_codebook', False),
        ema_decay=getattr(args, 'ema_decay', 0.99),
        ema_eps=getattr(args, 'ema_eps', 1e-5),
        ema_dead_threshold=getattr(args, 'ema_dead_threshold', 1.0),
        rq_depth=getattr(args, 'rq_depth', 1),
        residual_continuous=getattr(args, 'residual_continuous', False),
        component_prior=getattr(args, 'component_prior', False),
        sigma2_floor=getattr(args, 'sigma2_floor', 1e-3),
        sigma2_ceil=getattr(args, 'sigma2_ceil', 10.0),
        wavelet_detail=getattr(args, 'wavelet_detail', False),
        wavelet_band_channels=getattr(args, 'wavelet_band_channels', None),
        hierarchical_semantic=getattr(args, 'hierarchical_semantic', False),
        coarse_factor=getattr(args, 'coarse_factor', 4),
        coarse_num_embeddings=getattr(args, 'coarse_num_embeddings', 64),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def codebook_health_metrics(model):
    """Codebook-state diagnostics for wandb (read once per epoch, cheap).

    - Rel Quant Error: ||z_q - z||^2 / ||z||^2, scale-invariant (immune to the
      'encoder magnitude grew so sum-MSE looks worse' artifact).
    - Perplexity Depth d: assignment perplexity per RQ depth.
    - Pi Perplexity (EMA): exp(entropy of the EMA mixture weights pi_k) -- the
      long-horizon effective number of components (batch perplexity is one batch).
    - Restarted Codes: dead-code restarts in the last step (persistent churn here
      means the restart threshold is thrashing).
    - Sigma2 Mean/Min/Max + At-Floor Frac: distribution of the per-component prior
      variances; mass piling onto the floor is the collapse-ratchet signature.
    """
    vq = unwrap(model).vq_layer
    metrics = {"Codebook/Rel Quant Error": vq.rel_quant_error.item()}
    for d, perp in enumerate(getattr(vq, 'perplexity_per_depth', []) or []):
        metrics[f"Codebook/Perplexity Depth {d + 1}"] = perp
    if vq.use_ema:
        pi = vq.pi
        metrics["Codebook/Pi Perplexity (EMA)"] = torch.exp(-torch.sum(pi * torch.log(pi + 1e-10))).item()
        metrics["Codebook/Restarted Codes"] = vq.restarted_codes.item()
        sigma2 = vq.sigma2
        metrics["Codebook/Sigma2 Mean"] = sigma2.mean().item()
        metrics["Codebook/Sigma2 Min"] = sigma2.min().item()
        metrics["Codebook/Sigma2 Max"] = sigma2.max().item()
        metrics["Codebook/Sigma2 At Floor Frac"] = (sigma2 <= vq.sigma2_floor * 1.001).float().mean().item()
    return metrics


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, beta_kl_loss, recon_criterion, use_amp=False, gan=None,
                    ci_aug=None, ci_weight=0.0, ci_tau=0.5, ci_balance=False, ci_sinkhorn=False,
                    ci_mode='pooled', ci_target_tau=None, ci_div_weight=1.0,
                    msm_predictor=None, msm_opt=None, msm_type='code', msm_weight=0.0,
                    msm_balance=True, msm_max_scale=50.0, msm_mask_ratio=0.5, msm_tau=0.5,
                    coarse_kl_beta=0.001, limit_train_batches=0):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "scale_ratio": 0.0,
        "actual_mean_variance": 0.0,
        "cont_dropout_rate": 0.0,
        "gan_g_loss": 0.0,
        "gan_d_loss": 0.0,
        "gan_d_weight": 0.0,
        "code_invariance": 0.0,
        "code_invariance_eff_weight": 0.0,
        "cvi_agreement": 0.0,          # aligned-mode diagnostics (0 in pooled mode)
        "cvi_patch_entropy": 0.0,
        "cvi_marginal_entropy": 0.0,
        "masked_modeling": 0.0,        # MSM aux loss + diagnostics (0 when off)
        "masked_modeling_eff_weight": 0.0,
        "msm_acc": 0.0,
        "msm_bpd": 0.0,
        "msm_comp_entropy": 0.0,
        "coarse_commit": 0.0,          # hierarchical semantic level (0 when off)
        "coarse_kl": 0.0,
        "coarse_perplexity": 0.0,
        "num_batches": 0,
    }

    # `core` is the unwrapped module: the codebook/EMA diagnostics below live on it, and
    # `model` may be a DDP wrapper. Forward passes still go through `model` so DDP can hook
    # the backward -- only attribute ACCESS uses `core`.
    core = unwrap(model)
    # limit_train_batches > 0 truncates the epoch to that many steps PER RANK. Used by the
    # smoke config to exercise the whole pipeline (DDP, EMA all-reduce, GAN both phases,
    # FID sync, checkpointing) in minutes instead of a 1.28M-image epoch. Every rank applies
    # the SAME limit, so the ranks still agree on the step count and DDP cannot deadlock.
    n_steps = len(loader) if not limit_train_batches else min(int(limit_train_batches), len(loader))
    # One progress bar (rank 0) counting GLOBAL images, so the ETA is the run's, not one shard's.
    with tqdm(total=n_steps * loader.batch_size * world_size(), desc=f'Epoch {epoch}/{total_epochs}',
              unit='img', disable=not is_main_process()) as pbar:
        for step, batch in enumerate(loader):
            if step >= n_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad()
            if msm_opt is not None:
                msm_opt.zero_grad()
            # Forward pass under bf16 autocast (memory/speed); loss computation happens
            # OUTSIDE the autocast region in full fp32. The big sum-reductions (MSE over
            # every pixel, KL over every latent) lose real precision in bf16's 8-bit
            # mantissa, and .float() casts here are cheap. Gradients themselves already
            # accumulate in fp32 (params are fp32; autocast only affects op compute dtype).
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)

            vq_loss_val = vq_related_losses["vq_loss"].float()
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: total/b, recon/b, vq/b, kl/b, pixel_term/b, perceptual_term/b
            # keep_mask restricts the KL to samples whose continuous branch reached the
            # decoder this step (None when cont_dropout_p == 0 -> unmasked, old behavior).
            loss, recon_loss, vq_loss_final, kl_loss, pixel_term, perceptual_term = dualvae_loss(
                recon.float(), images.float(), vq_loss_val, beta_kl_loss,
                vanilla_vae_related_losses["mean"].float(), vanilla_vae_related_losses["log_variance"].float(),
                reduction='sum', recon_criterion=recon_criterion,
                keep_mask=vanilla_vae_related_losses["keep_mask"],
                prior_var=vanilla_vae_related_losses["prior_var"]
            )

            # Optional VQGAN-style adversarial term (inactive before gan_start_epoch;
            # the adaptive weight balances it against recon_loss at the decoder's last layer).
            gan_extra, g_loss_val, d_weight_val = generator_step_terms(gan, epoch, recon, recon_loss)
            loss = loss + gan_extra

            # Optional hierarchical SEMANTIC level: coarse commitment + coarse code-centered KL
            # (present only when the model was built with hierarchical_semantic). Same scale
            # convention as the fine terms (sum-reduced, per-sample).
            coarse_commit_val = 0.0
            coarse_kl_val = 0.0
            coarse_perp_val = 0.0
            if "coarse_commit" in vq_related_losses:
                b_ = images.size(0)
                c_commit = vq_related_losses["coarse_commit"].float()
                c_kl = kl_divergence_loss(vq_related_losses["coarse_mean"].float(),
                                          vq_related_losses["coarse_logvar"].float(),
                                          reduction='sum', prior_var=vq_related_losses["coarse_prior_var"])
                loss = loss + (c_commit + coarse_kl_beta * c_kl) / b_
                coarse_commit_val = (c_commit / b_).item()
                coarse_kl_val = (c_kl / b_).item()
                coarse_perp_val = vq_related_losses["coarse_perplexity"].item()

            # Optional cross-view code-invariance aux term (semantic codes; reconstruction path
            # untouched). +1 encoder pass on an augmented view; the clean z_e_vq is reused.
            ci_val = 0.0
            ci_eff_w = 0.0
            ci_diag = {}
            if ci_aug is not None and ci_weight > 0.0:
                # Augmented view = detached TARGET: encode it under no_grad so no graph is
                # retained (keeps the memory overhead of CVI ~negligible). Only the clean
                # view (reused z_e_vq, already in the main graph) carries the CVI gradient.
                if ci_mode == 'aligned':
                    # ALIGNED (recommended): per-location match on a PHOTOMETRICALLY augmented
                    # view (grids line up), with me-max anti-collapse. Real gradient, no
                    # Sinkhorn target-flattening. Target may be sharpened (ci_target_tau < ci_tau).
                    ttau = ci_target_tau if ci_target_tau is not None else ci_tau
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        a_aug = core.code_soft_assign(core.encode_zevq(ci_aug(images)).float(), tau=ttau)
                    a_clean = core.code_soft_assign(vq_related_losses["z_e_vq"].float(), tau=ci_tau)
                    ci_term, ci_diag = code_invariance_aligned(a_clean, a_aug, div_weight=ci_div_weight)
                else:
                    # POOLED (legacy): image-level bag-of-codes CE. Diagnosed as flat-at-ln(K).
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        h_aug = core.code_histogram(core.encode_zevq(ci_aug(images)).float(), tau=ci_tau)
                    if ci_sinkhorn:
                        h_aug = sinkhorn_balance(h_aug)      # anti-collapse: balanced-usage target
                    h_clean = core.code_histogram(vq_related_losses["z_e_vq"].float(), tau=ci_tau)
                    ci_term = code_invariance_loss(h_clean, h_aug)
                # The reconstruction loss is sum-reduced (huge), so a raw CVI weight is drowned.
                # ci_balance -> scale CVI to a fraction of the (detached) reconstruction
                # magnitude each step, so `ci_weight` becomes an O(1) *target fraction* of recon
                # (robust to image size / batch / beta). Clamped so early tiny ci_term can't blow up.
                if ci_balance:
                    scale = (recon_loss.detach().abs() / (ci_term.detach().abs() + 1e-8)).clamp(max=1e4)
                    eff_w = ci_weight * scale
                else:
                    eff_w = ci_weight
                loss = loss + eff_w * ci_term
                ci_val = ci_term.item()
                ci_eff_w = float(eff_w)   # effective CVI weight this step (varies in balanced mode)

            # Optional masked latent modeling aux term (generative/self-supervised; reconstruction
            # path untouched). Reuses the clean z_e_vq from the main forward (no extra encoder pass):
            # mask locations, predict them from the visible ones, and either match the GMM's soft
            # code assignment (type 'code') or score the true latent under the context-predicted
            # mixture (type 'latent'). Gradient flows through the VISIBLE tokens into the encoder;
            # the target is detached.
            msm_val = 0.0
            msm_eff_w = 0.0
            msm_diag = {}
            if msm_predictor is not None and msm_weight > 0.0:
                z_full = vq_related_losses["z_e_vq"].float()          # (B,C,h,w), carries grad
                b_, c_, h_, w_ = z_full.shape
                mask = sample_mask(b_, h_ * w_, msm_mask_ratio, z_full.device)
                logits = msm_predictor(z_full, mask)                  # visible tokens -> encoder grad
                if msm_type == 'latent':
                    means = core.vq_layer.embedding.weight.detach()
                    sigma2 = core.vq_layer.sigma2.detach()
                    z_tgt = z_full.detach().permute(0, 2, 3, 1).reshape(b_, h_ * w_, c_)
                    msm_term, msm_diag = masked_latent_gmm_loss(logits, z_tgt, means, sigma2, mask)
                else:
                    with torch.no_grad():
                        gamma = core.code_soft_assign(z_full.detach(), tau=msm_tau)  # (B,HW,K)
                    msm_term, msm_diag = masked_code_modeling_loss(logits, gamma, mask)
                if msm_balance:
                    scale = (recon_loss.detach().abs() / (msm_term.detach().abs() + 1e-8)).clamp(max=msm_max_scale)
                    msm_eff = msm_weight * scale
                else:
                    msm_eff = msm_weight
                loss = loss + msm_eff * msm_term
                msm_val = msm_term.item()
                msm_eff_w = float(msm_eff)

            loss.backward()
            optimizer.step()
            if msm_opt is not None:
                msm_opt.step()

            # Discriminator update on (real, fake.detach()), after the generator step.
            d_loss_val = discriminator_step(gan, epoch, images, recon)

            batch_scale_ratio = scale_ratio(vq_related_losses["z_vq"], vanilla_vae_related_losses["z_vanilla_post"])
            raw_variance = torch.exp(vanilla_vae_related_losses["log_variance"]).mean()

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
            running["perplexity"] += core.vq_layer.perplexity.item()
            running["codebook_usage"] += core.vq_layer.codebook_usage.item()
            running["scale_ratio"] += batch_scale_ratio.item()
            running["actual_mean_variance"] += raw_variance.item()
            running["cont_dropout_rate"] += core.last_drop_fraction
            running["gan_g_loss"] += g_loss_val
            running["gan_d_loss"] += d_loss_val
            running["gan_d_weight"] += d_weight_val
            running["code_invariance"] += ci_val
            running["code_invariance_eff_weight"] += ci_eff_w
            running["cvi_agreement"] += ci_diag.get("agreement", 0.0)
            running["cvi_patch_entropy"] += ci_diag.get("patch_entropy", 0.0)
            running["cvi_marginal_entropy"] += ci_diag.get("marginal_entropy", 0.0)
            running["masked_modeling"] += msm_val
            running["masked_modeling_eff_weight"] += msm_eff_w
            running["msm_acc"] += msm_diag.get("acc", 0.0)
            running["msm_bpd"] += msm_diag.get("bpd", 0.0)
            running["msm_comp_entropy"] += msm_diag.get("comp_entropy", 0.0)
            running["coarse_commit"] += coarse_commit_val
            running["coarse_kl"] += coarse_kl_val
            running["coarse_perplexity"] += coarse_perp_val
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0) * world_size())

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

def validate_one_epoch(model, loader, device, beta_kl_loss, dataset_name, recon_criterion, use_amp=False, fid_bundle=None):
    model.eval()
    core = unwrap(model)
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "scale_ratio": 0.0,
        "actual_mean_variance": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "num_batches": 0,
    }

    # Initialize metrics with a data range of 1.0 (since your images are 0-1)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            # Same split as training: bf16 forward, fp32 loss math. In eval mode
            # keep_mask is always None (dropout inactive), so the KL is unmasked.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: total/b, recon/b, vq/b, kl/b, pixel_term/b, perceptual_term/b
            loss, recon_loss, vq_loss_final, kl_loss, pixel_term, perceptual_term = dualvae_loss(
                recon.float(), images.float(), vq_related_losses["vq_loss"].float(), beta_kl_loss,
                mean=vanilla_vae_related_losses["mean"].float(), logvar=vanilla_vae_related_losses["log_variance"].float(),
                reduction='sum', recon_criterion=recon_criterion,
                keep_mask=vanilla_vae_related_losses["keep_mask"],
                prior_var=vanilla_vae_related_losses["prior_var"]
            )

            batch_scale_ratio = scale_ratio(vq_related_losses["z_vq"], vanilla_vae_related_losses["z_vanilla_post"])
            raw_variance = torch.exp(vanilla_vae_related_losses["log_variance"]).mean()

            # --- DENORMALIZATION FIX ---
            # Denormalize both targets and predictions back to [0, 1]; cast to fp32 first since
            # metrics/clamping are more reliable outside the autocast region.
            denorm_images = denormalize(images.float(), dataset_name, device)
            denorm_recon = denormalize(recon.float(), dataset_name, device)

            # Clamp after denormalization to ensure strict [0, 1] bounds for the metrics
            recon_clamped = denorm_recon.clamp(0, 1)
            images_clamped = denorm_images.clamp(0, 1)

            # Calculate metrics on the clean [0, 1] images
            batch_psnr = psnr_metric(recon_clamped, images_clamped)
            batch_ssim = ssim_metric(recon_clamped, images_clamped)
            update_val_fid(fid_bundle, images_clamped, recon_clamped)

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
            running["perplexity"] += core.vq_layer.perplexity.item()
            running["codebook_usage"] += core.vq_layer.codebook_usage.item()
            running["scale_ratio"] += batch_scale_ratio.item()
            running["actual_mean_variance"] += raw_variance.item()
            # Accumulate the batch metrics
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    out = {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}
    out.update(compute_val_fid(fid_bundle))  # adds 'rfid'/'kid_mean' when val_fid enabled
    return out

def log_metrics(epoch, train_metrics, val_metrics, valset, model, args):
    recon_grid = reconstruct_grid(model, valset, args, n_samples=8)

    wandb.log({
        # --- Images ---
        "epoch": epoch,
        "Sample Reconstructions": wandb.Image(recon_grid, caption=f"Epoch {epoch}"),
        "Train/Total Loss": train_metrics["loss"],
        "Train/Reconstruction Loss": train_metrics["recon_loss"],
        "Train/VQ Loss": train_metrics["vq_loss"],
        "Train/Commitment Loss": train_metrics["commitment_loss"],
        "Train/Codebook Loss": train_metrics["codebook_loss"],
        "Train/Pixel Term": train_metrics["pixel_term"],
        "Train/Perceptual Term": train_metrics["perceptual_term"],
        "Train/Codebook Perplexity": train_metrics["perplexity"],
        "Train/Codebook Usage": train_metrics["codebook_usage"],
        "Train/Scale Ratio (VQ/Cont)": train_metrics["scale_ratio"],
        "Train/Actual Mean Variance": train_metrics["actual_mean_variance"],
        "Train/Cont Dropout Rate": train_metrics["cont_dropout_rate"],
        "Train/GAN G Loss": train_metrics.get("gan_g_loss", 0.0),
        "Train/GAN D Loss": train_metrics.get("gan_d_loss", 0.0),
        "Train/GAN D Weight": train_metrics.get("gan_d_weight", 0.0),
        "Train/Code Invariance": train_metrics.get("code_invariance", 0.0),
        "Train/Code Invariance Eff Weight": train_metrics.get("code_invariance_eff_weight", 0.0),
        # CVI health gauges (aligned mode): agreement should RISE toward 1 (same code both
        # views); patch entropy should FALL (confident per-patch codes); marginal entropy
        # should stay HIGH (near ln(256)=5.545, i.e. all codes still used -> no collapse).
        "Train/CVI Agreement": train_metrics.get("cvi_agreement", 0.0),
        "Train/CVI Patch Entropy": train_metrics.get("cvi_patch_entropy", 0.0),
        "Train/CVI Marginal Entropy": train_metrics.get("cvi_marginal_entropy", 0.0),
        # Masked latent modeling gauges: MSM Accuracy should RISE (codes predictable from
        # context = becoming semantic); MSM Bits/Dim should FALL (latent mode: better prior).
        "Train/MSM Loss": train_metrics.get("masked_modeling", 0.0),
        "Train/MSM Eff Weight": train_metrics.get("masked_modeling_eff_weight", 0.0),
        "Train/MSM Accuracy": train_metrics.get("msm_acc", 0.0),
        "Train/MSM Bits Per Dim": train_metrics.get("msm_bpd", 0.0),
        "Train/MSM Component Entropy": train_metrics.get("msm_comp_entropy", 0.0),
        # Hierarchical semantic level: coarse codebook perplexity (effective #concepts used),
        # coarse commitment, and coarse code-centered KL. 0 when hierarchical_semantic is off.
        "Train/Coarse Perplexity": train_metrics.get("coarse_perplexity", 0.0),
        "Train/Coarse Commitment": train_metrics.get("coarse_commit", 0.0),
        "Train/Coarse KL": train_metrics.get("coarse_kl", 0.0),
        "Train/Learning Rate": train_metrics.get("lr", args.lr),
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/VQ Loss": val_metrics["vq_loss"],
        "Val/Commitment Loss": val_metrics["commitment_loss"],
        "Val/Codebook Loss": val_metrics["codebook_loss"],
        "Val/Pixel Term": val_metrics["pixel_term"],
        "Val/Perceptual Term": val_metrics["perceptual_term"],
        "Val/KL Divergence": val_metrics["kl_loss"],
        "Val/Codebook Perplexity": val_metrics["perplexity"],
        "Val/Codebook Usage": val_metrics["codebook_usage"],
        "Val/Scale Ratio (VQ/Cont)": val_metrics["scale_ratio"],
        "Val/Actual Mean Variance": val_metrics["actual_mean_variance"],
        # --- Image Quality Metrics ---
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
        # rFID / KID only present when val_fid: true (recon-FID over the val set).
        **({"Val/rFID": val_metrics["rfid"]} if "rfid" in val_metrics else {}),
        **({"Val/KID Mean": val_metrics["kid_mean"]} if "kid_mean" in val_metrics else {}),
        # --- Codebook / GMM health (current state, once per epoch) ---
        **codebook_health_metrics(model),
    }, step=epoch)


def save_checkpoint(model, epoch, best_loss, current_loss, patience_counter, checkpoint_dir):
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        filename = "best.pt"
        path = os.path.join(checkpoint_dir, filename)
        # unwrap() so the checkpoint holds plain `encoder.*` keys rather than DDP's
        # `module.encoder.*`. That keeps 4-GPU checkpoints loadable by every existing
        # single-GPU consumer (generate_test_inferences.py, tools/dualvae_latent_analysis.py,
        # the latent-flow trainer) with no changes.
        if is_main_process():
            torch.save(unwrap(model).state_dict(), path)
            print(f"Checkpoint saved: {filename}")
    else:
        patience_counter += 1
        if is_main_process():
            print(f"No improvement for {patience_counter} epoch(s).")
    return best_loss, patience_counter


def save_training_state(path, epoch, model, optimizer, lr_scheduler, gan,
                        msm_predictor, msm_opt, best_loss, patience_counter):
    """Write the FULL training state so a killed job can pick up exactly where it stopped.

    best.pt / final_epoch.pt hold weights only, which is all inference needs but not enough to
    continue training: without the Adam moments, the cosine schedule position, the epoch counter
    and the discriminator's own optimizer state, a "resumed" run silently restarts the schedule
    and re-warms the critic. On a multi-day ImageNet run that difference is worth hundreds of
    GPU-hours, so this is written every epoch.

    Saved via a temp file + os.replace: the rename is atomic, so a job killed mid-write (the
    normal way a job dies -- wall-clock limit) cannot leave a truncated last.pt behind.
    """
    state = {
        'epoch': epoch,
        'model': unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
        'patience_counter': patience_counter,
    }
    if lr_scheduler is not None:
        state['lr_scheduler'] = lr_scheduler.state_dict()
    if gan is not None:
        state['gan_disc'] = gan['disc'].state_dict()
        state['gan_opt'] = gan['opt'].state_dict()
    if msm_predictor is not None:
        state['msm'] = msm_predictor.state_dict()
        state['msm_opt'] = msm_opt.state_dict()
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def resolve_resume_path(args):
    """Turn --resume into a concrete last.pt path, or None.

    'auto' scans the configured checkpoints directory for the most recently modified last.pt,
    which is what a requeued SLURM job wants: the same submit line works whether it is the
    first attempt or the fourth. A missing file is NOT an error under 'auto' (the first
    attempt has nothing to resume), but an explicit path that does not exist is -- that is a
    typo, and silently starting from scratch would waste the whole job.
    """
    r = getattr(args, 'resume', None)
    if not r:
        return None
    if r != 'auto':
        if not os.path.isfile(r):
            raise FileNotFoundError(f"--resume {r} does not exist.")
        return r
    import glob
    cands = glob.glob(os.path.join(args.checkpoints, '*', 'last.pt'))
    if not cands:
        if is_main_process():
            print(f"[resume] auto: no last.pt under {args.checkpoints} -- starting from scratch.")
        return None
    return max(cands, key=os.path.getmtime)


def load_training_state(path, model, optimizer, lr_scheduler, gan, msm_predictor, msm_opt, device):
    """Restore what save_training_state wrote. Returns (start_epoch, best_loss, patience)."""
    # weights_only=False: this checkpoint intentionally carries optimizer/scheduler state, not
    # just tensors. It is a file we wrote ourselves, in our own scratch directory.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if lr_scheduler is not None and 'lr_scheduler' in ckpt:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    if gan is not None and 'gan_disc' in ckpt:
        gan['disc'].load_state_dict(ckpt['gan_disc'])
        gan['opt'].load_state_dict(ckpt['gan_opt'])
    if msm_predictor is not None and 'msm' in ckpt:
        msm_predictor.load_state_dict(ckpt['msm'])
        msm_opt.load_state_dict(ckpt['msm_opt'])
    return ckpt['epoch'] + 1, ckpt['best_loss'], ckpt['patience_counter']


def train_dualvae(args):
    # --- DDP bring-up ------------------------------------------------------------------------
    # No-op unless launched under torchrun, so `python main.py --config ...` still runs exactly
    # as before. Under torchrun this binds the process to its own GPU and overrides args.device,
    # which the whole trainer (and every .to(device)) reads.
    device, local_rank, rank, world = ddp_setup()
    args.device = device
    args.world_size = world
    # Per-rank seed offset: the ranks MUST differ in their stochastic paths (the VAE's
    # reparameterization noise, dead-code restart draws) or 4 GPUs would just average 4 copies
    # of the same noise. Model INITIALIZATION stays identical regardless -- DDP broadcasts
    # rank 0's parameters at construction.
    set_seed(args.seed + rank, args.deterministic, args.cudnn_benchmark)
    # TF32 matmuls: ~free throughput on H100 for the 1x1 bottleneck convs and attention, at a
    # precision that is irrelevant next to the bf16 autocast the forward already runs in.
    if getattr(args, 'allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Fail fast (configs can also reach this from a notebook, bypassing this trainer - the
    # model __init__ validates too, but we want the bad value caught before data/wandb setup).
    cont_dropout_p = getattr(args, 'cont_dropout_p', 0.0)
    validate_cont_dropout_p(cont_dropout_p)
    # Prepare logging & directories. Short unique run id; exact params saved as
    # config_used.yaml in the run dir and logged to wandb (vars(args)).
    # make_run_id() embeds a timestamp AND a random suffix, so every rank would invent a
    # DIFFERENT id and the ranks would scatter their checkpoints across 4 directories.
    # Rank 0's id is broadcast so all ranks agree on one run directory.
    resume_path = resolve_resume_path(args)
    if resume_path:
        # Continue INSIDE the original run directory rather than minting a new id, so a run
        # interrupted three times still leaves one coherent checkpoint dir and one config.
        checkpoint_dir = os.path.dirname(resume_path)
        model_name_ID = os.path.basename(checkpoint_dir)
        if is_main_process():
            print(f"[resume] continuing run {model_name_ID} from {resume_path}")
    else:
        model_name_ID = make_run_id(args.model)
        if is_dist():
            import torch.distributed as dist
            obj = [model_name_ID]
            dist.broadcast_object_list(obj, src=0)
            model_name_ID = obj[0]
        checkpoint_dir = os.path.join(args.checkpoints, model_name_ID)
    if is_main_process():
        create_directory(checkpoint_dir)
        save_config_copy(args, checkpoint_dir)
        if args.do_wandb:
            # One wandb run per JOB, not per rank: 4 processes calling wandb.init() would
            # create 4 runs logging quarter-batches against each other.
            setup_wandb(args, model_name_ID)
    barrier()   # nobody proceeds until the run directory exists

    # Prepare data & model
    #trainset, valset, testset, trainloader, valloader, testloader
    trainset, valset, testset, trainloader, valloader, testloader = prepare_data(args)
    model, optimizer = initialize_model(args)
    recon_criterion = build_recon_criterion(args)

    # Skipped when resuming: the codebook in last.pt is thousands of EMA steps past k-means,
    # and re-seeding it from a fresh batch would throw that away.
    if getattr(args, 'initialize_from_data', False) and not resume_path:
        vectors_per_img = (args.resize_img // args.downsample_factor) ** 2   # 32*32 = 1024
        target_vectors = 50 * args.num_embeddings                            # ~50 samples/centroid
        n_init = math.ceil(target_vectors / vectors_per_img)                 # = 13 for your config
        n_init = max(1, min(n_init, args.batch_size, 16))
        # k-means runs on ONE rank's batch and the result is broadcast. Letting every rank seed
        # from its own shard would produce 4 different codebooks, and the EMA path has no
        # gradient for DDP to reconcile them with -- they would stay different forever.
        if is_main_process():
            with torch.no_grad():
                init_batch = next(iter(trainloader))["image"][:n_init].to(args.device)
                z_e = model.encoder(init_batch.float())
                z_e_vq = model.bottle_neck_VQ(z_e)
                model.vq_layer.init_from_data(z_e_vq)
            del init_batch, z_e, z_e_vq
            torch.cuda.empty_cache()
            print("Codebook initialized from data via k-means.")
        # Copies both the centroids and the EMA statistics seeded alongside them
        # (ema_cluster_size / ema_embed_sum / ema_res_sq).
        broadcast_module_state_(model.vq_layer, src=0)

    # ---> ADD THE CHECK HERE <---
    if is_main_process():
        check_device_and_vram(model, trainloader, args.device)
    best_loss = float('inf')
    patience_counter = 0

    # GAN and LR schedule are built from the UNWRAPPED model (build_gan reaches into
    # model.decoder[-1].weight for the adaptive lambda), so wrap only after they exist.
    lr_scheduler = build_lr_scheduler(optimizer, args)
    gan = build_gan(args, model, args.device)
    if is_dist():
        # broadcast_buffers=False: the only buffers that matter are the EMA codebook
        # statistics, and embedding.py already keeps them identical on every rank by
        # all-reducing the batch statistics. Letting DDP re-broadcast them from rank 0 on
        # every forward would be a redundant collective in the hot loop.
        # device_ids must be omitted for a CPU (gloo) process group.
        dev_kw = dict(device_ids=[local_rank], output_device=local_rank) if device.type == 'cuda' else {}
        model = DDP(model, **dev_kw,
                    broadcast_buffers=False,
                    find_unused_parameters=getattr(args, 'ddp_find_unused_parameters', False))
        if is_main_process():
            eff_bs = args.batch_size * world
            print(f"[DDP] {world} ranks | per-GPU batch {args.batch_size} "
                  f"| effective batch {eff_bs} | lr {args.lr}")
    ci_aug = build_ci_augment(args, args.device)   # None unless code_invariance: true
    if ci_aug is not None:
        _bal = getattr(args, 'code_invariance_balance', False)
        _mode = getattr(args, 'code_invariance_mode', 'pooled')
        print(f"[CVI] cross-view code-invariance ON [mode={_mode}]: weight={getattr(args,'code_invariance_weight',0.5)}"
              f"{' (= target fraction of recon; balanced)' if _bal else ' (raw)'} "
              f"tau={getattr(args,'code_invariance_tau',0.5)} start_epoch={getattr(args,'code_invariance_start_epoch',8)}")
        if _mode == 'aligned':
            print(f"[CVI]   aligned: photometric-only aug, per-location match + me-max div "
                  f"(div_weight={getattr(args,'code_invariance_div_weight',1.0)}, "
                  f"target_tau={getattr(args,'code_invariance_target_tau',None)}). "
                  f"Watch Train/CVI Agreement (up), CVI Marginal Entropy (~5.5, no collapse).")
    msm_predictor, msm_opt = build_masked_predictor(args, args.device)  # None unless masked_modeling: true
    # Build FID/KID ONCE (not per epoch); set val_fid_device: cpu to keep it off the GPU.
    fid_bundle = build_val_fid(args, args.device)

    # Restore AFTER every component exists (optimizer, schedule, GAN, MSM head) so each one
    # gets its own state back. Loading into the unwrapped module keeps every rank consistent:
    # they all read the same file and DDP never has to reconcile them.
    start_epoch = 0
    if resume_path:
        start_epoch, best_loss, patience_counter = load_training_state(
            resume_path, model, optimizer, lr_scheduler, gan, msm_predictor, msm_opt, args.device)
        if is_main_process():
            print(f"[resume] restored through epoch {start_epoch - 1}; "
                  f"continuing at epoch {start_epoch}/{args.epochs} (best_loss={best_loss:.4f})")
        barrier()

    for epoch in range(start_epoch, args.epochs):
        # Without set_epoch the DistributedSampler replays ONE fixed permutation every epoch,
        # so each rank would see the same images in the same order for the whole run.
        if is_dist():
            trainloader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, args.kl_beta, recon_criterion, use_amp=args.use_amp, gan=gan,
                                        ci_aug=ci_aug, ci_weight=code_invariance_weight(args, epoch), ci_tau=getattr(args, 'code_invariance_tau', 0.5),
                                        ci_balance=getattr(args, 'code_invariance_balance', False),
                                        ci_sinkhorn=getattr(args, 'code_invariance_sinkhorn', False),
                                        ci_mode=getattr(args, 'code_invariance_mode', 'pooled'),
                                        ci_target_tau=getattr(args, 'code_invariance_target_tau', None),
                                        ci_div_weight=getattr(args, 'code_invariance_div_weight', 1.0),
                                        msm_predictor=msm_predictor, msm_opt=msm_opt,
                                        msm_type=getattr(args, 'masked_modeling_type', 'code'),
                                        msm_weight=masked_modeling_weight(args, epoch),
                                        msm_balance=getattr(args, 'msm_balance', True),
                                        msm_max_scale=getattr(args, 'msm_max_scale', 50.0),
                                        msm_mask_ratio=getattr(args, 'msm_mask_ratio', 0.5),
                                        msm_tau=getattr(args, 'msm_tau', 0.5),
                                        coarse_kl_beta=getattr(args, 'coarse_kl_beta', getattr(args, 'kl_beta', 0.001)),
                                        limit_train_batches=getattr(args, 'limit_train_batches', 0))
        run_fid = should_run_val_fid(args, epoch, args.epochs)
        epoch_fid = fid_bundle if run_fid else None
        reset_val_fid(epoch_fid)
        val_metrics = validate_one_epoch(model, valloader, args.device, args.kl_beta, args.dataset_name, recon_criterion, use_amp=args.use_amp, fid_bundle=epoch_fid)
        reset_val_fid(epoch_fid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Every rank only saw its own shard, so the raw per-rank averages describe a quarter of
        # the data. Average them across ranks before anything logs or checkpoints on them.
        # (FID/KID are excluded: torchmetrics already gathers its own state across ranks inside
        # compute(), so those entries are identical everywhere and re-averaging is a no-op.)
        train_metrics = all_reduce_metrics(train_metrics, args.device)
        val_metrics = all_reduce_metrics(val_metrics, args.device)

        # Record the LR actually used this epoch, THEN advance the schedule.
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        if is_main_process():
            # Peak VRAM is the number that decides whether batch_size can go up. Reported per
            # epoch and reset, so it reflects THIS epoch -- which matters because the GAN turns
            # on partway through the run and permanently raises the high-water mark.
            mem = ""
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
                total = torch.cuda.get_device_properties(args.device).total_memory / 1024 ** 3
                mem = f", PeakVRAM={peak:.1f}/{total:.0f}GiB"
                torch.cuda.reset_peak_memory_stats(args.device)
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}{mem}")
            if args.do_wandb:
                log_metrics(epoch, train_metrics, val_metrics, valset, model, args)

        # Save checkpoint if improved. best_loss/patience_counter are derived from the
        # all-reduced train loss, so every rank tracks the same value and agrees on the
        # early-stopping decision below; only rank 0 actually writes the file.
        prev_best = best_loss
        best_loss, patience_counter = save_checkpoint(model, epoch, best_loss, train_metrics["loss"], patience_counter, checkpoint_dir)
        # Keep the MSM predictor (needed to sample the generative prior in 'latent' mode) in sync
        # with best.pt, and always keep the latest for resuming.
        if msm_predictor is not None and is_main_process():
            torch.save(msm_predictor.state_dict(), os.path.join(checkpoint_dir, "masked_predictor_last.pt"))
            if best_loss < prev_best:
                torch.save(msm_predictor.state_dict(), os.path.join(checkpoint_dir, "masked_predictor_best.pt"))

        # Full resumable state, every epoch. This is what caps the damage from a wall-clock
        # kill at ONE epoch instead of the whole run.
        if is_main_process():
            save_training_state(os.path.join(checkpoint_dir, "last.pt"), epoch, model, optimizer,
                                lr_scheduler, gan, msm_predictor, msm_opt, best_loss, patience_counter)

        # Early stopping
        if args.do_early_stopping:
            if patience_counter >= args.patience:
                if is_main_process():
                    print("Early stopping triggered.")
                break
    filename = f"final_epoch.pt"
    path = os.path.join(checkpoint_dir, filename)
    if is_main_process():
        torch.save(unwrap(model).state_dict(), path)
        if gan is not None:
            torch.save(gan['disc'].state_dict(), os.path.join(checkpoint_dir, "final_epoch_disc.pt"))
        if msm_predictor is not None:
            torch.save(msm_predictor.state_dict(), os.path.join(checkpoint_dir, "masked_predictor_final.pt"))
        print(f"Final Checkpoint saved: {filename}")
        if args.do_wandb:
            wandb.finish()
    ddp_cleanup()
    return unwrap(model)
