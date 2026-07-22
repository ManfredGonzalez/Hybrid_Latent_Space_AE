import os
import wandb
import torch
import numpy as np
import random

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)


def make_run_id(prefix="run"):
    """Short, unique, sortable run id: '<prefix>_<YYYYmmdd-HHMMSS>_<rand6>'.

    Replaces the old flag-encoded model_name_ID, which grew past the filesystem's 255-char
    limit as ablation flags accumulated. The timestamp keeps runs chronologically sortable;
    the random suffix guarantees two runs launched the same second don't collide. The actual
    hyperparameters are recorded by save_config_copy() (a config_used.yaml in the run dir)
    and by wandb (vars(args) is logged, so you can still filter/group by any flag)."""
    import datetime
    import uuid
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def save_config_copy(args, checkpoint_dir, filename="config_used.yaml"):
    """Save the exact config used for this run into its checkpoint dir, so the short run id
    can always be mapped back to its hyperparameters. Copies the source YAML when available;
    otherwise dumps vars(args) as a best-effort fallback."""
    import shutil
    dst = os.path.join(checkpoint_dir, filename)
    src = getattr(args, 'config', None)
    try:
        if src and os.path.exists(src):
            shutil.copy(src, dst)
            return
    except Exception as e:
        print(f"[config] could not copy {src}: {e}; dumping vars(args) instead.")
    try:
        import yaml
        serializable = {}
        for k, v in vars(args).items():
            serializable[k] = v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)
        with open(dst, 'w') as f:
            yaml.safe_dump(serializable, f, sort_keys=True)
    except Exception as e:
        print(f"[config] could not dump args to {dst}: {e}")

def setup_wandb(args, model_name_ID):
    """Login to Weights & Biases and initialize a new run."""

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=model_name_ID,
        config=vars(args),  # Include all args dynamically
    )

    # Epoch-level and per-step metrics are logged at different granularities, so each group gets
    # its own step field instead of relying on wandb's shared implicit step (every wandb.log()
    # call advances that counter regardless of which metrics it carries, so mixing an explicit
    # step=epoch with more frequent per-step calls makes the epoch value go "backwards").
    wandb.define_metric("epoch")
    wandb.define_metric("Train/*", step_metric="epoch")
    wandb.define_metric("Val/*", step_metric="epoch")
    wandb.define_metric("Codebook/*", step_metric="epoch")
    wandb.define_metric("Sample Reconstructions", step_metric="epoch")

    wandb.define_metric("train_step")
    wandb.define_metric("Train/Queue Fill Ratio", step_metric="train_step")

    return run

def build_val_fid(args, device):
    """Optional reconstruction-FID (+ KID) metric for the validation loop.

    BUILD ONCE before the epoch loop (not per epoch) and reset_val_fid() each time --
    each metric carries its own InceptionV3, and KID buffers ALL extracted features, so
    rebuilding every epoch churns GPU memory and can OOM a full card.

    Device: metrics live on `val_fid_device` (default = training device). Set it to 'cpu'
    to run FID/KID with ZERO added GPU memory -- the Inception forward then runs on CPU
    (slower, but the training GPU is untouched). update_val_fid moves batches accordingly.

    Returns {fid, kid, device} or None when disabled/unavailable.
    """
    if not getattr(args, 'val_fid', False):
        return None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        print("WARNING: val_fid=true but torchmetrics FID/KID unavailable "
              "(pip install torchmetrics torch-fidelity) - skipping.")
        return None
    fid_device = getattr(args, 'val_fid_device', device)
    # kid_subset_size must be <= number of val images; 100 is safe for Imagenette.
    return {
        'fid': FrechetInceptionDistance(normalize=True).to(fid_device),
        'kid': KernelInceptionDistance(subset_size=getattr(args, 'kid_subset_size', 100), normalize=True).to(fid_device),
        'device': fid_device,
    }


def should_run_val_fid(args, epoch, total_epochs):
    """Frequency gate: run FID/KID every `val_fid_every_n_epochs` epochs (default 1) and
    always on the final epoch. Cuts the per-epoch Inception cost when set > 1."""
    if not getattr(args, 'val_fid', False):
        return False
    n = max(1, getattr(args, 'val_fid_every_n_epochs', 1))
    return (epoch % n == 0) or (epoch == total_epochs - 1)


def reset_val_fid(fid_bundle):
    """Clear the metrics' accumulated feature buffers before a fresh val pass."""
    if fid_bundle is None:
        return
    fid_bundle['fid'].reset()
    fid_bundle['kid'].reset()


def update_val_fid(fid_bundle, images_01, recon_01):
    """Feed one batch of [0,1] reals + reconstructions into the FID/KID metrics.
    Moves tensors to the metric's device (e.g. cpu) so the training GPU is untouched
    when val_fid_device='cpu'. No-op when fid_bundle is None."""
    if fid_bundle is None:
        return
    dev = fid_bundle['device']
    real = images_01.float().to(dev)
    fake = recon_01.float().to(dev)
    fid_bundle['fid'].update(real, real=True)
    fid_bundle['fid'].update(fake, real=False)
    fid_bundle['kid'].update(real, real=True)
    fid_bundle['kid'].update(fake, real=False)


def compute_val_fid(fid_bundle):
    """Return {'rfid': float, 'kid_mean': float} after a validation pass, or {} when
    disabled/not-run this epoch."""
    if fid_bundle is None:
        return {}
    out = {'rfid': fid_bundle['fid'].compute().item()}
    kid_mean, _ = fid_bundle['kid'].compute()
    out['kid_mean'] = kid_mean.item()
    return out


def build_lr_scheduler(optimizer, args):
    """Optional per-epoch LR schedule: linear warmup then cosine decay.

    Controlled by config keys (all optional, defaults preserve the previous
    constant-LR behavior exactly):
      lr_schedule:      "constant" (default) or "cosine". Cosine = linear warmup for
                        lr_warmup_epochs, then cosine decay from lr down to
                        lr * lr_min_ratio over the remaining epochs (Huh et al. 2023,
                        Sec. 5.3: warmup + cosine notably improves VQ-network
                        convergence and codebook perplexity).
      lr_warmup_epochs: int, default 5.
      lr_min_ratio:     float, default 0.01 (final lr = 1% of base lr).

    Returns a torch LambdaLR scheduler (call .step() once per epoch, after the
    train/val epoch completes) or None for the constant schedule.
    """
    import math

    schedule = getattr(args, 'lr_schedule', 'constant')
    if schedule == 'constant':
        return None
    if schedule != 'cosine':
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {schedule!r}.")

    total_epochs = args.epochs
    warmup_epochs = getattr(args, 'lr_warmup_epochs', 5)
    min_ratio = getattr(args, 'lr_min_ratio', 0.01)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: epoch 0 trains at (1/warmup)*lr, reaching full lr at the
            # end of warmup. Protects the codebook early: EMA stats and k-means init
            # get a few gentle epochs before full-size encoder updates start moving
            # the embedding distribution (the covariate-shift failure mode).
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def select_device(cfg_device: str = "cuda") -> str:
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[Device] CUDA not available, falling back to CPU.")
    return device


def set_seed(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f"[Seed] {seed} | deterministic={deterministic} | cudnn.benchmark={torch.backends.cudnn.benchmark}")


def seed_worker(worker_id):
    # Ensure each worker has a different but reproducible seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scale_ratio(z_vq, z_cont, eps=1e-8):
    """Mean ||z_vq|| / ||z_cont|| per spatial position, averaged over batch and space.

    z_vq, z_cont: (B, C, H, W) tensors from the VQ and continuous branches, taken before
    they're combined. A ratio far from 1 flags a magnitude mismatch between the two
    branches (e.g. one dominating the residual add / cross-attention).
    """
    vq_norm = z_vq.norm(dim=1)
    cont_norm = z_cont.norm(dim=1)
    return (vq_norm / (cont_norm + eps)).mean()
