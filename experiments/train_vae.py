import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.utils import (create_directory, set_seed, setup_wandb, seed_worker, build_lr_scheduler,
                         build_val_fid, update_val_fid, compute_val_fid, should_run_val_fid,
                         reset_val_fid, make_run_id, save_config_copy)
from tools.distributed import (ddp_setup, ddp_cleanup, is_dist, is_main_process,
                               world_size, unwrap, all_reduce_metrics, barrier)
from tools.normalization import denormalize
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.vae import VAE
from losses.loss import vae_loss
from losses.reconstruction import build_reconstruction_criterion
from losses.gan import build_gan, generator_step_terms, discriminator_step
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import torchvision.utils as vutils


def _maybe_subset(dataset, n, seed, what="val"):
    """Deterministically cap a split at `n` images (`n` <= 0 / None -> untouched).

    Same helper (and same seeded permutation) as experiments/train_dualvae.py and
    experiments/train_vqvae.py, so the vanilla-VAE baseline scores its FID/KID on the
    IDENTICAL subset of ImageNet val the DualVAE/VQVAE runs did -- which is what makes the
    three numbers comparable at all.
    """
    if not n or n <= 0 or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    if is_main_process():
        print(f"[data] {what}: using a fixed {n}-image subset of {len(dataset)} (seed {seed}).")
    return Subset(dataset, idx)


def get_dataloaders(args):
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
    else:
        # Load CIFAR, MNIST, Imagenette or ImageNet
        if dataset_name in ("imagenette", "imagenet"):
            # Both return (train, val) directly and ignore the `split` argument. For imagenet
            # `dataset_path` is the single .h5 from tools/build_imagenet_subset.py, read by
            # data.datasets.HDF5ImageDataset.
            trainset, valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, resize_img=args.resize_img, seed=args.seed)
        else:
            # Load CIFAR or MNIST
            trainset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='train', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='val', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)

    # Optional fixed val subset (see _maybe_subset): the headline cost saver on ImageNet.
    valset = _maybe_subset(valset, getattr(args, 'val_subset_size', 0), args.seed, "val")

    # --- DDP: shard each split across ranks -------------------------------------------------
    # Without a DistributedSampler every rank would iterate the SAME images, so a 4-GPU run
    # would do 4x the work of a 1-GPU run for exactly the same gradient. The sampler also owns
    # the shuffling (hence shuffle=False on the loader), and needs set_epoch() each epoch or
    # every epoch replays one identical permutation.
    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True) if is_dist() else None
    val_sampler = DistributedSampler(valset, shuffle=False, drop_last=False) if is_dist() else None

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

    return trainset, valset, trainloader, valloader

def setup_model_and_optimizer(args):
    # latent_channels defaults to 4 (the historical VAE width). Set it to match the
    # DUALVAE run you are comparing against -- otherwise the vanilla VAE competes with
    # half the latent capacity.
    model = VAE(downsample_factor=args.downsample_factor,
                latent_channels=getattr(args, 'latent_channels', 4)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def build_recon_criterion(args):
    return build_reconstruction_criterion(
        name=getattr(args, 'perceptual_loss', 'none'),
        device=args.device,
        perceptual_weight=getattr(args, 'perceptual_weight', 1.0),
        ffl_alpha=getattr(args, 'ffl_alpha', 1.0),
        dataset_name=args.dataset_name,
        perceptual_batch_fraction=getattr(args, 'perceptual_batch_fraction', 1.0),
    )

def train_step(model, dataloader, optimizer, device, beta_kl_loss, recon_criterion, use_amp=False,
               gan=None, epoch=0, total_epochs=1, limit_train_batches=0):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "gan_g_loss": 0.0,
        "gan_d_loss": 0.0,
        "gan_d_weight": 0.0,
        "num_batches": 0,
    }

    # limit_train_batches > 0 truncates the epoch to that many steps PER RANK (smoke config).
    # Every rank applies the SAME limit, so the ranks still agree on the step count.
    n_steps = len(dataloader) if not limit_train_batches else min(int(limit_train_batches), len(dataloader))
    # One progress bar (rank 0) counting GLOBAL images, so the ETA is the run's, not one shard's.
    with tqdm(total=n_steps * dataloader.batch_size * world_size(), desc=f'Epoch {epoch}/{total_epochs}',
              unit='img', disable=not is_main_process()) as pbar:
        for step, batch in enumerate(dataloader):
            if step >= n_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad()
            # Forward under bf16 autocast; losses in fp32 OUTSIDE it -- same convention as
            # train_dualvae.py/train_vqvae.py, so the objectives are computed at identical
            # precision across the models being compared (matters most for LPIPS, whose
            # VGG trunk otherwise runs in bf16 and returns a visibly quantized term).
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon.float(), images.float(), mu.float(), logvar.float(),
                                 kl_beta=beta_kl_loss, recon_criterion=recon_criterion)
            loss = loss_dict["total"]

            # Optional VQGAN-style adversarial term (inactive before gan_start_epoch; the
            # adaptive weight balances it against the reconstruction loss at the decoder's
            # last layer). Identical helper to the dualvae/vqvae trainers.
            gan_extra, g_loss_val, d_weight_val = generator_step_terms(gan, epoch, recon, loss_dict["reconstruction"])
            loss = loss + gan_extra

            loss.backward()
            optimizer.step()

            # Discriminator update on (real, fake.detach()), after the generator step.
            d_loss_val = discriminator_step(gan, epoch, images, recon)

            running["loss"] += loss.item()
            running["recon_loss"] += loss_dict["reconstruction"].item()
            running["kl_loss"] += loss_dict["kl"].item()
            running["pixel_term"] += loss_dict["pixel_term"].item()
            running["perceptual_term"] += loss_dict["perceptual_term"].item()
            running["gan_g_loss"] += g_loss_val
            running["gan_d_loss"] += d_loss_val
            running["gan_d_weight"] += d_weight_val
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            # * world_size(): the bar's total is GLOBAL images, but this rank only sees its
            # own shard, so it must advance by the global batch. Without the factor the
            # counter runs at 1/world_size of the total and the ETA is world_size x too
            # pessimistic. Matches train_dualvae.py/train_vqvae.py.
            pbar.update(images.size(0) * world_size())

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

def validation_step(model, dataloader, args, recon_criterion, fid_bundle=None):
    model.eval()

    # Initialize metrics with a data range of 1.0 (since your images are 0-1)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)

    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "num_batches": 0,
    }
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(args.device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                recon, mu, logvar = model(images)

            # Losses in fp32 outside autocast (see train_step).
            loss_dict = vae_loss(recon.float(), images.float(), mu.float(), logvar.float(),
                                 kl_beta=args.kl_beta, recon_criterion=recon_criterion)

            # Denormalize both targets and predictions back to [0, 1]; cast to fp32 first since
            # metrics/clamping are more reliable outside the autocast region.
            denorm_images = denormalize(images.float(), args.dataset_name, args.device)
            denorm_recon = denormalize(recon.float(), args.dataset_name, args.device)

            # Clamp after denormalization to ensure strict [0, 1] bounds for the metrics
            recon_clamped = denorm_recon.clamp(0, 1)
            images_clamped = denorm_images.clamp(0, 1)

            # Calculate metrics on the clean [0, 1] images
            batch_psnr = psnr_metric(recon_clamped, images_clamped)
            batch_ssim = ssim_metric(recon_clamped, images_clamped)
            update_val_fid(fid_bundle, images_clamped, recon_clamped)

            running["loss"] += loss_dict["total"].item()
            running["recon_loss"] += loss_dict["reconstruction"].item()
            running["kl_loss"] += loss_dict["kl"].item()
            running["pixel_term"] += loss_dict["pixel_term"].item()
            running["perceptual_term"] += loss_dict["perceptual_term"].item()
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    out = {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}
    out.update(compute_val_fid(fid_bundle))  # adds 'rfid'/'kid_mean' when val_fid enabled
    return out

def reconstruct_sample(model, dataset, device):
    sample_img = dataset[0]['image']
    sample_img = torch.tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = unwrap(model)(sample_img)
        recon = recon.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0)) * 255
    return recon.astype(np.uint8)

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

def log_metrics_to_wandb(epoch, train_metrics, val_metrics, recon_grid, args):
    # Key names kept identical to train_dualvae.py/train_vqvae.py so runs overlay
    # directly in wandb.
    wandb.log({
        "epoch": epoch,
        "Sample Reconstructions": wandb.Image(recon_grid, caption=f"Epoch {epoch}"),
        "Train/Total Loss": train_metrics["loss"],
        "Train/Reconstruction Loss": train_metrics["recon_loss"],
        "Train/Pixel Term": train_metrics["pixel_term"],
        "Train/Perceptual Term": train_metrics["perceptual_term"],
        "Train/KL Divergence": train_metrics["kl_loss"],
        "Train/GAN G Loss": train_metrics.get("gan_g_loss", 0.0),
        "Train/GAN D Loss": train_metrics.get("gan_d_loss", 0.0),
        "Train/GAN D Weight": train_metrics.get("gan_d_weight", 0.0),
        "Train/Learning Rate": train_metrics.get("lr", args.lr),
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/Pixel Term": val_metrics["pixel_term"],
        "Val/Perceptual Term": val_metrics["perceptual_term"],
        "Val/KL Divergence": val_metrics["kl_loss"],
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
        **({"Val/rFID": val_metrics["rfid"]} if "rfid" in val_metrics else {}),
        **({"Val/KID Mean": val_metrics["kid_mean"]} if "kid_mean" in val_metrics else {}),
    }, step=epoch)


def save_if_best_val(model, loss, best_loss, path, epoch):
    """Kept for the historical name; `loss` is now the TRAIN loss, matching the
    selection rule in train_dualvae.py/train_vqvae.py so best.pt means the same thing
    across the models being compared.

    `loss` must already be all-reduced across ranks, so every rank agrees on the returned
    best_loss (and therefore on the early-stopping decision); only rank 0 writes the file.
    """
    min_delta = 1e-6
    if loss < best_loss - min_delta:
        # unwrap() so the checkpoint holds plain `encoder.*` keys rather than DDP's
        # `module.encoder.*`, keeping 4-GPU checkpoints loadable by every existing
        # single-GPU consumer (inference, latent analysis) with no changes.
        if is_main_process():
            torch.save(unwrap(model).state_dict(), os.path.join(path, f"best.pt"))
            print(f"Checkpoint saved at epoch {epoch}.")
        return loss, True
    else:
        if is_main_process():
            print("No improvement in loss.")
        return best_loss, False


def save_training_state(path, epoch, model, optimizer, lr_scheduler, gan, best_loss, patience_counter):
    """Write the FULL training state so a killed job can pick up exactly where it stopped.

    best.pt / final_epoch.pt hold weights only, which is all inference needs but not enough to
    continue training: without the Adam moments, the cosine schedule position, the epoch counter
    and the discriminator's own optimizer state, a "resumed" run silently restarts the schedule
    and re-warms the critic.

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
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def resolve_resume_path(args):
    """Turn --resume into a concrete last.pt path, or None.

    'auto' scans the configured checkpoints directory for the most recently modified last.pt,
    which is what a requeued SLURM job wants: the same submit line works whether it is the
    first attempt or the fourth. A missing file is NOT an error under 'auto' (the first
    attempt has nothing to resume), but an explicit path that does not exist is.
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


def load_training_state(path, model, optimizer, lr_scheduler, gan, device):
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
    return ckpt['epoch'] + 1, ckpt['best_loss'], ckpt['patience_counter']

# ---- train_vae.py ----

def train_vae(args):
    # --- DDP bring-up ------------------------------------------------------------------------
    # No-op unless launched under torchrun, so `python main.py --config ...` still runs exactly
    # as before. Under torchrun this binds the process to its own GPU and overrides args.device.
    device, local_rank, rank, world = ddp_setup()
    args.device = device
    args.world_size = world
    # Per-rank seed offset: the ranks must differ in their stochastic paths (the posterior's
    # reparameterization noise, dataloader worker seeds). Model INITIALIZATION stays identical
    # regardless -- DDP broadcasts rank 0's parameters at construction.
    set_seed(args.seed + rank, args.deterministic, args.cudnn_benchmark)
    # TF32 matmuls: ~free throughput on H100, at a precision that is irrelevant next to the
    # bf16 autocast the forward already runs in.
    if getattr(args, 'allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Short unique run id + a config_used.yaml copy, matching train_dualvae/train_vqvae
    # (the old descriptive name collided whenever two runs shared beta/downsample/recon,
    # and it could not encode the GAN/latent settings anyway -- the config copy can).
    # make_run_id() embeds a timestamp AND a random suffix, so every rank would invent a
    # DIFFERENT id and scatter its checkpoints across 4 directories. Rank 0's id is broadcast.
    resume_path = resolve_resume_path(args)
    if resume_path:
        # Continue INSIDE the original run directory rather than minting a new id, so a run
        # interrupted three times still leaves one coherent checkpoint dir and one config.
        path_to_save_checkpoints = os.path.dirname(resume_path)
        model_name_ID = os.path.basename(path_to_save_checkpoints)
        if is_main_process():
            print(f"[resume] continuing run {model_name_ID} from {resume_path}")
    else:
        model_name_ID = make_run_id(args.model)
        if is_dist():
            import torch.distributed as dist
            obj = [model_name_ID]
            dist.broadcast_object_list(obj, src=0)
            model_name_ID = obj[0]
        path_to_save_checkpoints = os.path.join(args.checkpoints, model_name_ID)
    if is_main_process():
        create_directory(path_to_save_checkpoints)
        save_config_copy(args, path_to_save_checkpoints)
        if args.do_wandb:
            # One wandb run per JOB, not per rank: 4 processes calling wandb.init() would
            # create 4 runs logging quarter-batches against each other.
            setup_wandb(args, model_name_ID)
    barrier()   # nobody proceeds until the run directory exists

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)
    recon_criterion = build_recon_criterion(args)

    best_loss = float('inf')
    patience_counter = 0

    # GAN and LR schedule are built from the UNWRAPPED model (build_gan reaches into the
    # decoder's last layer for the adaptive lambda), so wrap only after they exist.
    lr_scheduler = build_lr_scheduler(optimizer, args)
    gan = build_gan(args, model, args.device)
    if is_dist():
        # broadcast_buffers=False: this model has no buffers whose value the forward depends
        # on, so the per-step buffer broadcast would be pure overhead. Same setting as the
        # dualvae/vqvae trainers.
        dev_kw = dict(device_ids=[local_rank], output_device=local_rank) if device.type == 'cuda' else {}
        model = DDP(model, **dev_kw,
                    broadcast_buffers=False,
                    find_unused_parameters=getattr(args, 'ddp_find_unused_parameters', False))
        if is_main_process():
            print(f"[DDP] {world} ranks | per-GPU batch {args.batch_size} "
                  f"| effective batch {args.batch_size * world} | lr {args.lr}")
    # Build FID/KID ONCE (not per epoch); set val_fid_device: cpu to keep it off the GPU.
    fid_bundle = build_val_fid(args, args.device)

    # Restore AFTER every component exists (optimizer, schedule, GAN) so each one gets its own
    # state back. Loading into the unwrapped module keeps every rank consistent.
    start_epoch = 0
    if resume_path:
        start_epoch, best_loss, patience_counter = load_training_state(
            resume_path, model, optimizer, lr_scheduler, gan, args.device)
        if is_main_process():
            print(f"[resume] restored through epoch {start_epoch - 1}; "
                  f"continuing at epoch {start_epoch}/{args.epochs} (best_loss={best_loss:.4f})")
        barrier()

    for epoch in range(start_epoch, args.epochs):
        # Without set_epoch the DistributedSampler replays ONE fixed permutation every epoch,
        # so each rank would see the same images in the same order for the whole run.
        if is_dist():
            trainloader.sampler.set_epoch(epoch)
        train_metrics = train_step(model, trainloader, optimizer, device, args.kl_beta, recon_criterion,
                                   use_amp=args.use_amp, gan=gan, epoch=epoch, total_epochs=args.epochs,
                                   limit_train_batches=getattr(args, 'limit_train_batches', 0))
        run_fid = should_run_val_fid(args, epoch, args.epochs)
        epoch_fid = fid_bundle if run_fid else None
        reset_val_fid(epoch_fid)
        val_metrics = validation_step(model, valloader, args, recon_criterion, fid_bundle=epoch_fid)
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
            print(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, Recon={train_metrics['recon_loss']:.4f}, KL={train_metrics['kl_loss']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f}, Recon={val_metrics['recon_loss']:.4f}, KL={val_metrics['kl_loss']:.4f}, "
                f"PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.3f}{mem}"
            )

            # Reconstruct and log
            if args.do_wandb:
                recon_grid = reconstruct_grid(model, valset, args, n_samples=8)
                log_metrics_to_wandb(epoch, train_metrics, val_metrics, recon_grid, args)

        # Checkpoint and early stopping (on the all-reduced TRAIN loss, as in the
        # dualvae/vqvae trainers).
        best_loss, improved = save_if_best_val(model, train_metrics["loss"], best_loss, path_to_save_checkpoints, epoch)
        patience_counter = 0 if improved else patience_counter + 1

        # Full resumable state, every epoch. This is what caps the damage from a wall-clock
        # kill at ONE epoch instead of the whole run.
        if is_main_process():
            save_training_state(os.path.join(path_to_save_checkpoints, "last.pt"), epoch, model,
                                optimizer, lr_scheduler, gan, best_loss, patience_counter)

        if args.do_early_stopping:
            if patience_counter >= args.patience:
                if is_main_process():
                    print("Early stopping triggered.")
                break
    filename = f"final_epoch.pt"
    path = os.path.join(path_to_save_checkpoints, filename)
    if is_main_process():
        torch.save(unwrap(model).state_dict(), path)
        if gan is not None:
            torch.save(gan['disc'].state_dict(), os.path.join(path_to_save_checkpoints, "final_epoch_disc.pt"))
        print(f"Final Checkpoint saved: {filename}")
        if args.do_wandb:
            wandb.finish()
    ddp_cleanup()
    return unwrap(model)
