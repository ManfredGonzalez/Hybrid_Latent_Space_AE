import os
import torch
import wandb
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, set_seed, setup_wandb, seed_worker, build_lr_scheduler, build_val_fid, update_val_fid, compute_val_fid, should_run_val_fid, reset_val_fid, make_run_id, save_config_copy
from tools.normalization import denormalize
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.vqvae import VQVAE
from losses.loss import vqvae_loss
from losses.reconstruction import build_reconstruction_criterion
from losses.gan import build_gan, generator_step_terms, discriminator_step
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.utils as vutils


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
    else:
        # Load CIFAR, MNIST, or Imagenette
        if dataset_name == "imagenette":
            # get_benchmark_dataset returns (train, val) for imagenette, ignoring split parameter
            trainset, valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, resize_img=args.resize_img, seed=args.seed)
        else:
            # Load CIFAR or MNIST
            trainset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='train', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='val', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,worker_init_fn=seed_worker,generator=generator)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,worker_init_fn=seed_worker,generator=generator)
    return trainset, valset, trainloader, valloader

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
    model = VQVAE(
        commitment_cost=args.commitment_cost,
        latent_channels=args.latent_channels,
        num_embeddings=args.num_embeddings,
        downsample_factor=args.downsample_factor,
        l2_normalize_codes=getattr(args, 'l2_normalize_codes', False),
        use_ema_codebook=getattr(args, 'use_ema_codebook', False),
        ema_decay=getattr(args, 'ema_decay', 0.99),
        ema_eps=getattr(args, 'ema_eps', 1e-5),
        ema_dead_threshold=getattr(args, 'ema_dead_threshold', 1.0),
        rq_depth=getattr(args, 'rq_depth', 1)
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def codebook_health_metrics(model):
    """Codebook-state diagnostics for wandb (read once per epoch, cheap).
    See experiments/train_dualvae.py for the meaning of each metric."""
    vq = model.vq_layer
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


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, recon_criterion, use_amp=False, gan=None):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "gan_g_loss": 0.0,
        "gan_d_loss": 0.0,
        "gan_d_weight": 0.0,
        "num_batches": 0,
    }

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_loss_val, commitment_loss, codebook_loss = model(images)
                loss, recon_loss, vq_loss_final, pixel_term, perceptual_term = vqvae_loss(
                    recon, images, vq_loss_val, recon_criterion=recon_criterion
                )

            # Optional VQGAN-style adversarial term (inactive before gan_start_epoch).
            gan_extra, g_loss_val, d_weight_val = generator_step_terms(gan, epoch, recon, recon_loss)
            loss = loss + gan_extra

            loss.backward()
            optimizer.step()

            # Discriminator update on (real, fake.detach()), after the generator step.
            d_loss_val = discriminator_step(gan, epoch, images, recon)

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += commitment_loss.item()
            running["codebook_loss"] += codebook_loss.item()
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
            running["perplexity"] += model.vq_layer.perplexity.item()
            running["codebook_usage"] += model.vq_layer.codebook_usage.item()
            running["gan_g_loss"] += g_loss_val
            running["gan_d_loss"] += d_loss_val
            running["gan_d_weight"] += d_weight_val
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0))

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}


def validate_one_epoch(model, loader, args, recon_criterion, fid_bundle=None):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "num_batches": 0,
    }
    # Initialize metrics with a data range of 1.0 (since your images are 0-1)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(args.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                recon, vq_loss_val, commitment_loss, codebook_loss = model(images)
                loss, recon_loss, vq_loss_final, pixel_term, perceptual_term = vqvae_loss(
                    recon, images, vq_loss_val, recon_criterion=recon_criterion
                )

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

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += commitment_loss.item()
            running["codebook_loss"] += codebook_loss.item()
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
            running["perplexity"] += model.vq_layer.perplexity.item()
            running["codebook_usage"] += model.vq_layer.codebook_usage.item()
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    out = {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}
    out.update(compute_val_fid(fid_bundle))  # adds 'rfid'/'kid_mean' when val_fid enabled
    return out

def reconstruct_grid(model, dataset, args, n_samples=8):
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    imgs = [dataset[i]["image"] for i in idxs]
    imgs = torch.tensor(np.stack(imgs)).to(args.device)

    with torch.no_grad():
        recon, _, _, _ = model(imgs)

    # Denormalize if needed (here assume already in [0,1])
    grid = vutils.make_grid(torch.cat([denormalize(imgs, args.dataset_name, args.device), denormalize(recon, args.dataset_name, args.device)], dim=0), nrow=n_samples, normalize=True, scale_each=True)
    return grid

def log_metrics(epoch, train_metrics, val_metrics, valset, model, args):
    recon_grid = reconstruct_grid(model, valset, args, n_samples=8)
    wandb.log({
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
        "Train/GAN G Loss": train_metrics.get("gan_g_loss", 0.0),
        "Train/GAN D Loss": train_metrics.get("gan_d_loss", 0.0),
        "Train/GAN D Weight": train_metrics.get("gan_d_weight", 0.0),
        "Train/Learning Rate": train_metrics.get("lr", args.lr),
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/VQ Loss": val_metrics["vq_loss"],
        "Val/Commitment Loss": val_metrics["commitment_loss"],
        "Val/Codebook Loss": val_metrics["codebook_loss"],
        "Val/Pixel Term": val_metrics["pixel_term"],
        "Val/Perceptual Term": val_metrics["perceptual_term"],
        "Val/Codebook Perplexity": val_metrics["perplexity"],
        "Val/Codebook Usage": val_metrics["codebook_usage"],
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
        **({"Val/rFID": val_metrics["rfid"]} if "rfid" in val_metrics else {}),
        **({"Val/KID Mean": val_metrics["kid_mean"]} if "kid_mean" in val_metrics else {}),
        # --- Codebook health (current state, once per epoch) ---
        **codebook_health_metrics(model),
    }, step=epoch)


def save_checkpoint(model, epoch, best_loss, current_loss, patience_counter, checkpoint_dir):
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        filename = f"best.pt"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Checkpoint saved: {filename}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")
    return best_loss, patience_counter


def train_vqvae(args):
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)
    # Prepare logging & directories
    fold = os.path.splitext(os.path.basename(args.path_test_ids))[0]
    # Short unique run id; exact params saved as config_used.yaml + logged to wandb.
    model_name_ID = make_run_id(args.model)
    checkpoint_dir = os.path.join(args.checkpoints, model_name_ID)
    create_directory(checkpoint_dir)
    save_config_copy(args, checkpoint_dir)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    # Prepare data & model
    trainset, valset, trainloader, valloader = prepare_data(args)
    model, optimizer = initialize_model(args)
    recon_criterion = build_recon_criterion(args)

    if getattr(args, 'initialize_from_data', False):
        vectors_per_img = (args.resize_img // args.downsample_factor) ** 2   # 32*32 = 1024
        target_vectors = 50 * args.num_embeddings                            # ~50 samples/centroid
        n_init = math.ceil(target_vectors / vectors_per_img)                 # = 13 for your config
        n_init = max(1, min(n_init, args.batch_size, 16))
        with torch.no_grad():
            init_batch = next(iter(trainloader))["image"][:n_init].to(args.device)
            z_e = model.encoder(init_batch.float())
            model.vq_layer.init_from_data(z_e)
        del init_batch, z_e
        torch.cuda.empty_cache()
        print("Codebook initialized from data via k-means.")

    best_loss = float('inf')
    patience_counter = 0

    lr_scheduler = build_lr_scheduler(optimizer, args)
    gan = build_gan(args, model, args.device)
    # Build FID/KID ONCE (not per epoch); set val_fid_device: cpu to keep it off the GPU.
    fid_bundle = build_val_fid(args, args.device)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, recon_criterion, use_amp=args.use_amp, gan=gan)
        run_fid = should_run_val_fid(args, epoch, args.epochs)
        epoch_fid = fid_bundle if run_fid else None
        reset_val_fid(epoch_fid)
        val_metrics = validate_one_epoch(model, valloader, args, recon_criterion, fid_bundle=epoch_fid)
        reset_val_fid(epoch_fid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Record the LR actually used this epoch, THEN advance the schedule.
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

        # Log image reconstruction and metrics
        if args.do_wandb:
            log_metrics(epoch, train_metrics, val_metrics, valset, model, args)

        # Save checkpoint if improved
        best_loss, patience_counter = save_checkpoint(model, epoch, best_loss, train_metrics["loss"], patience_counter, checkpoint_dir)

        # Early stopping
        if args.do_early_stopping:
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    filename = f"final_epoch.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), path)
    if gan is not None:
        torch.save(gan['disc'].state_dict(), os.path.join(checkpoint_dir, "final_epoch_disc.pt"))
    print(f"Final Checkpoint saved: {filename}")
    if args.do_wandb:
        wandb.finish()
    return model
