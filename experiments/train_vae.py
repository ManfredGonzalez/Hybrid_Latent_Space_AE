import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import *
from tools.normalization import denormalize
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.vae import VAE
from losses.loss import vae_loss
from losses.reconstruction import build_reconstruction_criterion
from losses.gan import build_gan, generator_step_terms, discriminator_step
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import torchvision.utils as vutils

def get_dataloaders(args):
    generator = torch.Generator().manual_seed(args.seed)
    #trainset = PineappleDataset(train=True, val=False, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
    #valset = PineappleDataset(train=False, val=True, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
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
        # Load CIFAR, MNIST, or Imagenette
        if dataset_name == "imagenette":
            # get_benchmark_dataset returns (train, val) for imagenette, ignoring split parameter
            trainset, valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, resize_img=args.resize_img, seed=args.seed)
            testset = valset # Or a dedicated test split if available
        else:
            # Load CIFAR or MNIST
            trainset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='train', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            valset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='val', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            testset = get_benchmark_dataset(dataset_name, path=args.dataset_path, split='test', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

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
               gan=None, epoch=0, total_epochs=1):
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

    with tqdm(total=len(dataloader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in dataloader:
            images = batch["image"].to(device)
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
            pbar.update(images.size(0))

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
            images = batch["image"].to(args.device)
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
        recon, _, _ = model(sample_img)
        recon = recon.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0)) * 255
    return recon.astype(np.uint8)

def reconstruct_grid(model, dataset, args, n_samples=8):
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    imgs = [dataset[i]["image"] for i in idxs]
    imgs = torch.tensor(np.stack(imgs)).to(args.device)

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
    across the models being compared."""
    min_delta = 1e-6
    if loss < best_loss - min_delta:
        torch.save(model.state_dict(), os.path.join(path, f"best.pt"))
        print(f"Checkpoint saved at epoch {epoch}.")
        return loss, True
    else:
        print("No improvement in loss.")
        return best_loss, False

# ---- train_vae.py ----

def train_vae(args):
    # Device & seed setup
    device = select_device(args.device)
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)
    # Short unique run id + a config_used.yaml copy, matching train_dualvae/train_vqvae
    # (the old descriptive name collided whenever two runs shared beta/downsample/recon,
    # and it could not encode the GAN/latent settings anyway -- the config copy can).
    model_name_ID = make_run_id(args.model)
    path_to_save_checkpoints = os.path.join(args.checkpoints, model_name_ID)
    create_directory(path_to_save_checkpoints)
    save_config_copy(args, path_to_save_checkpoints)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)
    recon_criterion = build_recon_criterion(args)

    best_loss = float('inf')
    patience_counter = 0

    lr_scheduler = build_lr_scheduler(optimizer, args)
    gan = build_gan(args, model, device)
    # Build FID/KID ONCE (not per epoch); set val_fid_device: cpu to keep it off the GPU.
    fid_bundle = build_val_fid(args, device)

    for epoch in range(args.epochs):
        train_metrics = train_step(model, trainloader, optimizer, device, args.kl_beta, recon_criterion,
                                   use_amp=args.use_amp, gan=gan, epoch=epoch, total_epochs=args.epochs)
        run_fid = should_run_val_fid(args, epoch, args.epochs)
        epoch_fid = fid_bundle if run_fid else None
        reset_val_fid(epoch_fid)
        val_metrics = validation_step(model, valloader, args, recon_criterion, fid_bundle=epoch_fid)
        reset_val_fid(epoch_fid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Record the LR actually used this epoch, THEN advance the schedule.
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, Recon={train_metrics['recon_loss']:.4f}, KL={train_metrics['kl_loss']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f}, Recon={val_metrics['recon_loss']:.4f}, KL={val_metrics['kl_loss']:.4f}, "
            f"PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.3f}"
        )

        # Reconstruct and log
        if args.do_wandb:
            recon_grid = reconstruct_grid(model, valset, args, n_samples=8)
            log_metrics_to_wandb(epoch, train_metrics, val_metrics, recon_grid, args)

        # Checkpoint and early stopping (on TRAIN loss, as in the dualvae/vqvae trainers).
        best_loss, improved = save_if_best_val(model, train_metrics["loss"], best_loss, path_to_save_checkpoints, epoch)
        patience_counter = 0 if improved else patience_counter + 1

        if args.do_early_stopping:
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    filename = f"final_epoch.pt"
    path = os.path.join(path_to_save_checkpoints, filename)
    torch.save(model.state_dict(), path)
    if gan is not None:
        torch.save(gan['disc'].state_dict(), os.path.join(path_to_save_checkpoints, "final_epoch_disc.pt"))
    print(f"Final Checkpoint saved: {filename}")
    if args.do_wandb:
        wandb.finish()
    return model