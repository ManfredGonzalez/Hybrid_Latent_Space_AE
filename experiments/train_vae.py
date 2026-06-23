import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import *
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.vae import VAE
from losses.loss import vae_loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import torchvision.utils as vutils

# ---- helpers.py ----
def denormalize(tensor, dataset_name, device):
    """Reverses Z-score normalization for visualization and metrics."""
    if dataset_name.lower() == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # Pineapple and MNIST are already in [0, 1] range via Min-Max scaling
    return tensor

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
    model = VAE(downsample_factor=args.downsample_factor).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def train_step(model, dataloader, optimizer, device, beta_kl_loss):
    model.train()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0

    with tqdm(total=len(dataloader.dataset), desc="Training", unit='img') as pbar:
        for batch in dataloader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon, images, mu, logvar, kl_beta=beta_kl_loss)
            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            count += 1

            pbar.set_postfix(loss=loss_dict["total"].item())
            pbar.update(images.size(0))

    return total_loss / count, total_recon / count, total_kl / count

def validation_step(model, dataloader, args):
    model.eval()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0
    total_psnr, total_ssim = 0, 0

    # Initialize metrics with a data range of 1.0 (since your images are 0-1)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)

    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "num_batches": 0,
    }
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(args.device)
            recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon, images, mu, logvar, kl_beta=args.kl_beta)
            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()

            denorm_images = denormalize(images, args.dataset_name, args.device)
            denorm_recon = denormalize(recon, args.dataset_name, args.device)

            # Clamp after denormalization to ensure strict [0, 1] bounds for the metrics
            recon_clamped = denorm_recon.clamp(0, 1)
            images_clamped = denorm_images.clamp(0, 1)

            # Calculate metrics on the clean [0, 1] images
            batch_psnr = psnr_metric(recon_clamped, images_clamped)
            batch_ssim = ssim_metric(recon_clamped, images_clamped)

            running["loss"] += loss_dict["total"].item()
            running["recon_loss"] += loss_dict["reconstruction"].item()
            running["kl_loss"] += loss_dict["kl"].item()
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    return (
        running["loss"] / running["num_batches"],
        running["recon_loss"] / running["num_batches"],
        running["kl_loss"] / running["num_batches"],
        running["psnr"] / running["num_batches"],
        running["ssim"] / running["num_batches"],
    )

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

def log_metrics_to_wandb(epoch, train_losses, val_losses, recon_grid):
    train_loss, train_recon, train_kl = train_losses
    val_loss, val_recon, val_kl, val_psnr, val_ssim = val_losses

    wandb.log({
        "epoch": epoch,
        "Sample Reconstructions": wandb.Image(recon_grid, caption=f"Epoch {epoch}"),
        "train/total_loss": train_loss,
        "train/recon_loss": train_recon,
        "train/kl_loss": train_kl,
        "val/total_loss": val_loss,
        "val/recon_loss": val_recon,
        "val/kl_loss": val_kl,
        "val/psnr": val_psnr,
        "val/ssim": val_ssim,
    }, step=epoch)


def save_if_best_val(model, loss, best_loss, path, epoch):
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
    model_name_ID = f"VAE_betaKL@{args.kl_beta}@Downsample_{args.downsample_factor}"
    path_to_save_checkpoints = os.path.join(args.checkpoints, model_name_ID)
    create_directory(path_to_save_checkpoints)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_losses = train_step(model, trainloader, optimizer, device, args.kl_beta)
        val_losses = validation_step(model, valloader, args)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_losses[0]:.4f}, Recon={train_losses[1]:.4f}, KL={train_losses[2]:.4f} | "
            f"Val Loss={val_losses[0]:.4f}, Recon={val_losses[1]:.4f}, KL={val_losses[2]:.4f}, "
            f"PSNR={val_losses[3]:.2f}, SSIM={val_losses[4]:.3f}"
        )

        # Reconstruct and log
        if args.do_wandb:
            recon_grid = reconstruct_grid(model, valset, args, n_samples=8)
            log_metrics_to_wandb(epoch, train_losses, val_losses, recon_grid)

        # Checkpoint and early stopping
        best_val_loss, improved = save_if_best_val(model, val_losses[0], best_val_loss, path_to_save_checkpoints, epoch)
        patience_counter = 0 if improved else patience_counter + 1
    
        if args.do_early_stopping:
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    filename = f"final_epoch.pt"
    path = os.path.join(path_to_save_checkpoints, filename)
    torch.save(model.state_dict(), path)
    print(f"Final Checkpoint saved: {filename}")
    if args.do_wandb:
        wandb.finish()
    return model