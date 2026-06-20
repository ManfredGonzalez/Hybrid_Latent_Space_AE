import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import *
from data.datasets import PineappleDataset
from models.vae import VAE
from losses.loss import vae_loss, psnr, ssim

import torchvision.utils as vutils

# ---- helpers.py ----

def get_dataloaders(args):
    generator = torch.Generator().manual_seed(args.seed)
    #trainset = PineappleDataset(train=True, val=False, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
    #valset = PineappleDataset(train=False, val=True, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
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

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return trainset, valset, trainloader, valloader

def setup_model_and_optimizer(args):
    model = VAE().to(args.device)
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

def validation_step(model, dataloader, device, kl_beta):
    model.eval()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0
    total_psnr, total_ssim = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon, images, mu, logvar, kl_beta=kl_beta)
            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()

            total_psnr += psnr(recon, images)
            total_ssim += ssim(recon, images)

            count += 1

    return (
        total_loss / count,
        total_recon / count,
        total_kl / count,
        total_psnr / count,
        total_ssim / count,
    )

def reconstruct_sample(model, dataset, device):
    sample_img = dataset[0]['image']
    sample_img = torch.tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(sample_img)
        recon = recon.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0)) * 255
    return recon.astype(np.uint8)

def reconstruct_grid(model, dataset, device, n_samples=8):
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    imgs = [dataset[i]["image"] for i in idxs]
    imgs = torch.tensor(np.stack(imgs)).to(device)

    with torch.no_grad():
        recon, _, _ = model(imgs)

    # Denormalize if needed (here assume already in [0,1])
    grid = vutils.make_grid(torch.cat([imgs, recon], dim=0), nrow=n_samples, normalize=True, scale_each=True)
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

    path_to_save_checkpoints = os.path.join(args.checkpoints, f"betaKL@{args.kl_beta}")
    create_directory(path_to_save_checkpoints)
    
    setup_wandb(args)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_losses = train_step(model, trainloader, optimizer, device, args.kl_beta)
        val_losses = validation_step(model, valloader, device, args.kl_beta)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_losses[0]:.4f}, Recon={train_losses[1]:.4f}, KL={train_losses[2]:.4f} | "
            f"Val Loss={val_losses[0]:.4f}, Recon={val_losses[1]:.4f}, KL={val_losses[2]:.4f}, "
            f"PSNR={val_losses[3]:.2f}, SSIM={val_losses[4]:.3f}"
        )

        # Reconstruct and log
        recon_grid = reconstruct_grid(model, valset, args.device, n_samples=8)
        log_metrics_to_wandb(epoch, train_losses, val_losses, recon_grid)

        # Checkpoint and early stopping
        best_val_loss, improved = save_if_best_val(model, val_losses[0], best_val_loss, path_to_save_checkpoints, epoch)
        patience_counter = 0 if improved else patience_counter + 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    wandb.finish()
    return model