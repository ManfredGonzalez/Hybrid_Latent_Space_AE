import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, set_seed, setup_wandb, seed_worker
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.vqvae import VQVAE
from losses.loss import vqvae_loss
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

def denormalize(tensor, dataset_name, device):
    """Reverses Z-score normalization for visualization and metrics."""
    if dataset_name.lower() == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    if dataset_name.lower() == 'imagenette':
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # Pineapple and MNIST are already in [0, 1] range via Min-Max scaling
    return tensor

def initialize_model(args):
    model = VQVAE(
        commitment_cost=args.commitment_cost,
        embedding_dim=args.codebook_dim,
        num_embeddings=args.num_embeddings,
        downsample_factor=args.downsample_factor
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, use_amp=False):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "num_batches": 0,
    }

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_loss_val, commitment_loss, codebook_loss = model(images)
                loss, recon_loss, vq_loss_final = vqvae_loss(recon, images, vq_loss_val)
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += commitment_loss.item()
            running["codebook_loss"] += codebook_loss.item()
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0))

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}


def validate_one_epoch(model, loader, args):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
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
                loss, recon_loss, vq_loss_final = vqvae_loss(recon, images, vq_loss_val)

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

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += commitment_loss.item()
            running["codebook_loss"] += codebook_loss.item()
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

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
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/VQ Loss": val_metrics["vq_loss"],
        "Val/Commitment Loss": val_metrics["commitment_loss"],
        "Val/Codebook Loss": val_metrics["codebook_loss"],
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
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
    model_name_ID = f"VQVAE_Codebok_{args.codebook_dim}@Commit_{args.commitment_cost}@NumEmb_{args.num_embeddings}@Downsample_{args.downsample_factor}"
    checkpoint_dir = os.path.join(args.checkpoints, model_name_ID)
    create_directory(checkpoint_dir)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    # Prepare data & model
    trainset, valset, trainloader, valloader = prepare_data(args)
    model, optimizer = initialize_model(args)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, use_amp=args.use_amp)
        val_metrics = validate_one_epoch(model, valloader, args)

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
    print(f"Final Checkpoint saved: {filename}")
    if args.do_wandb:
        wandb.finish()
    return model
