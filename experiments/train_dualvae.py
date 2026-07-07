import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, seed_worker, set_seed, setup_wandb, scale_ratio
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.dual_vae import DUALVAE
from losses.loss import dualvae_loss
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
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,worker_init_fn=seed_worker,generator=generator)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,worker_init_fn=seed_worker,generator=generator)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,worker_init_fn=seed_worker,generator=generator)
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
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    imgs = [dataset[i]["image"] for i in idxs]
    imgs = torch.tensor(np.stack(imgs)).to(args.device)

    with torch.no_grad():
        recon, _, _ = model(imgs)

    # Denormalize if needed (here assume already in [0,1])
    grid = vutils.make_grid(torch.cat([denormalize(imgs, args.dataset_name, args.device), denormalize(recon, args.dataset_name, args.device)], dim=0), nrow=n_samples, normalize=True, scale_each=True)
    return grid

def initialize_model(args):
    model = DUALVAE(
        commitment_cost=args.commitment_cost,
        latent_channels=args.latent_channels,
        num_embeddings=args.num_embeddings,
        downsample_factor=getattr(args, 'downsample_factor', 8),
        l2_normalize_codes=getattr(args, 'l2_normalize_codes', False)
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, beta_kl_loss, use_amp=False):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "scale_ratio": 0.0,
        "actual_mean_variance": 0.0,
        "num_batches": 0,
    }

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)
                vq_loss_val = vq_related_losses["vq_loss"]
                # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
                # it returns: return total_loss/b_size, recon_loss/b_size, vq_loss/b_size, kl_loss/b_size
                loss, recon_loss, vq_loss_final, kl_loss = dualvae_loss(recon, images, vq_loss_val, beta_kl_loss, vanilla_vae_related_losses["mean"], vanilla_vae_related_losses["log_variance"], reduction='sum')

            loss.backward()
            optimizer.step()

            batch_scale_ratio = scale_ratio(vq_related_losses["z_vq"], vanilla_vae_related_losses["z_vanilla_post"])
            raw_variance = torch.exp(vanilla_vae_related_losses["log_variance"]).mean()

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["perplexity"] += model.vq_layer.perplexity.item()
            running["codebook_usage"] += model.vq_layer.codebook_usage.item()
            running["scale_ratio"] += batch_scale_ratio.item()
            running["actual_mean_variance"] += raw_variance.item()
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0))

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

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

def validate_one_epoch(model, loader, device, beta_kl_loss, dataset_name, use_amp=False):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
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
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)
                # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
                # it returns: return total_loss/b_size, recon_loss/b_size, vq_loss/b_size, kl_loss/b_size
                loss, recon_loss, vq_loss_final, kl_loss = dualvae_loss(recon, images, vq_related_losses["vq_loss"], beta_kl_loss, mean=vanilla_vae_related_losses["mean"], logvar=vanilla_vae_related_losses["log_variance"], reduction='sum')

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

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["perplexity"] += model.vq_layer.perplexity.item()
            running["codebook_usage"] += model.vq_layer.codebook_usage.item()
            running["scale_ratio"] += batch_scale_ratio.item()
            running["actual_mean_variance"] += raw_variance.item()
            # Accumulate the batch metrics
            running["psnr"] += batch_psnr.item()
            running["ssim"] += batch_ssim.item()
            running["num_batches"] += 1

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

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
        "Train/Codebook Perplexity": train_metrics["perplexity"],
        "Train/Codebook Usage": train_metrics["codebook_usage"],
        "Train/Scale Ratio (VQ/Cont)": train_metrics["scale_ratio"],
        "Train/Actual Mean Variance": train_metrics["actual_mean_variance"],
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/VQ Loss": val_metrics["vq_loss"],
        "Val/Commitment Loss": val_metrics["commitment_loss"],
        "Val/Codebook Loss": val_metrics["codebook_loss"],
        "Val/KL Divergence": val_metrics["kl_loss"],
        "Val/Codebook Perplexity": val_metrics["perplexity"],
        "Val/Codebook Usage": val_metrics["codebook_usage"],
        "Val/Scale Ratio (VQ/Cont)": val_metrics["scale_ratio"],
        "Val/Actual Mean Variance": val_metrics["actual_mean_variance"],
        # --- Image Quality Metrics ---
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
    }, step=epoch)


def save_checkpoint(model, epoch, best_loss, current_loss, patience_counter, checkpoint_dir):
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        filename = "best.pt"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Checkpoint saved: {filename}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")
    return best_loss, patience_counter


def train_dualvae(args):
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)
    # Prepare logging & directories
    model_name_ID = f"Hybrid_VAE_LatentC_{args.latent_channels}@Commit_{args.commitment_cost}@NumEmb_{args.num_embeddings}betaKL@{args.kl_beta}@Downsample_{args.downsample_factor}"
    checkpoint_dir = os.path.join(args.checkpoints, model_name_ID)
    create_directory(checkpoint_dir)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    # Prepare data & model
    #trainset, valset, testset, trainloader, valloader, testloader
    trainset, valset, testset, trainloader, valloader, testloader = prepare_data(args)
    model, optimizer = initialize_model(args)

    if getattr(args, 'initialize_from_data', False):
        init_batch = next(iter(trainloader))["image"].to(args.device)
        with torch.no_grad():
            z_e = model.encoder(init_batch.float())
            z_e_vq = model.bottle_neck_VQ(z_e)
        model.vq_layer.init_from_data(z_e_vq)
        print("Codebook initialized from data via k-means.")

    # ---> ADD THE CHECK HERE <---
    check_device_and_vram(model, trainloader, args.device)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, args.kl_beta, use_amp=args.use_amp)
        val_metrics = validate_one_epoch(model, valloader, args.device, args.kl_beta, args.dataset_name, use_amp=args.use_amp)

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

        # Log image reconstruction and metrics
        # Grab two distinct images from the validation set
        img1 = torch.tensor(valset[0]['image'])
        img2 = torch.tensor(valset[1]['image']) # Assumes valset has at least 2 images
        
        # Stack them to create a batch of shape (2, Channels, Height, Width)
        test_images = torch.stack([img1, img2]).to(args.device)
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
