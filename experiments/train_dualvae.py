import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, setup_wandb
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.dual_vae import DUALVAE
from losses.loss import dualvae_loss

def prepare_data(args):
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
            trainset = get_benchmark_dataset(dataset_name, path='./datasets', split='train', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            valset = get_benchmark_dataset(dataset_name, path='./datasets', split='val', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
            testset = get_benchmark_dataset(dataset_name, path='./datasets', split='test', val_ratio=args.val_ratio, resize_img=args.resize_img, seed=args.seed)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
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

def initialize_model(args):
    model = DUALVAE(
        commitment_cost=args.commitment_cost,
        embedding_dim=args.codebook_dim,
        num_embeddings=args.num_embeddings
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, beta_kl_loss):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "num_batches": 0,
    }

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            recon, vq_related_losses, vanilla_vae_related_losses = model(images)
            vq_loss_val = vq_related_losses["vq_loss"]
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: return total_loss/b_size, recon_loss/b_size, vq_loss/b_size, kl_loss/b_size
            loss, recon_loss, vq_loss_final, kl_loss = dualvae_loss(recon, images, vq_loss_val, beta_kl_loss, vanilla_vae_related_losses["mean"], vanilla_vae_related_losses["log_variance"], reduction='sum')
            
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0))

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}


def validate_one_epoch(model, loader, device,beta_kl_loss):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "num_batches": 0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            recon, vq_related_losses, vanilla_vae_related_losses = model(images)
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: return total_loss/b_size, recon_loss/b_size, vq_loss/b_size, kl_loss/b_size
            loss, recon_loss, vq_loss_final, kl_loss = dualvae_loss(recon, images, vq_related_losses["vq_loss"], beta_kl_loss, mean=vanilla_vae_related_losses["mean"], logvar=vanilla_vae_related_losses["log_variance"], reduction='sum')
            

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += vq_related_losses["commitment_loss"].item()
            running["codebook_loss"] += vq_related_losses["codebook_loss"].item()
            running["kl_loss"] += kl_loss.item()
            running["num_batches"] += 1

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}


def log_metrics(epoch, train_metrics, val_metrics, test_image, model, device):
    model.eval()
    with torch.no_grad():
        test_image = test_image.unsqueeze(0).to(device)
        recon_img, _, _ = model(test_image)
    wandb.log({
        "Sample Reconstructed": wandb.Image(recon_img.clamp(0, 1)),
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
        "Val/KL Loss": val_metrics["kl_loss"],
    }, step=epoch)


def save_checkpoint(model, epoch, best_loss, current_loss, patience_counter, checkpoint_dir):
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        filename = f"weights_ck_{epoch}.pt"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Checkpoint saved: {filename}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")
    return best_loss, patience_counter


def train_dualvae(args):
    # Prepare logging & directories
    checkpoint_dir = os.path.join(args.checkpoints,
                                  f"Codebok_{args.codebook_dim}@Commit_{args.commitment_cost}@NumEmb_{args.num_embeddings}")
    create_directory(checkpoint_dir)
    if args.do_wandb:
        setup_wandb(args)

    # Prepare data & model
    #trainset, valset, testset, trainloader, valloader, testloader
    trainset, valset, testset, trainloader, valloader, testloader = prepare_data(args)
    model, optimizer = initialize_model(args)
    # ---> ADD THE CHECK HERE <---
    check_device_and_vram(model, trainloader, args.device)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, args.kl_beta)
        val_metrics = validate_one_epoch(model, valloader, args.device, args.kl_beta)

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

        # Log image reconstruction and metrics
        test_image = torch.tensor(valset[0]['image']).to(args.device)
        if args.do_wandb:
            log_metrics(epoch, train_metrics, val_metrics, test_image, model, args.device)

        # Save checkpoint if improved
        best_loss, patience_counter = save_checkpoint(model, epoch, best_loss, train_metrics["loss"], patience_counter, checkpoint_dir)

        # Early stopping
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    if args.do_wandb:
        wandb.finish()
    return model
