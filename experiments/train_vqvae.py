import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, setup_wandb
from data.datasets import PineappleDataset
from models.vqvae import VQVAE
from losses.loss import vqvae_loss

def prepare_data(args):
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
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return trainset, valset, trainloader, valloader


def initialize_model(args):
    model = VQVAE(
        commitment_cost=args.commitment_cost,
        embedding_dim=args.codebook_dim,
        num_embeddings=args.num_embeddings
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
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


def validate_one_epoch(model, loader, device):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "num_batches": 0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            recon, vq_loss_val, commitment_loss, codebook_loss = model(images)
            loss, recon_loss, vq_loss_final = vqvae_loss(recon, images, vq_loss_val)

            running["loss"] += loss.item()
            running["recon_loss"] += recon_loss.item()
            running["vq_loss"] += vq_loss_final.item()
            running["commitment_loss"] += commitment_loss.item()
            running["codebook_loss"] += codebook_loss.item()
            running["num_batches"] += 1

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}


def log_metrics(epoch, train_metrics, val_metrics, test_image, model, device):
    model.eval()
    with torch.no_grad():
        test_image = test_image.unsqueeze(0).to(device)
        recon_img, _, _, _ = model(test_image)
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


def train_vqvae(args):
    # Prepare logging & directories
    fold = os.path.splitext(os.path.basename(args.path_test_ids))[0]
    checkpoint_dir = os.path.join(args.checkpoints,
                                  f"Codebok_{args.codebook_dim}@Fold_{fold}@Commit_{args.commitment_cost}@NumEmb_{args.num_embeddings}")
    create_directory(checkpoint_dir)
    setup_wandb(args)

    # Prepare data & model
    trainset, valset, trainloader, valloader = prepare_data(args)
    model, optimizer = initialize_model(args)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs)
        val_metrics = validate_one_epoch(model, valloader, args.device)

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

        # Log image reconstruction and metrics
        test_image = torch.tensor(valset[0]['image']).to(args.device)
        log_metrics(epoch, train_metrics, val_metrics, test_image, model, args.device)

        # Save checkpoint if improved
        best_loss, patience_counter = save_checkpoint(model, epoch, best_loss, train_metrics["loss"], patience_counter, checkpoint_dir)

        # Early stopping
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    wandb.finish()
    return model
