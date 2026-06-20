import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, setup_wandb
from data.datasets import PineappleDataset
from models.vae_modules import VAE
from losses.loss import vae_perceptual_loss

# ---- helpers.py ----

def load_backbone_model(repository, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(repository, model_name)
    return model.to(device)

def get_dataloaders(args):
    trainset = PineappleDataset(train=True, val=False, train_ratio=args.train_ratio, path=args.dataset)
    valset = PineappleDataset(train=False, val=True, train_ratio=args.train_ratio, path=args.dataset)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainset, valset, trainloader, valloader

def setup_model_and_optimizer(args):
    model = VAE().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def train_step(model, dataloader, optimizer, device, beta_kl_loss, model_perceptual, layers_ids, mode):
    model.train()
    total_loss, total_recon, total_kl, total_perceptual, count = 0, 0, 0, 0, 0

    with tqdm(total=len(dataloader.dataset), desc="Training", unit='img') as pbar:
        for batch in dataloader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)

            loss_dict = vae_perceptual_loss(recon, images, mu, logvar, kl_beta=beta_kl_loss, 
                                 perceptual_loss=True, model_perceptual=model_perceptual, layers_ids=layers_ids, mode=mode)
            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            total_perceptual +=  loss_dict["perceptual"].item()
            count += 1

            pbar.set_postfix(loss=loss_dict["total"].item())
            pbar.update(images.size(0))

    return total_loss / count, total_recon / count, total_kl / count, total_perceptual / count

def validation_step(model, dataloader, device, beta_kl_loss, model_perceptual, layers_ids, mode):
    model.eval()
    total_loss, total_recon, total_kl, total_perceptual, count = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            recon, mu, logvar = model(images)

            loss_dict = vae_perceptual_loss(recon, images, mu, logvar, kl_beta=beta_kl_loss, 
                                 perceptual_loss=True, model_perceptual=model_perceptual, layers_ids=layers_ids, mode=mode)

            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            total_perceptual +=  loss_dict["perceptual"].item()
            count += 1

    return total_loss / count, total_recon / count, total_kl / count, total_perceptual / count

def reconstruct_sample(model, dataset, device):
    sample_img = dataset[0]['image']
    sample_img = torch.tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(sample_img)
        recon = recon.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0)) * 255
    return recon.astype(np.uint8)

def log_metrics_to_wandb(epoch, train_losses, val_losses, recon_img):
    train_loss, train_recon, train_kl, train_perceptual = train_losses
    val_loss, val_recon, val_kl, val_perceptual = val_losses

    wandb.log({
        "Sample Reconstructed": wandb.Image(recon_img, caption=f"Epoch {epoch}"),
        "epoch_train_loss": train_loss,
        "train_recon_loss": train_recon,
        "train_kl_loss": train_kl,
        "train_perceptual_loss": train_perceptual,
        "epoch_val_loss": val_loss,
        "val_recon_loss": val_recon,
        "val_kl_loss": val_kl,
        "val_perceptual_loss": val_perceptual
    }, step=epoch)

def save_checkpoint_if_best(model, loss, best_loss, path, epoch):
    if loss < best_loss:
        torch.save(model.state_dict(), os.path.join(path, f"weights_ck_{epoch}.pt"))
        print(f"Checkpoint saved at epoch {epoch}.")
        return loss, 0
    else:
        print("No improvement in loss.")
        return best_loss, 1

# ---- train_vae.py ----

def train_vae(args):
    path_to_save_checkpoints = os.path.join(args.checkpoints, f"betaKL@{args.beta_kl_loss}")
    create_directory(path_to_save_checkpoints)

    setup_wandb(args)

    model_perceptual = load_backbone_model(args.repository, args.model_name)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        train_losses = train_step(model, trainloader, optimizer, args.device, args.beta_kl_loss, model_perceptual, layers_ids=args.layers_ids, mode=args.mode)

        # Validation
        val_losses = validation_step(model, valloader, args.device, args.beta_kl_loss, model_perceptual, layers_ids=args.layers_ids, mode=args.mode)

        print(f"Epoch {epoch}: "
              f"Train Loss={train_losses[0]:.4f}, Recon={train_losses[1]:.4f}, KL={train_losses[2]:.4f}, Perceptual={train_losses[3]:.4f}, "
              f"Val Loss={val_losses[0]:.4f}, Recon={val_losses[1]:.4f}, KL={val_losses[2]:.4f}, Perceptual={val_losses[3]:.4f}")

        # Reconstruct and log
        recon_img = reconstruct_sample(model, valset, args.device)
        log_metrics_to_wandb(epoch, train_losses, val_losses, recon_img)

        # Checkpoint and early stopping
        best_loss, delta_patience = save_checkpoint_if_best(model, train_losses[0], best_loss, path_to_save_checkpoints, epoch)
        patience_counter += delta_patience

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    wandb.finish()
    return model