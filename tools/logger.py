import os
import wandb
import argparse




def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def setup_wandb(args):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "architecture": args.architecture,
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "beta_kl_loss": args.beta_kl_loss,
            "reconstruction_loss": "mse_pixel"
        },
    )
    return run


def setup_wandb_vqvae(args):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "architecture": args.architecture,
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "beta_kl_loss": args.beta_kl_loss,
            "reconstruction_loss": "mse_pixel"
        },
    )
    return run