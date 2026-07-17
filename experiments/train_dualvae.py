import os
import torch
import wandb
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import create_directory, seed_worker, set_seed, setup_wandb, scale_ratio, build_lr_scheduler
from tools.normalization import denormalize
from data.datasets import PineappleDataset, get_benchmark_dataset
from models.dual_vae import DUALVAE
from models.modules.cont_dropout import validate_cont_dropout_p
from losses.loss import dualvae_loss
from losses.reconstruction import build_reconstruction_criterion
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
    model = DUALVAE(
        commitment_cost=args.commitment_cost,
        latent_channels=args.latent_channels,
        num_embeddings=args.num_embeddings,
        downsample_factor=getattr(args, 'downsample_factor', 8),
        l2_normalize_codes=getattr(args, 'l2_normalize_codes', False),
        cont_dropout_p=getattr(args, 'cont_dropout_p', 0.0),
        use_ema_codebook=getattr(args, 'use_ema_codebook', False),
        ema_decay=getattr(args, 'ema_decay', 0.99),
        ema_eps=getattr(args, 'ema_eps', 1e-5),
        ema_dead_threshold=getattr(args, 'ema_dead_threshold', 1.0),
        rq_depth=getattr(args, 'rq_depth', 1),
        residual_continuous=getattr(args, 'residual_continuous', False),
        component_prior=getattr(args, 'component_prior', False),
        sigma2_floor=getattr(args, 'sigma2_floor', 1e-3),
        sigma2_ceil=getattr(args, 'sigma2_ceil', 10.0)
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def codebook_health_metrics(model):
    """Codebook-state diagnostics for wandb (read once per epoch, cheap).

    - Rel Quant Error: ||z_q - z||^2 / ||z||^2, scale-invariant (immune to the
      'encoder magnitude grew so sum-MSE looks worse' artifact).
    - Perplexity Depth d: assignment perplexity per RQ depth.
    - Pi Perplexity (EMA): exp(entropy of the EMA mixture weights pi_k) -- the
      long-horizon effective number of components (batch perplexity is one batch).
    - Restarted Codes: dead-code restarts in the last step (persistent churn here
      means the restart threshold is thrashing).
    - Sigma2 Mean/Min/Max + At-Floor Frac: distribution of the per-component prior
      variances; mass piling onto the floor is the collapse-ratchet signature.
    """
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


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, beta_kl_loss, recon_criterion, use_amp=False):
    model.train()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
        "perplexity": 0.0,
        "codebook_usage": 0.0,
        "scale_ratio": 0.0,
        "actual_mean_variance": 0.0,
        "cont_dropout_rate": 0.0,
        "num_batches": 0,
    }

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            # Forward pass under bf16 autocast (memory/speed); loss computation happens
            # OUTSIDE the autocast region in full fp32. The big sum-reductions (MSE over
            # every pixel, KL over every latent) lose real precision in bf16's 8-bit
            # mantissa, and .float() casts here are cheap. Gradients themselves already
            # accumulate in fp32 (params are fp32; autocast only affects op compute dtype).
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)

            vq_loss_val = vq_related_losses["vq_loss"].float()
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: total/b, recon/b, vq/b, kl/b, pixel_term/b, perceptual_term/b
            # keep_mask restricts the KL to samples whose continuous branch reached the
            # decoder this step (None when cont_dropout_p == 0 -> unmasked, old behavior).
            loss, recon_loss, vq_loss_final, kl_loss, pixel_term, perceptual_term = dualvae_loss(
                recon.float(), images.float(), vq_loss_val, beta_kl_loss,
                vanilla_vae_related_losses["mean"].float(), vanilla_vae_related_losses["log_variance"].float(),
                reduction='sum', recon_criterion=recon_criterion,
                keep_mask=vanilla_vae_related_losses["keep_mask"],
                prior_var=vanilla_vae_related_losses["prior_var"]
            )

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
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
            running["perplexity"] += model.vq_layer.perplexity.item()
            running["codebook_usage"] += model.vq_layer.codebook_usage.item()
            running["scale_ratio"] += batch_scale_ratio.item()
            running["actual_mean_variance"] += raw_variance.item()
            running["cont_dropout_rate"] += model.last_drop_fraction
            running["num_batches"] += 1

            pbar.set_postfix(loss=loss.item())
            pbar.update(images.size(0))

    return {k: v / running["num_batches"] for k, v in running.items() if k != "num_batches"}

def validate_one_epoch(model, loader, device, beta_kl_loss, dataset_name, recon_criterion, use_amp=False):
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "kl_loss": 0.0,
        "pixel_term": 0.0,
        "perceptual_term": 0.0,
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
            # Same split as training: bf16 forward, fp32 loss math. In eval mode
            # keep_mask is always None (dropout inactive), so the KL is unmasked.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                recon, vq_related_losses, vanilla_vae_related_losses = model(images)
            # dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum')
            # it returns: total/b, recon/b, vq/b, kl/b, pixel_term/b, perceptual_term/b
            loss, recon_loss, vq_loss_final, kl_loss, pixel_term, perceptual_term = dualvae_loss(
                recon.float(), images.float(), vq_related_losses["vq_loss"].float(), beta_kl_loss,
                mean=vanilla_vae_related_losses["mean"].float(), logvar=vanilla_vae_related_losses["log_variance"].float(),
                reduction='sum', recon_criterion=recon_criterion,
                keep_mask=vanilla_vae_related_losses["keep_mask"],
                prior_var=vanilla_vae_related_losses["prior_var"]
            )

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
            running["pixel_term"] += pixel_term.item()
            running["perceptual_term"] += perceptual_term.item()
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
        "Train/Pixel Term": train_metrics["pixel_term"],
        "Train/Perceptual Term": train_metrics["perceptual_term"],
        "Train/Codebook Perplexity": train_metrics["perplexity"],
        "Train/Codebook Usage": train_metrics["codebook_usage"],
        "Train/Scale Ratio (VQ/Cont)": train_metrics["scale_ratio"],
        "Train/Actual Mean Variance": train_metrics["actual_mean_variance"],
        "Train/Cont Dropout Rate": train_metrics["cont_dropout_rate"],
        "Train/Learning Rate": train_metrics.get("lr", args.lr),
        "Val/Total Loss": val_metrics["loss"],
        "Val/Reconstruction Loss": val_metrics["recon_loss"],
        "Val/VQ Loss": val_metrics["vq_loss"],
        "Val/Commitment Loss": val_metrics["commitment_loss"],
        "Val/Codebook Loss": val_metrics["codebook_loss"],
        "Val/Pixel Term": val_metrics["pixel_term"],
        "Val/Perceptual Term": val_metrics["perceptual_term"],
        "Val/KL Divergence": val_metrics["kl_loss"],
        "Val/Codebook Perplexity": val_metrics["perplexity"],
        "Val/Codebook Usage": val_metrics["codebook_usage"],
        "Val/Scale Ratio (VQ/Cont)": val_metrics["scale_ratio"],
        "Val/Actual Mean Variance": val_metrics["actual_mean_variance"],
        # --- Image Quality Metrics ---
        "Val/PSNR": val_metrics["psnr"],
        "Val/SSIM": val_metrics["ssim"],
        # --- Codebook / GMM health (current state, once per epoch) ---
        **codebook_health_metrics(model),
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
    # Fail fast (configs can also reach this from a notebook, bypassing this trainer - the
    # model __init__ validates too, but we want the bad value caught before data/wandb setup).
    cont_dropout_p = getattr(args, 'cont_dropout_p', 0.0)
    validate_cont_dropout_p(cont_dropout_p)
    # Prepare logging & directories
    perceptual_loss_name = getattr(args, 'perceptual_loss', 'none')
    use_ema_codebook = getattr(args, 'use_ema_codebook', False)
    rq_depth = getattr(args, 'rq_depth', 1)
    residual_continuous = getattr(args, 'residual_continuous', False)
    component_prior = getattr(args, 'component_prior', False)
    model_name_ID = f"Hybrid_VAE_LatentC_{args.latent_channels}@Commit_{args.commitment_cost}@NumEmb_{args.num_embeddings}betaKL@{args.kl_beta}@Downsample_{args.downsample_factor}@ContDrop_{cont_dropout_p}@Recon_{perceptual_loss_name}@EMA_{use_ema_codebook}@RQ_{rq_depth}@ResCont_{residual_continuous}@CompPrior_{component_prior}"
    checkpoint_dir = os.path.join(args.checkpoints, model_name_ID)
    create_directory(checkpoint_dir)
    if args.do_wandb:
        setup_wandb(args, model_name_ID)

    # Prepare data & model
    #trainset, valset, testset, trainloader, valloader, testloader
    trainset, valset, testset, trainloader, valloader, testloader = prepare_data(args)
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
            z_e_vq = model.bottle_neck_VQ(z_e)
            model.vq_layer.init_from_data(z_e_vq)
        del init_batch, z_e, z_e_vq
        torch.cuda.empty_cache()
        print("Codebook initialized from data via k-means.")

    # ---> ADD THE CHECK HERE <---
    check_device_and_vram(model, trainloader, args.device)
    best_loss = float('inf')
    patience_counter = 0

    lr_scheduler = build_lr_scheduler(optimizer, args)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, trainloader, optimizer, args.device, epoch, args.epochs, args.kl_beta, recon_criterion, use_amp=args.use_amp)
        val_metrics = validate_one_epoch(model, valloader, args.device, args.kl_beta, args.dataset_name, recon_criterion, use_amp=args.use_amp)

        # Record the LR actually used this epoch, THEN advance the schedule.
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

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
