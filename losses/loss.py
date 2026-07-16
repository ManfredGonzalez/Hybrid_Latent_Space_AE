import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from typing import Optional, List, Dict

def convnext_perceptual_loss(    
    x_real,
    x_recon,
    convnext_loss
):
    loss_value = convnext_loss(x_recon, x_real)
    return loss_value
    
    

def dino_perceptual_loss(
    x_real,
    x_recon,
    dino_model,
    layer_ids=[11],
    mode='cls',         # 'cls', 'mean', or 'tokens'
    reduction='mean'    # or 'none'
):
    """
    Compute perceptual loss between x_real and x_recon using DINO ViT features.

    Args:
        x_real (Tensor): Original image batch [B, 3, H, W]
        x_recon (Tensor): Reconstructed image batch [B, 3, H, W]
        dino_model (nn.Module): DINO ViT model with get_intermediate_layers
        layer_ids (list[int]): Layer indices to use for perceptual comparison
        mode (str): 'cls' | 'mean' | 'tokens'
        reduction (str): 'mean' | 'sum' | 'none'

    Returns:
        Tensor: Scalar loss (or per-sample if reduction='none')
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_real.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_real.device).view(1, 3, 1, 1)

    x_real = (x_real - mean) / std
    x_recon = (x_recon - mean) / std

    # Get intermediate layers
    feats_recon = dino_model.get_intermediate_layers(x_recon, n=len(dino_model.blocks)+1)

    with torch.no_grad():
        feats_real = dino_model.get_intermediate_layers(x_real, n=len(dino_model.blocks)+1)

    loss = 0.0

    for layer in layer_ids:
        f_real = feats_real[layer]  # [B, T, D]
        f_recon = feats_recon[layer]

        if mode == 'cls':
            v_real = f_real[:, 0]    # CLS token
            v_recon = f_recon[:, 0]

        elif mode == 'mean':
            v_real = f_real.mean(dim=1)
            v_recon = f_recon.mean(dim=1)

        elif mode == 'tokens':
            v_real = f_real
            v_recon = f_recon

        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss += F.mse_loss(v_real, v_recon, reduction=reduction)

    return loss #/ len(layer_ids)

def mse_loss(reconstructed, original, reduction='sum'):
    """
    Computes MSE reconstruction loss.

    Args:
        reconstructed: Reconstructed image tensor [B, C, H, W]
        original: Original image tensor [B, C, H, W]
        reduction: 'sum' or 'mean' for loss aggregation

    Returns:
        MSE loss (float)
    """
    loss_fn = nn.MSELoss(reduction=reduction)
    return loss_fn(reconstructed, original)


def _recon_terms(reconstructed, original, recon_criterion, reduction):
    """(recon_loss, pixel_term, extra_term) in the same reduction convention as `reduction`.

    reduction='sum' matches the historical nn.MSELoss(reduction='sum') convention used by
    vae_loss/dualvae_loss/vqvae_loss (sum over all elements; the caller divides by batch size
    separately). reduction='mean' matches sw_dualvae_loss's per-element mean convention, which
    is exactly what a ReconstructionCriterion already returns. When recon_criterion is None,
    falls back to plain MSE so existing callers are unaffected.
    """
    if recon_criterion is None:
        recon_loss = mse_loss(reconstructed, original, reduction=reduction)
        zero = torch.zeros((), device=reconstructed.device, dtype=recon_loss.dtype)
        return recon_loss, recon_loss, zero

    total_recon_loss, pixel_term, extra_term = recon_criterion(reconstructed, original)
    if reduction == 'sum':
        scale = reconstructed.numel()
        return total_recon_loss * scale, pixel_term * scale, extra_term * scale
    return total_recon_loss, pixel_term, extra_term


def kl_divergence_loss(mean, logvar, reduction='sum', keep_mask=None, prior_var=None):
    """
    Computes the KL divergence between the latent posterior and its prior.

    Args:
        mean: Mean of latent distribution [B, latent_dim] or [B, C, H, W]
        logvar: Log variance of latent distribution, same shape as mean
        reduction: 'sum' or 'mean' for loss aggregation
        keep_mask: Optional (B, 1, 1, 1) or (B,) 0/1 tensor from apply_cont_dropout.
            When given, samples with mask 0 (continuous branch dropped for this step)
            contribute zero KL: they received no reconstruction gradient through the
            continuous branch, so they should not receive prior-shrinkage gradient
            either. Kept samples keep exactly the same per-sample gradient scale as
            the unmasked case (no re-normalization by the kept count -- rescaling
            would push the effective beta on kept samples back up and defeat the
            purpose). None means no masking (identical to previous behavior).
        prior_var: Optional per-location prior variance, broadcastable to mean's shape
            (e.g. (B, 1, H, W) from the per-component GMM prior sigma_k^2). The KL is
            then computed against N(0, prior_var) instead of N(0, I):
                KL = 0.5 * [log s2 - logvar + (var + mu^2)/s2 - 1]
            Should be detached (EMA-estimated, not gradient-trained). prior_var=None
            (or all-ones) is identical to the standard-normal prior.

    Returns:
        KL divergence loss (float)
    """
    if prior_var is None:
        kl_elementwise = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    else:
        log_pv = torch.log(prior_var)
        kl_elementwise = 0.5 * (log_pv - logvar + (logvar.exp() + mean.pow(2)) / prior_var - 1.0)
    # Per-sample KL: sum over all non-batch dims.
    kl_per_sample = torch.sum(kl_elementwise, dim=list(range(1, mean.dim())))
    if keep_mask is not None:
        kl_per_sample = kl_per_sample * keep_mask.reshape(-1).to(kl_per_sample.dtype)
    kl = kl_per_sample.sum()
    if reduction == 'mean':
        return kl / mean.size(0)
    return kl




def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float = 0.1,
    reduction: str = 'sum',
    recon_criterion=None,
) -> Dict[str, torch.Tensor]:
    """
    Computes the VAE loss: reconstruction + β * KL divergence.

    Args:
        reconstructed (Tensor): Reconstructed images [B, C, H, W]
        original (Tensor): Original images [B, C, H, W]
        mean (Tensor): Latent means [B, latent_dim]
        logvar (Tensor): Latent log-variances [B, latent_dim]
        kl_beta (float): Scaling factor for KL divergence.
        reduction (str): Reduction method: 'sum' or 'mean'.
        recon_criterion: Optional callable from
            losses.reconstruction.build_reconstruction_criterion. When None, falls back to
            plain MSE (bit-identical to the pre-existing behavior).

    Returns:
        Dict[str, Tensor]: Dictionary with keys 'total', 'reconstruction', 'kl',
            'pixel_term', 'perceptual_term'.
    """
    batch_size = reconstructed.size(0)

    # Core losses
    recon_loss, pixel_term, extra_term = _recon_terms(reconstructed, original, recon_criterion, reduction)
    kl_loss = kl_divergence_loss(mean, logvar, reduction=reduction)
    total_loss = recon_loss + kl_beta * kl_loss

    # Normalize all by batch size
    loss_dict = {
        "total": total_loss / batch_size,
        "reconstruction": recon_loss / batch_size,
        "kl": kl_loss / batch_size,
        "pixel_term": pixel_term / batch_size,
        "perceptual_term": extra_term / batch_size,
    }

    return loss_dict

def sw_dualvae_loss(recon_x, x, vq_loss, z_vanilla_post, logvar, swd_criterion, recon_criterion=None, keep_mask=None, prior_var=None):

    # recon_loss, vq_loss, and swd_criterion's outputs are all already
    # mean-normalized (per-element), so no additional batch-size division here.
    recon_loss, pixel_term, extra_term = _recon_terms(recon_x, x, recon_criterion, reduction='mean')
    # Per-component GMM prior: whiten Delta by sigma_k per location, so the SWD /
    # variance-budget terms match the WHITENED offset against N(0, I) -- which is
    # exactly what the offsets should look like if the mixture story holds. prior_var
    # should be detached ((B, 1, H, W), broadcasting over channels).
    if prior_var is not None:
        prior_std = prior_var.sqrt()
        z_vanilla_post = z_vanilla_post / prior_std
        if logvar is not None:
            logvar = logvar - torch.log(prior_var)
    # Continuous regularization (SWD shape + variance budget). With a dropout
    # keep_mask, only the samples whose continuous branch actually reached the
    # decoder are regularized -- dropped samples got no reconstruction gradient
    # through that branch, so they shouldn't get shrinkage gradient either.
    # (This also keeps the SWD queue populated only with latents that were used.)
    if keep_mask is not None:
        kept = keep_mask.reshape(-1).bool()
        if kept.any():
            z_reg = z_vanilla_post[kept]
            logvar_reg = logvar[kept] if logvar is not None else None
            cont_reg_loss, swd_loss, var_loss = swd_criterion(z_reg, logvar_reg)
        else:
            # Whole batch dropped (rare at moderate p): skip the regularizer entirely.
            zero = torch.zeros((), device=recon_x.device, dtype=recon_loss.dtype)
            cont_reg_loss, swd_loss, var_loss = zero, zero, zero
    else:
        cont_reg_loss, swd_loss, var_loss = swd_criterion(z_vanilla_post, logvar)

    total_loss = recon_loss + vq_loss + cont_reg_loss

    return total_loss, recon_loss, vq_loss, cont_reg_loss, swd_loss, var_loss, pixel_term, extra_term

def dualvae_loss(recon_x, x, vq_loss, kl_beta, mean, logvar, reduction: str = 'sum', recon_criterion=None, keep_mask=None, prior_var=None):
    b_size = recon_x.size(0)
    recon_loss, pixel_term, extra_term = _recon_terms(recon_x, x, recon_criterion, reduction)
    kl_loss = kl_divergence_loss(mean, logvar, reduction=reduction, keep_mask=keep_mask, prior_var=prior_var)

    total_loss = recon_loss + vq_loss + kl_beta * kl_loss

    return (total_loss / b_size, recon_loss / b_size, vq_loss / b_size, kl_loss / b_size,
            pixel_term / b_size, extra_term / b_size)

def vae_perceptual_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float = 0.1,
    reduction: str = 'sum',
    perceptual_loss: bool = False,
    model_perceptual: Optional[torch.nn.Module] = None,
    layers_ids: Optional[List[int]] = None,
    mode: str = 'cls', model_name: str= "dino"
) -> Dict[str, torch.Tensor]:
    """
    Computes the VAE loss: MSE + β * KL divergence + optional perceptual loss.

    Args:
        reconstructed (Tensor): Reconstructed images [B, C, H, W]
        original (Tensor): Original images [B, C, H, W]
        mean (Tensor): Latent means [B, latent_dim]
        logvar (Tensor): Latent log-variances [B, latent_dim]
        kl_beta (float): Scaling factor for KL divergence.
        reduction (str): Reduction method: 'sum' or 'mean'.
        perceptual_loss (bool): Whether to include perceptual loss.
        model_perceptual (nn.Module, optional): DINO model for perceptual loss.
        layers_ids (List[int], optional): List of layer indices.
        mode (str): Feature mode: 'cls', 'mean', or 'tokens'.

    Returns:
        Dict[str, Tensor]: Dictionary with keys:
            - 'total'
            - 'reconstruction'
            - 'kl'
            - 'perceptual' (only if enabled)
    """
    batch_size = reconstructed.size(0)

    # Core losses
    recon_loss = mse_loss(reconstructed, original, reduction=reduction)
    kl_loss = kl_divergence_loss(mean, logvar, reduction=reduction)
    total_loss = recon_loss + kl_beta * kl_loss

    # Optional perceptual loss
    perceptual = None
    if perceptual_loss:
        if model_perceptual is None or layers_ids is None:
            raise ValueError("Perceptual loss is enabled but model_perceptual or layers_ids is not provided.")
        if model_name == "dino" or model_name == "dinov2":
            perceptual = dino_perceptual_loss(
                original,
                reconstructed,
                model_perceptual,
                layer_ids=layers_ids,
                mode=mode,
                reduction=reduction
            )
            total_loss += perceptual
        elif model_name == "convnext":
            perceptual = convnext_perceptual_loss(
                original,
                reconstructed,
                model_perceptual
            )
            total_loss += perceptual

    # Normalize all by batch size
    loss_dict = {
        "total": total_loss / batch_size,
        "reconstruction": recon_loss / batch_size,
        "kl": kl_loss / batch_size,
    }

    if perceptual is not None:
        loss_dict["perceptual"] = perceptual / batch_size

    return loss_dict


def vqvae_loss(recon_x, x, vq_loss, recon_criterion=None):
    b_size = recon_x.size(0)
    recon_loss, pixel_term, extra_term = _recon_terms(recon_x, x, recon_criterion, reduction='sum')
    total_loss = recon_loss + vq_loss

    return (total_loss / b_size, recon_loss / b_size, vq_loss / b_size,
            pixel_term / b_size, extra_term / b_size)


def psnr(reconstructed: torch.Tensor, original: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((reconstructed - original) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val / math.sqrt(mse))


def ssim(reconstructed: torch.Tensor, original: torch.Tensor, max_val: float = 1.0, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    # Simplified SSIM over the whole image
    mu_x = reconstructed.mean()
    mu_y = original.mean()
    sigma_x = reconstructed.var()
    sigma_y = original.var()
    sigma_xy = ((reconstructed - mu_x) * (original - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (numerator / denominator).item()