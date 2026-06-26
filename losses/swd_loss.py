import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SWDVarianceBudgetLoss(nn.Module):
    def __init__(self, num_projections=512, variance_budget_lambda=0.1, swd_weight=1.0, var_weight=1.0):
        """
        Calculates the Sliced-Wasserstein Distance (SWD) enforcing an isotropic Gaussian shape,
        along with a variance budget to prevent deterministic collapse of the continuous branch.
        """
        super().__init__()
        self.num_projections = num_projections
        self.variance_budget_lambda = variance_budget_lambda
        self.swd_weight = swd_weight
        self.var_weight = var_weight
        
        # Standard normal prior for the SWD target
        self.normal_dist = Normal(0, 1)

    def forward(self, z, log_var):
        """
        Args:
            z: Sampled latents (e.g., from the reparameterization trick).
               Shape: [B, D] or [B, C, H, W]
            log_var: Log variance from the encoder.
                     Shape: [B, D] or [B, C, H, W]
        Returns:
            total_loss, swd_loss, var_loss
        """
        # 1. Flatten spatial dimensions if the latents are 2D feature maps
        if z.dim() > 2:
            B = z.size(0)
            z = z.view(B, -1)
            log_var = log_var.view(B, -1)
        else:
            B, D = z.shape

        D = z.size(1)
        device = z.device

        # --- 1. Variance Budget Loss ---
        # Extract variance from log_var: var = exp(log_var)
        # Calculate the mean variance across the entire batch and feature dimensions
        mean_variance = torch.exp(log_var).mean()
        
        # Hard margin loss: heavily penalize if mean variance drops below the lambda threshold
        var_loss = F.relu(self.variance_budget_lambda - mean_variance)

        # --- 2. Sliced-Wasserstein Distance (SWD) Loss ---
        # Generate random projection matrix [D, K]
        W = torch.randn(D, self.num_projections, device=device)
        
        # Normalize directions to project onto the unit sphere
        W = W / torch.norm(W, p=2, dim=0, keepdim=True)

        # Project latents: [B, D] @ [D, K] -> [B, K]
        projections = z @ W

        # Sort projected values along the batch dimension
        projections_sorted, _ = torch.sort(projections, dim=0)

        # Generate target standard normal quantiles dynamically based on current batch size B.
        # (This prevents crashes on the last batch of the dataloader if it's smaller than the set batch_size)
        u = (torch.arange(1, B + 1, device=device) - 0.5) / B
        
        # Calculate the inverse CDF to get the theoretical Gaussian quantiles
        target_quantiles = self.normal_dist.icdf(u).unsqueeze(1).expand(B, self.num_projections)

        # SWD is the Mean Squared Error between the empirical sorted projections and the theoretical quantiles
        swd_loss = F.mse_loss(projections_sorted, target_quantiles)

        # --- 3. Total Loss Calculation ---
        total_loss = (self.swd_weight * swd_loss) + (self.var_weight * var_loss)

        return total_loss, swd_loss, var_loss