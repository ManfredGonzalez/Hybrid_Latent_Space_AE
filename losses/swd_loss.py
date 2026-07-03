import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SWDVarianceBudgetLoss(nn.Module):
    def __init__(self, num_projections=512, variance_budget_lambda=0.1, swd_weight=1.0, var_weight=1.0, queue_size=0):
        """
        Calculates the Sliced-Wasserstein Distance (SWD) enforcing an isotropic Gaussian shape,
        along with a variance budget to prevent deterministic collapse of the continuous branch.

        queue_size: size of a FIFO memory bank of past (detached) latents that is concatenated
        with the current batch before the quantile matching step. This lets the empirical
        distribution used for the Gaussian-shape check be much larger than what fits in one
        forward/backward pass, without needing gradients through the queued samples.
        """
        super().__init__()
        self.num_projections = num_projections
        self.variance_budget_lambda = variance_budget_lambda
        self.swd_weight = swd_weight
        self.var_weight = var_weight
        self.queue_size = queue_size

        # Lazily allocated once the latent dimension D is known from the first forward call
        self.queue = None
        self.queue_filled = 0
        self.queue_ptr = 0

        # Standard normal prior for the SWD target
        self.normal_dist = Normal(0, 1)

    def _enqueue(self, z_detached):
        B = z_detached.size(0)
        if self.queue is None:
            D = z_detached.size(1)
            self.queue = torch.zeros(self.queue_size, D, device=z_detached.device, dtype=z_detached.dtype)

        if B >= self.queue_size:
            self.queue.copy_(z_detached[-self.queue_size:])
            self.queue_filled = self.queue_size
            self.queue_ptr = 0
            return

        end = self.queue_ptr + B
        if end <= self.queue_size:
            self.queue[self.queue_ptr:end] = z_detached
        else:
            first_part = self.queue_size - self.queue_ptr
            self.queue[self.queue_ptr:] = z_detached[:first_part]
            self.queue[:end - self.queue_size] = z_detached[first_part:]
        self.queue_ptr = end % self.queue_size
        self.queue_filled = min(self.queue_filled + B, self.queue_size)

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
        # Bring in the memory bank (detached, no grad) so the quantile matching sees a much
        # larger effective sample than the current batch, while gradients only flow into `z`.
        if self.queue_size > 0 and self.queue_filled > 0:
            z_extended = torch.cat([z, self.queue[:self.queue_filled].to(device)], dim=0)
        else:
            z_extended = z
        B_total = z_extended.size(0)

        # Generate random projection matrix [D, K]
        W = torch.randn(D, self.num_projections, device=device)

        # Normalize directions to project onto the unit sphere
        W = W / torch.norm(W, p=2, dim=0, keepdim=True)

        # Project latents: [B_total, D] @ [D, K] -> [B_total, K]
        projections = z_extended @ W

        # Sort projected values along the batch dimension
        projections_sorted, _ = torch.sort(projections, dim=0)

        # Generate target standard normal quantiles dynamically based on the extended sample size.
        # (This prevents crashes on the last batch of the dataloader if it's smaller than the set batch_size)
        u = (torch.arange(1, B_total + 1, device=device) - 0.5) / B_total

        # Calculate the inverse CDF to get the theoretical Gaussian quantiles
        target_quantiles = self.normal_dist.icdf(u).unsqueeze(1).expand(B_total, self.num_projections)

        # Normalize by the real batch size (not B_total) so the loss scale matches the no-queue
        # case exactly when queue_size=0, and doesn't get diluted as the queue grows.
        swd_loss = ((projections_sorted - target_quantiles) ** 2).sum() / (B * self.num_projections)

        # --- 3. Total Loss Calculation ---
        total_loss = (self.swd_weight * swd_loss) + (self.var_weight * var_loss)

        if self.queue_size > 0:
            self._enqueue(z.detach())

        return total_loss, swd_loss, var_loss