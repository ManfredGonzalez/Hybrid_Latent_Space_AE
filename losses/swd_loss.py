import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SWDVarianceBudgetLoss(nn.Module):
    """
    Sliced-Wasserstein Distance to N(0, I) plus a variance budget, with two modes:

    mode='global'        : each image's full latent map is one sample.
                           z [B, C, H, W] -> [B, C*H*W].  (Original behavior.)
    mode='per_location'  : each spatial position is one sample.
                           z [B, C, H, W] -> [B*H*W, C].
                           This matches the unit the VQ branch quantizes, so it is
                           the mode consistent with the per-location GMM story.

    Notes for per_location mode:
      - One batch already yields B*H*W samples, so queue_size=0 is the sensible
        default. If a queue is used, a random subsample of the batch's vectors
        is enqueued (dumping all of them would overwrite the queue every step,
        and slicing would be spatially biased).
      - num_projections can be small (64-128); directions in low dimension are
        cheap to cover.
    """

    def __init__(self, num_projections=128, variance_budget_lambda=0.1,
                 swd_weight=1.0, var_weight=1.0, queue_size=0,
                 mode='per_location', max_enqueue_per_step=None, sigma0=None):
        super().__init__()
        assert mode in ('global', 'per_location')
        self.mode = mode
        self.num_projections = num_projections
        self.variance_budget_lambda = variance_budget_lambda
        self.swd_weight = swd_weight
        self.var_weight = var_weight
        self.queue_size = queue_size
        # SWAE mode: when sigma0 is not None, the SWD is matched against a FIXED
        # N(0, sigma0^2 I) target and the variance-budget floor is disabled. The fixed
        # target is itself the scale anchor, so no KL / floor / per-component whitening
        # is needed. sigma0=None keeps the original N(0, I) target + variance floor.
        #
        # sigma0 may be:
        #   * a scalar  -> isotropic fixed target N(0, sigma0^2 I). Implemented by scaling
        #     the N(0,1) quantiles by sigma0 (cheap, exact).
        #   * a (D,) tensor -> ANISOTROPIC fixed target N(0, diag(sigma0^2)), i.e. a
        #     per-CHANNEL scale. Used by the wavelet detail branch to give each frequency
        #     band its own fixed variance. Since the SWD's random projections mix channels,
        #     an anisotropic target is imposed by WHITENING z per channel (z / sigma0) and
        #     matching the whitened z against N(0, I). This is a FIXED whitening (sigma0 is
        #     a constant, not measured from Delta), so it does NOT reintroduce the
        #     self-referentiality of the EMA sigma_k whitening.
        if isinstance(sigma0, torch.Tensor) and sigma0.dim() == 1:
            self.register_buffer('sigma0_vec', sigma0.float(), persistent=False)
            self.sigma0 = None
        else:
            self.sigma0 = sigma0
            self.sigma0_vec = None
        # Cap on how many (detached) samples enter the queue per step.
        # None -> default to queue_size // 8 so the queue mixes many steps.
        self.max_enqueue_per_step = max_enqueue_per_step

        self.queue = None
        self.queue_filled = 0
        self.queue_ptr = 0

        self.normal_dist = Normal(0, 1)

    # ------------------------------------------------------------------ #
    # Queue helpers
    # ------------------------------------------------------------------ #
    def _enqueue(self, samples_detached):
        """samples_detached: [N, D], already detached."""
        if self.queue_size <= 0:
            return

        # Subsample so one step never floods the whole queue.
        cap = self.max_enqueue_per_step
        if cap is None:
            cap = max(1, self.queue_size // 8)
        N = samples_detached.size(0)
        if N > cap:
            idx = torch.randperm(N, device=samples_detached.device)[:cap]
            samples_detached = samples_detached[idx]
            N = cap

        if self.queue is None:
            D = samples_detached.size(1)
            self.queue = torch.zeros(self.queue_size, D,
                                     device=samples_detached.device,
                                     dtype=samples_detached.dtype)

        end = self.queue_ptr + N
        if end <= self.queue_size:
            self.queue[self.queue_ptr:end] = samples_detached
        else:
            first = self.queue_size - self.queue_ptr
            self.queue[self.queue_ptr:] = samples_detached[:first]
            self.queue[:end - self.queue_size] = samples_detached[first:]
        self.queue_ptr = end % self.queue_size
        self.queue_filled = min(self.queue_filled + N, self.queue_size)

    # ------------------------------------------------------------------ #
    # Reshaping
    # ------------------------------------------------------------------ #
    def _as_samples(self, t):
        """Reshape z / log_var to a 2-D sample matrix according to self.mode."""
        if t.dim() == 2:
            return t  # already [N, D]
        if self.mode == 'per_location':
            # [B, C, H, W] -> [B*H*W, C]
            B, C = t.size(0), t.size(1)
            return t.permute(0, 2, 3, 1).reshape(-1, C)
        # global: [B, C, H, W] -> [B, C*H*W]
        return t.reshape(t.size(0), -1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, z, log_var):
        """
        z, log_var: [B, C, H, W] (spatial latents) or [B, D]. log_var may be None
        (deterministic continuous branch), in which case the variance-budget term
        is skipped and only the SWD shape term is computed.
        Returns: total_loss, swd_loss, var_loss
        """
        z = self._as_samples(z)
        N, D = z.shape
        device = z.device

        # Per-channel fixed-sigma0 whitening (wavelet per-band target): divide each
        # channel by its fixed sigma0, then match the whitened z against N(0, I).
        anisotropic = self.sigma0_vec is not None
        if anisotropic:
            z = z / self.sigma0_vec.to(device).clamp(min=1e-6).unsqueeze(0)

        # --- 1. Variance budget (identical under both modes: global mean) ---
        # SWAE mode (sigma0 set) drops the floor entirely: the fixed-target SWD
        # below is the sole scale anchor, so there is no separate budget term.
        if self.sigma0 is not None or anisotropic or log_var is None:
            var_loss = torch.zeros((), device=device)
        else:
            log_var = self._as_samples(log_var)
            mean_variance = torch.exp(log_var).mean()
            var_loss = F.relu(self.variance_budget_lambda - mean_variance)

        # --- 2. Sliced-Wasserstein distance to N(0, I) ---
        if self.queue_size > 0 and self.queue_filled > 0:
            z_ext = torch.cat([z, self.queue[:self.queue_filled].to(device)], dim=0)
        else:
            z_ext = z
        N_total = z_ext.size(0)

        W = torch.randn(D, self.num_projections, device=device)
        W = W / torch.norm(W, p=2, dim=0, keepdim=True)

        projections = z_ext @ W                       # [N_total, K]
        projections_sorted, _ = torch.sort(projections, dim=0)

        u = (torch.arange(1, N_total + 1, device=device) - 0.5) / N_total
        target_q = self.normal_dist.icdf(u).unsqueeze(1)  # [N_total, 1], broadcasts
        # SWAE mode: quantiles of N(0, sigma0^2) are sigma0 * quantiles of N(0, 1).
        # This is what anchors the absolute scale of Delta to a fixed sigma0
        # (the un-whitened target the fixed-sigma0 recipe requires).
        if self.sigma0 is not None:
            target_q = self.sigma0 * target_q

        # .mean() normalizes by N_total * K: scale is stable regardless of
        # queue fill level or batch size (fixes the old inflation bug).
        swd_loss = ((projections_sorted - target_q) ** 2).mean()

        total_loss = self.swd_weight * swd_loss + self.var_weight * var_loss

        if self.queue_size > 0:
            self._enqueue(z.detach())

        return total_loss, swd_loss, var_loss
