"""Finite Scalar Quantization (Mentzer et al., 2023) as a drop-in for VQEmbedding.

FSQ replaces the learned codebook with a FIXED grid: each of the d latent channels is
squashed into a bounded range and rounded to one of L_i levels, so the implicit codebook is
the Cartesian product of the per-channel level sets, |C| = prod(L_i). Nothing is learned
here -- no codewords, no commitment loss, no dead-code restarts, no k-means seeding.

WHY THIS MODULE STILL KEEPS EMA STATISTICS
------------------------------------------
The DualVAE's code-centered prior does NOT need a learned codebook. It needs, per code k:
    pi_k     mixture weight            = N_k / sum(N)
    mu_k     component mean            = grid_k + E[r | k]
    sigma2_k component variance        = Var[r | k]
where r is the within-cell residual. Those are statistics OF THE DATA, so they survive the
switch untouched -- we keep exactly the EMA machinery VQEmbedding uses (and its DDP
all-reduce), and only drop the part that MOVED the codes.

One thing genuinely changes. With an EMA codebook, e_k is the cluster MEAN, so the residual
r = z - e_k is zero-mean by construction and N(0, sigma_k^2) is the correct component. An
FSQ grid point is the nearest CORNER, not the centre of mass, so r has a systematic offset.
That offset is mu_k, tracked here and applied as the prior MEAN. Without it the prior would
be a mis-centred mixture and the KL would spend itself fighting real structure. See
`gmm_params()` -- the fitted mixture rides along in the checkpoint.

Variance convention: sigma2 = E[r^2] - E[r]^2 (a true variance around mu_k), not the raw
second moment VQEmbedding stores. Under VQ the two coincide because E[r] ~= 0; here they do
not, and the mixture parameters have to be internally consistent to be usable downstream.

Per-channel (diagonal) statistics: mu and sigma2 are (K, d), not (K,). A within-cell offset
is directional -- "which corner the mass sits in" is d numbers, not one -- so an isotropic
scalar would throw away most of the signal.

SCALE. Following the reference implementation, the quantized output is renormalized to
~[-1, 1] by dividing by (L_i // 2). Grid spacing is therefore 2/(L_i-ish) rather than 1, and
the residual is bounded by HALF A CELL. For levels=[6,6,6,6]: spacing 1/3, |r| <= 1/6, so
sigma2 <= 1/36 = 0.0278 with a uniform-in-cell value of 1/108 = 0.00926. sigma2_ceil/floor
must be set for THAT range -- a floor of 0.1 (the VQ default) would clamp every code and
silently flatten the prior into N(mu_k, const).
"""

import math

import torch
import torch.nn as nn

from tools.distributed import all_reduce_sum_


class FSQEmbedding(nn.Module):
    """Drop-in replacement for VQEmbedding: same forward signature, same return tuple, same
    diagnostic attribute names, so dual_vae.py / the trainers / the wandb panels are
    unchanged. `levels` is the only new required argument."""

    def __init__(self, levels, embedding_dim=None, ema_decay=0.99, ema_eps=1e-5,
                 sigma2_floor=1e-4, sigma2_ceil=0.0278, rq_depth=1, reduction='sum',
                 eps=1e-3, **_ignored):
        super().__init__()
        if not levels or any((not isinstance(l, int)) or l < 2 for l in levels):
            raise ValueError(f"fsq_levels must be a list of ints >= 2, got {levels!r}.")
        if rq_depth != 1:
            # Not an arbitrary restriction: after FSQ the residual is bounded by half a
            # cell, so re-quantizing it on the SAME grid rounds every value straight back
            # to zero -- depth >= 2 would be a silent no-op that costs a forward pass.
            # Extra capacity comes from a larger prod(levels), which is free here.
            raise ValueError(f"rq_depth must be 1 with quantizer='fsq' (got {rq_depth}); "
                             f"enlarge fsq_levels instead -- FSQ has no per-code parameters, "
                             f"so a bigger implicit codebook costs nothing.")
        if embedding_dim is not None and embedding_dim != len(levels):
            raise ValueError(
                f"latent_channels ({embedding_dim}) must equal len(fsq_levels) ({len(levels)}). "
                f"FSQ quantizes every latent channel, so d IS the channel count. Either set "
                f"latent_channels={len(levels)} or give {embedding_dim} levels.")
        if any(l < 5 for l in levels):
            # Paper's heuristic (App. A.4.1): L_i < 5 measurably underperforms. Warned, not
            # blocked -- [4,4,4,4] is the only exact-256 grid at d=4 and is a legitimate
            # ablation if that parity is what you are buying.
            print(f"[FSQ] WARNING: levels {levels} contain L_i < 5; the paper reports subpar "
                  f"performance below 5 levels per channel.")

        self.levels = list(levels)
        self.embedding_dim = len(self.levels)          # d
        self.num_embeddings = int(math.prod(self.levels))   # |C|
        self.rq_depth = 1
        self.reduction = reduction
        self.eps = eps
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.sigma2_floor = sigma2_floor
        self.sigma2_ceil = sigma2_ceil
        # Interface parity with VQEmbedding. `use_ema` gates the GMM panels in
        # codebook_health_metrics and is honestly True: the mixture statistics ARE EMA-tracked.
        # The two below exist only so VQ-shaped code paths keep working.
        self.use_ema = True
        self.l2_normalize = False
        self.commitment_cost = 0.0

        lv = torch.tensor(self.levels, dtype=torch.float32)
        # Bounding constants (reference implementation, App. A.1). `shift` recentres the grid
        # for EVEN L, where the integer levels straddle zero asymmetrically.
        self.register_buffer('_levels_f', lv, persistent=False)
        self.register_buffer('_half_l', (lv - 1) * (1 - eps) / 2, persistent=False)
        self.register_buffer('_offset', torch.where(lv % 2 == 0,
                                                    torch.tensor(0.5), torch.tensor(0.0)),
                             persistent=False)
        self.register_buffer('_shift', torch.tan(
            torch.where(lv % 2 == 0, torch.tensor(0.5), torch.tensor(0.0)) / ((lv - 1) * (1 - eps) / 2)
        ), persistent=False)
        self.register_buffer('_half_width', (lv // 2).clamp(min=1), persistent=False)
        # Mixed-radix basis for the level-tuple <-> index bijection.
        basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer('_basis', basis, persistent=False)

        # --- GMM statistics (persistent: the checkpoint IS the fitted mixture) ---
        # N_k, sum of residuals, sum of squared residuals. Initialized so mu_k = 0 and
        # sigma2_k = the uniform-in-cell value at step 0, i.e. a sane broad prior before any
        # data has been seen.
        self.register_buffer('ema_cluster_size', torch.ones(self.num_embeddings))
        self.register_buffer('ema_res_sum', torch.zeros(self.num_embeddings, self.embedding_dim))
        uniform_var = ((1.0 / self._half_width) ** 2) / 12.0        # (d,)
        self.register_buffer('ema_res_sq', uniform_var.expand(self.num_embeddings, -1).clone())
        # Cumulative "has this code EVER been assigned". The EMA count above starts at 1 for
        # every code so that mu/sigma2 are well-defined at step 0, which means N_k alone
        # cannot distinguish "never seen" from "seen and forgotten" until the decay has run
        # long enough to bury the init. This flag is unambiguous from the first step, and it
        # is also the honest measure of FSQ's headline claim: what fraction of the implicit
        # codebook the data ever reaches.
        self.register_buffer('codes_seen', torch.zeros(self.num_embeddings))

        # The one silent failure mode worth a hard warning. VQ's floor (0.1 in Run C) was set
        # for the encoder's own scale; on this grid the ENTIRE achievable range of sigma2 is
        # [0, (1/(2*half_width))^2] -- 0.0278 for levels of 6. Inheriting the VQ floor would
        # clamp every code, flatten the prior to a constant, and the run would look healthy
        # the whole way (Sigma2 At Floor Frac just pegs at 1.0).
        max_sigma2 = float(((1.0 / self._half_width) ** 2 / 4).max())
        if self.sigma2_floor >= float(uniform_var.min()):
            print(f"[FSQ] WARNING: sigma2_floor={self.sigma2_floor:g} is at or above the "
                  f"uniform-in-cell variance ({float(uniform_var.min()):.2e}) for levels "
                  f"{self.levels}. Nearly every code will clamp and the per-code prior will "
                  f"be constant. Expected range is [0, {max_sigma2:.2e}]; try ~1e-4.")

        # Diagnostics, mirroring VQEmbedding's names exactly.
        self.register_buffer('perplexity', torch.zeros(()), persistent=False)
        self.register_buffer('codebook_usage', torch.zeros(()), persistent=False)
        self.register_buffer('rel_quant_error', torch.zeros(()), persistent=False)
        # Always 0: a fixed grid has no code to restart. Kept so the wandb panel is
        # continuous across VQ and FSQ runs instead of appearing/disappearing.
        self.register_buffer('restarted_codes', torch.zeros(()), persistent=False)
        self.perplexity_per_depth = []

    # ------------------------------------------------------------------ #
    # GMM views of the EMA statistics
    # ------------------------------------------------------------------ #
    @property
    def pi(self):
        """Mixture weights pi_k = N_k / sum(N)."""
        n = self.ema_cluster_size
        return n / n.sum().clamp(min=1e-12)

    @property
    def mu(self):
        """(K, d) per-code MEAN within-cell offset E[r | k]. This is the term an EMA codebook
        gets for free (its codes ARE the means) and a fixed grid does not."""
        return self.ema_res_sum / self.ema_cluster_size.clamp(min=self.ema_eps).unsqueeze(1)

    @property
    def sigma2_raw(self):
        """(K, d) diagonal variance Var[r|k] = E[r^2] - E[r]^2, UNCLAMPED.

        Exposed separately because the clamp below is a training-stability device, not a
        property of the data: anything that SAMPLES the mixture (flow-matching source,
        the MSM latent head) wants the estimate, not the floored version.
        """
        n = self.ema_cluster_size.clamp(min=self.ema_eps).unsqueeze(1)
        return (self.ema_res_sq / n) - (self.ema_res_sum / n) ** 2

    @property
    def sigma2(self):
        """(K, d) diagonal variance, clamped to [sigma2_floor, sigma2_ceil] for the KL.

        The floor is still the anti-collapse valve even though the codes cannot move: an
        encoder that piles every vector onto the cell centres drives sigma2 -> 0, which makes
        the prior infinitely tight and squeezes Delta out of the model entirely.
        """
        return self.sigma2_raw.clamp(self.sigma2_floor, self.sigma2_ceil)

    @property
    def codebook(self):
        """(K, d) the materialized grid, in the SAME normalized space as forward()'s output.
        Cheap at |C| ~ 1e3-1e4; the analysis/CVI paths want an explicit matrix."""
        idx = torch.arange(self.num_embeddings, device=self.ema_cluster_size.device)
        digits = (idx.unsqueeze(1) // self._basis.unsqueeze(0)) % self._levels_f.long().unsqueeze(0)
        ints = digits.float() - (self._levels_f // 2).unsqueeze(0)      # inverse of _to_index
        return ints / self._half_width.unsqueeze(0)

    @property
    def component_means(self):
        """(K, d) GMM component means in latent space: grid point + its EMA offset.

        This is what `masked_latent_gmm_loss` and any downstream generative sampler must use
        as `means`. Using the bare grid instead would hand them a mixture whose components
        are all displaced from the data.
        """
        return self.codebook + self.mu

    def gmm_params(self, clamped=False):
        """(pi, mu, sigma2) with mu in LATENT space -- the fitted mixture, ready to sample.

        This is the whole of "Stage 1" for a mode-coupled flow-matching source: no EM, no
        responsibilities, no dimensionality workaround. Defaults to the UNCLAMPED variance
        for the reason given on `sigma2_raw`.
        """
        s2 = self.sigma2 if clamped else self.sigma2_raw.clamp(min=1e-12)
        return self.pi.detach(), self.component_means.detach(), s2.detach()

    @torch.no_grad()
    def init_from_data(self, z, num_iters=10):
        """No-op. The grid is fixed; there is nothing to seed. Present so the trainer's
        `initialize_from_data` path does not need to know which quantizer it is holding."""
        return

    # ------------------------------------------------------------------ #
    # Quantization
    # ------------------------------------------------------------------ #
    def _bound(self, z):
        """Squash into the rounding range: tanh(z + shift) * half_l - offset, channel-wise."""
        shape = (1, -1) + (1,) * (z.dim() - 2)      # broadcast over (B, d, H, W)
        return (torch.tanh(z + self._shift.view(shape)) * self._half_l.view(shape)
                - self._offset.view(shape))

    def pre_quant(self, z):
        """The value the quantizer actually rounds, in the SAME space as its output.

        dual_vae.py forms the continuous branch's input as r = pre_quant(z_e) - z_q.detach().
        For VQ that is the identity (codes live in the encoder's space). For FSQ the tanh
        bound moves z into the grid's space, and skipping this would subtract a bounded
        quantity from an unbounded one -- r would be dominated by the encoder's raw scale
        instead of being the within-cell offset the prior is defined on.
        """
        shape = (1, -1) + (1,) * (z.dim() - 2)
        return self._bound(z) / self._half_width.view(shape)

    def _to_index(self, ints):
        """(N, d) integer levels -> (N,) code index, mixed radix."""
        digits = ints + (self._levels_f // 2).long().unsqueeze(0)       # -> {0 .. L_i-1}
        return (digits * self._basis.unsqueeze(0)).sum(dim=1)

    def forward(self, z):
        b, c, h, w = z.shape
        if c != self.embedding_dim:
            raise ValueError(f"FSQ expects {self.embedding_dim} channels (len(levels)), got {c}.")
        shape = (1, -1, 1, 1)
        hw = self._half_width.view(shape)

        # Bound + round in fp32 even under bf16 autocast: rounding decides the CODE INDEX,
        # and a bf16 value sitting a rounding-error away from x.5 would flip it, making the
        # assignment (and every statistic keyed on it) non-reproducible.
        with torch.autocast(device_type=z.device.type, enabled=False):
            u = self._bound(z.float())                       # (B, d, H, W), grid units
            u_int = torch.round(u)
            # Straight-through: forward uses the rounded value, backward passes the gradient
            # through `u` -- so it reaches the encoder VIA the tanh, which is what keeps the
            # encoder inside the representable range instead of saturating past it.
            u_ste = u + (u_int - u).detach()
            z_q = u_ste / hw                                 # renormalize to ~[-1, 1]
            u_norm = u / hw                                  # bounded, pre-round (same space)

            flat_ints = u_int.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim).long()
            indices = self._to_index(flat_ints)              # (N,)

            # Within-cell residual, in the normalized space the prior lives in.
            residual = (u_norm - u_ste.detach() / hw)
            res_flat = residual.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

            with torch.no_grad():
                # bincount / index_add_ rather than the (N, K) one-hot VQEmbedding builds:
                # at |C| = 1296 and N = 40k that one-hot would already be 200 MB, and it
                # grows linearly with the implicit codebook FSQ exists to make large.
                counts = torch.bincount(indices, minlength=self.num_embeddings).float()
                p = counts / counts.sum().clamp(min=1e-12)
                perp = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))
                self.perplexity = perp
                self.codebook_usage = (counts > 0).float().mean()
                self.perplexity_per_depth = [perp.item()]
                self.rel_quant_error = (
                    (z_q.detach() - u_norm).pow(2).sum() / u_norm.pow(2).sum().clamp(min=1e-12)
                )

                if self.training:
                    res_sum = torch.zeros_like(self.ema_res_sum)
                    res_sq = torch.zeros_like(self.ema_res_sq)
                    res_sum.index_add_(0, indices, res_flat)
                    res_sq.index_add_(0, indices, res_flat ** 2)
                    # Same contract as VQEmbedding: these are SUMS over the local shard, so
                    # summing across ranks reproduces the full-batch statistics exactly. DDP
                    # syncs gradients, and these have none -- without this each rank would
                    # fit the mixture on 1/world_size of the data.
                    all_reduce_sum_(counts, res_sum, res_sq)
                    # After the all-reduce, so a code seen by ANY rank counts as seen.
                    self.codes_seen.copy_(torch.maximum(self.codes_seen, (counts > 0).float()))
                    d = self.ema_decay
                    self.ema_cluster_size.mul_(d).add_(counts, alpha=1 - d)
                    self.ema_res_sum.mul_(d).add_(res_sum, alpha=1 - d)
                    self.ema_res_sq.mul_(d).add_(res_sq, alpha=1 - d)

            # Quantization error, reported on the same scale convention as VQEmbedding's
            # codebook_loss so the panel stays readable. Detached: nothing here is trained.
            if self.reduction == 'mean':
                quant_err = (z_q.detach() - u_norm).pow(2).mean()
            else:
                quant_err = (z_q.detach() - u_norm).pow(2).sum()

        z_q = z_q.to(z.dtype)
        zero = torch.zeros((), device=z.device, dtype=z.dtype)
        # loss = 0 and commitment = 0: FSQ has nothing to commit to. They are returned so the
        # tuple, the trainer's running dict and the wandb panels keep the same shape as a VQ
        # run -- the terms simply log as zero.
        return z_q, zero, indices, zero, quant_err.to(z.dtype)
