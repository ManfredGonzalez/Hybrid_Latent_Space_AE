import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, reduction='sum', l2_normalize=False,
                 use_ema=False, ema_decay=0.99, ema_eps=1e-5, ema_dead_threshold=1.0,
                 rq_depth=1, sigma2_floor=1e-3, sigma2_ceil=10.0):
        super().__init__()
        if not (isinstance(rq_depth, int) and rq_depth >= 1):
            raise ValueError(f"rq_depth must be an int >= 1, got {rq_depth!r}.")
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings # Number of vectors in the codebook
        self.commitment_cost = commitment_cost # Beta, the commitment loss weight
        self.reduction = reduction # How to reduce the loss: 'sum' or 'mean'
        self.l2_normalize = l2_normalize # Normalize codes/lookup to the unit sphere before the distance computation (helps codebook utilization at low dims)

        # --- Residual quantization (Lee et al. 2022, RQ-VAE) ---
        # rq_depth D > 1 represents each location as D codes chosen coarse-to-fine on the
        # successive residuals, all drawn from ONE shared codebook. Capacity per location
        # becomes D*log2(K) bits (partition capacity of a K^D codebook) without the
        # collapse risk of actually training K^D codes. rq_depth=1 is exactly the
        # original single-step VQ (bit-identical losses), so it doubles as the ablation off-switch.
        self.rq_depth = rq_depth

        # --- EMA codebook (van den Oord et al. 2017, App. A.1; Razavi et al. 2019) ---
        # When use_ema=True the codebook is NOT trained by gradients: each code is the
        # running mean of the encoder vectors assigned to it (online k-means), tracked
        # via an EMA count N_k (ema_cluster_size) and an EMA vector-sum m_k (ema_embed_sum),
        # with e_k = m_k / N_k. The codebook_loss term is then excluded from the returned
        # `loss` (commitment only) and kept as a detached diagnostic. Codes whose smoothed
        # count falls below ema_dead_threshold are restarted from random encoder vectors
        # of the current batch (Jukebox-style). use_ema=False keeps the original
        # gradient-trained codebook, bit-identical to previous behavior (for ablations).
        #
        # GMM statistics: alongside N_k and m_k we track v_k (ema_res_sq), the EMA of the
        # per-code sum of mean-squared depth-1 residuals. sigma2 = v_k / N_k is then the
        # within-component variance (online EM for a mixture of Gaussians), and
        # pi = N_k / sum(N) the mixture weights. sigma2 is clamped to
        # [sigma2_floor, sigma2_ceil]: the floor is the anti-collapse valve that breaks
        # the "tight prior -> smaller residuals -> tighter prior" ratchet, the ceiling
        # bounds the "inflate residuals to loosen your own prior" loophole.
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_threshold = ema_dead_threshold
        self.sigma2_floor = sigma2_floor
        self.sigma2_ceil = sigma2_ceil

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #Initializes the embedding weights uniformly to help with training stability.
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        if self.use_ema:
            # Codebook is updated in-place under no_grad; freeze it so the optimizer
            # never applies a competing gradient update.
            self.embedding.weight.requires_grad_(False)
            # Persistent so checkpoints resume with consistent statistics. Initialized
            # to N_k=1 and m_k=e_k so e_k = m_k / N_k holds at step 0; v_k=1 so the
            # initial per-component prior variance is a broad 1.0.
            self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
            self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())
            self.register_buffer('ema_res_sq', torch.ones(num_embeddings))
            self.register_buffer('restarted_codes', torch.zeros(()), persistent=False)

        # Codebook health stats from the most recent forward pass (diagnostics only,
        # not used in any loss). perplexity == exp(entropy of code usage) at DEPTH 1
        # (the component identity): num_embeddings when every code is used equally often,
        # 1 when only one code is ever picked. codebook_usage == fraction of codes used
        # at least once in the batch across ALL depths. perplexity_per_depth is a plain
        # python list with one perplexity per RQ depth. rel_quant_error is the
        # scale-invariant quantization error ||z_q - z||^2 / ||z||^2 (immune to the
        # "encoder magnitude grew, so sum-MSE looks worse" artifact).
        self.register_buffer('perplexity', torch.zeros(()), persistent=False)
        self.register_buffer('codebook_usage', torch.zeros(()), persistent=False)
        self.register_buffer('rel_quant_error', torch.zeros(()), persistent=False)
        self.perplexity_per_depth = []

    # ------------------------------------------------------------------ #
    # GMM views of the EMA statistics
    # ------------------------------------------------------------------ #
    @property
    def pi(self):
        """Mixture weights pi_k = N_k / sum(N): empirical probability of component k."""
        n = self.ema_cluster_size
        return n / n.sum().clamp(min=1e-12)

    @property
    def sigma2(self):
        """Per-component (isotropic, per-dim) variance sigma_k^2 = v_k / N_k, clamped."""
        s2 = self.ema_res_sq / self.ema_cluster_size.clamp(min=self.ema_eps)
        return s2.clamp(self.sigma2_floor, self.sigma2_ceil)

    def _pairwise_distances(self, z_flattened, codebook):
        if self.l2_normalize:
            # Cosine-similarity nearest-neighbor search only: for unit vectors
            # ||a-b||^2 = 2 - 2*a.b. The returned codeword is still RAW, so the rest
            # of the pipeline sees the same raw-scale space as when l2_normalize is
            # off -- only which code gets picked changes.
            z_cmp = F.normalize(z_flattened, dim=-1)
            cb_cmp = F.normalize(codebook, dim=-1)
            return 2.0 - 2.0 * torch.matmul(z_cmp, cb_cmp.t())
        # Efficient Euclidean distances via |a-b|^2 = a^2 + b^2 - 2ab
        return (
            torch.sum(z_flattened ** 2, dim=-1, keepdim=True)
            + torch.sum(codebook.t() ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(z_flattened, codebook.t())
        )

    @torch.no_grad()
    def init_from_data(self, z, num_iters=10):
        """Re-seed the codebook via k-means (Lloyd's algorithm) on encoder activations.

        z: encoder output with the same (B, C, H, W) layout `forward` expects, e.g. from
        one batch pushed through the encoder (in fp32, outside autocast) before training
        starts. Replaces the default uniform init with centroids of the actual data
        distribution, which fixes the scale mismatch against the codebook's uniform
        [-1/N, 1/N] init and cuts down on dead codes early in training.
        """
        b, c, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 1).reshape(b * h * w, self.embedding_dim)
        n_vectors = z_flattened.shape[0]

        if n_vectors < self.num_embeddings:
            raise ValueError(
                f"init_from_data needs at least num_embeddings ({self.num_embeddings}) vectors "
                f"to seed centroids, got {n_vectors}. Pass a larger batch."
            )

        # Seed centroids from the data itself (k-means++ style would be nicer, but plain
        # random seeding is enough at these codebook sizes and converges in a few iters).
        perm = torch.randperm(n_vectors, device=z_flattened.device)[:self.num_embeddings]
        centroids = z_flattened[perm].clone()

        for _ in range(num_iters):
            # Assign in the same metric `forward` uses for lookup (cosine when
            # l2_normalize, else Euclidean), but always update/store raw centroids --
            # mirrors forward(), which compares normalized vectors but returns the raw
            # codeword.
            distances = self._pairwise_distances(z_flattened, centroids)
            assignments = torch.argmin(distances, dim=-1)

            for k in range(self.num_embeddings):
                mask = assignments == k
                if mask.any():
                    centroids[k] = z_flattened[mask].mean(dim=0)
                else:
                    # Empty cluster: teleport to a random data point instead of freezing
                    # it in a dead region, so it gets a chance to win points next pass.
                    centroids[k] = z_flattened[torch.randint(n_vectors, (1,), device=z_flattened.device)].squeeze(0)

        self.embedding.weight.data.copy_(centroids)

        # Seed the EMA statistics to be consistent with the k-means result, so the
        # first EMA steps refine these centroids instead of dragging them back toward
        # the pre-init state (e_k = m_k / N_k must hold at handoff). v_k is seeded from
        # the actual within-cluster variance, so the per-component priors start out
        # measured rather than guessed.
        if self.use_ema:
            counts = torch.bincount(assignments, minlength=self.num_embeddings).to(centroids.dtype).clamp(min=1.0)
            self.ema_cluster_size.copy_(counts)
            self.ema_embed_sum.copy_(centroids * counts.unsqueeze(1))
            sq = ((z_flattened - centroids[assignments]) ** 2).mean(dim=1)
            res_sq = torch.zeros(self.num_embeddings, device=z_flattened.device, dtype=centroids.dtype)
            res_sq.index_add_(0, assignments, sq)
            self.ema_res_sq.copy_(res_sq.clamp(min=self.sigma2_floor))

    @torch.no_grad()
    def _ema_update_from_stats(self, batch_cluster_size, batch_embed_sum, batch_res_sq_sum, z_flattened):
        """One EMA codebook step from this batch's (depth-aggregated) assignment stats.

        batch_cluster_size: (K,) n_k -- vectors assigned to code k this batch (all depths).
        batch_embed_sum:    (K, D) s_k -- sum of those (raw, un-normalized) vectors.
        batch_res_sq_sum:   (K,) sum over DEPTH-1 assigned vectors of mean-squared residual
                            (feeds sigma2, the within-component variance).
        z_flattened:        (N, D) fp32 raw depth-1 encoder vectors, used for restarts.
        Even with l2_normalize=True the RAW vectors are averaged -- mirroring forward(),
        which normalizes only for the nearest-neighbor lookup but returns raw codewords.
        """
        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(batch_cluster_size, alpha=1 - decay)  # N_k
        self.ema_embed_sum.mul_(decay).add_(batch_embed_sum, alpha=1 - decay)        # m_k
        self.ema_res_sq.mul_(decay).add_(batch_res_sq_sum, alpha=1 - decay)          # v_k

        # Laplace smoothing of the counts so m_k / N_k never divides by ~0.
        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (
            (self.ema_cluster_size + self.ema_eps) / (n + self.num_embeddings * self.ema_eps) * n
        )
        self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1))

        # Dead-code restart: codes whose smoothed usage collapsed get teleported onto
        # random encoder vectors from the current batch, and their EMA stats are re-seeded
        # at the batch-average count (and global-average variance for v_k) so they aren't
        # immediately flagged dead again and don't start with a degenerate prior.
        dead = self.ema_cluster_size < self.ema_dead_threshold
        num_dead = int(dead.sum().item())
        self.restarted_codes = torch.tensor(float(num_dead), device=z_flattened.device)
        if num_dead > 0:
            rand_idx = torch.randint(0, z_flattened.size(0), (num_dead,), device=z_flattened.device)
            new_codes = z_flattened[rand_idx]
            avg_size = self.ema_cluster_size.mean().clamp(min=1.0)
            avg_var = self.ema_res_sq.sum() / self.ema_cluster_size.sum().clamp(min=1e-12)
            self.embedding.weight.data[dead] = new_codes
            self.ema_embed_sum[dead] = new_codes * avg_size
            self.ema_cluster_size[dead] = avg_size
            self.ema_res_sq[dead] = avg_var * avg_size

    def forward(self, z):
        b, c, h, w = z.shape
        z_channel_last = z.permute(0, 2, 3, 1) # (B, H, W, C)
        z_flattened = z_channel_last.reshape(b*h*w, self.embedding_dim)

        codebook = self.embedding.weight
        mse_loss = nn.MSELoss(reduction=self.reduction)

        # ---------------- Residual quantization loop ----------------
        # Depth 1 is plain VQ. For d >= 2 each step quantizes the residual left by the
        # previous steps against the SAME codebook (RQ-VAE, Lee et al. 2022, Eq. 3-4).
        residual = z_flattened
        z_q_sum = torch.zeros_like(z_flattened)
        commitment_loss = torch.zeros((), device=z.device, dtype=z_flattened.dtype)
        codebook_loss_grad = torch.zeros((), device=z.device, dtype=z_flattened.dtype)
        first_indices = None
        perplexity_per_depth = []
        used_codes = torch.zeros(self.num_embeddings, device=z.device, dtype=torch.bool)
        collect_ema = self.use_ema and self.training
        if collect_ema:
            ema_counts = torch.zeros(self.num_embeddings, device=z.device)
            ema_sums = torch.zeros(self.num_embeddings, self.embedding_dim, device=z.device)
            ema_res_sq_sum = torch.zeros(self.num_embeddings, device=z.device)

        for d in range(self.rq_depth):
            distances = self._pairwise_distances(residual, codebook)
            encoding_indices = torch.argmin(distances, dim=-1)
            if d == 0:
                first_indices = encoding_indices

            # Codebook health diagnostics, detached (per depth).
            with torch.no_grad():
                one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
                avg_probs = one_hot.mean(dim=0)
                perp_d = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                perplexity_per_depth.append(perp_d.item())
                used_codes |= (avg_probs > 0)
                if d == 0:
                    self.perplexity = perp_d

            # Codebook lookup: raw codeword for this depth.
            z_q_d = codebook[encoding_indices]

            # EMA statistics, accumulated across depths (shared codebook) and applied
            # once after the loop. Forced to fp32 (autocast disabled) so the running
            # statistics never accumulate bf16 rounding error.
            if collect_ema:
                with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
                    oh32 = one_hot.float()
                    res32 = residual.detach().float()
                    ema_counts += oh32.sum(dim=0)
                    ema_sums += oh32.t() @ res32
                    if d == 0:
                        # Depth-1 residual (z - e_k1): the within-component offset that
                        # sigma2 must describe. Isotropic: mean squared over channels.
                        sq = ((res32 - z_q_d.detach().float()) ** 2).mean(dim=1)
                        ema_res_sq_sum.index_add_(0, encoding_indices, sq)

            if not self.use_ema:
                # Gradient codebook: pull each depth's codes toward the residual they
                # quantize. Depth 1 reduces to the original mse(z_q, z.detach()).
                codebook_loss_grad = codebook_loss_grad + mse_loss(z_q_d, residual.detach())

            z_q_sum = z_q_sum + z_q_d
            # Coarse-to-fine commitment (RQ-VAE Eq. 7): commit z to EVERY partial sum
            # so each depth sequentially reduces the quantization error. Depth 1
            # reduces to the original mse(z_q.detach(), z).
            commitment_loss = commitment_loss + mse_loss(z_q_sum.detach(), z_flattened)

            residual = residual - z_q_d

        commitment_loss = self.commitment_cost * commitment_loss
        self.perplexity_per_depth = perplexity_per_depth
        with torch.no_grad():
            self.codebook_usage = used_codes.float().mean()
            self.rel_quant_error = (
                (z_q_sum.detach().float() - z_flattened.detach().float()).pow(2).sum()
                / z_flattened.detach().float().pow(2).sum().clamp(min=1e-12)
            )

        # Apply the (single, decayed) EMA update for this batch.
        if collect_ema:
            with torch.autocast(device_type=z.device.type, enabled=False):
                self._ema_update_from_stats(ema_counts, ema_sums, ema_res_sq_sum,
                                            z_flattened.detach().float())

        # Reshape the summed quantization back to the (B, C, H, W) layout.
        z_q = z_q_sum.reshape(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)

        if self.use_ema:
            # Codes are trained by the EMA update, not by gradients: codebook_loss is
            # returned as a detached diagnostic (total quantization error, same scale
            # as before so runs stay comparable) but EXCLUDED from the optimized loss.
            codebook_loss = mse_loss(z_q.detach(), z.detach())
            loss = commitment_loss
        else:
            codebook_loss = codebook_loss_grad
            loss = codebook_loss + commitment_loss

        # Straight-through estimator trick for gradient backpropagation
        # Ensures gradients flow from z_q to z during backpropagation while using quantized values for the forward pass.
        z_q = z + (z_q - z).detach()

        # encoding_indices returned are the DEPTH-1 codes (the GMM component identity),
        # keeping the (N,) shape callers expect regardless of rq_depth.
        return z_q, loss, first_indices, commitment_loss, codebook_loss
