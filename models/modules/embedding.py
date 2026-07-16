import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, reduction='sum', l2_normalize=False,
                 use_ema=False, ema_decay=0.99, ema_eps=1e-5, ema_dead_threshold=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings # Number of vectors in the codebook
        self.commitment_cost = commitment_cost # Beta, the commitment loss weight
        self.reduction = reduction # How to reduce the loss: 'sum' or 'mean'
        self.l2_normalize = l2_normalize # Normalize codes/lookup to the unit sphere before the distance computation (helps codebook utilization at low dims)

        # --- EMA codebook (van den Oord et al. 2017, App. A.1; Razavi et al. 2019) ---
        # When use_ema=True the codebook is NOT trained by gradients: each code is the
        # running mean of the encoder vectors assigned to it (online k-means), tracked
        # via an EMA count N_k (ema_cluster_size) and an EMA vector-sum m_k (ema_embed_sum),
        # with e_k = m_k / N_k. The codebook_loss term is then excluded from the returned
        # `loss` (commitment only) and kept as a detached diagnostic. Codes whose smoothed
        # count falls below ema_dead_threshold are restarted from random encoder vectors
        # of the current batch (Jukebox-style). use_ema=False keeps the original
        # gradient-trained codebook, bit-identical to previous behavior (for ablations).
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_threshold = ema_dead_threshold

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #Initializes the embedding weights uniformly to help with training stability.
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        if self.use_ema:
            # Codebook is updated in-place under no_grad; freeze it so the optimizer
            # never applies a competing gradient update.
            self.embedding.weight.requires_grad_(False)
            # Persistent so checkpoints resume with consistent statistics. Initialized
            # to N_k=1 and m_k=e_k so e_k = m_k / N_k holds at step 0.
            self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
            self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())
            self.register_buffer('restarted_codes', torch.zeros(()), persistent=False)

        # Codebook health stats from the most recent forward pass (diagnostics only,
        # not used in any loss). perplexity == exp(entropy of code usage): num_embeddings
        # when every code is used equally often, 1 when only one code is ever picked.
        # codebook_usage == fraction of codes used at least once in the batch.
        self.register_buffer('perplexity', torch.zeros(()), persistent=False)
        self.register_buffer('codebook_usage', torch.zeros(()), persistent=False)

    @torch.no_grad()
    def _ema_update(self, z_flattened, encoding_indices):
        """One EMA codebook step from this batch's assignments.

        z_flattened: (N, D) raw (un-normalized) encoder vectors, fp32 recommended.
        Even with l2_normalize=True the RAW vectors are averaged -- mirroring forward(),
        which normalizes only for the nearest-neighbor lookup but returns raw codewords.
        """
        one_hot = F.one_hot(encoding_indices, self.num_embeddings).to(z_flattened.dtype)  # (N, K)
        batch_cluster_size = one_hot.sum(dim=0)                    # n_k: vectors per code this batch
        batch_embed_sum = one_hot.t() @ z_flattened                # s_k: their sum, (K, D)

        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(batch_cluster_size, alpha=1 - decay)  # N_k
        self.ema_embed_sum.mul_(decay).add_(batch_embed_sum, alpha=1 - decay)        # m_k

        # Laplace smoothing of the counts so m_k / N_k never divides by ~0.
        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (
            (self.ema_cluster_size + self.ema_eps) / (n + self.num_embeddings * self.ema_eps) * n
        )
        self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1))

        # Dead-code restart: codes whose smoothed usage collapsed get teleported onto
        # random encoder vectors from the current batch, and their EMA stats are re-seeded
        # at the batch-average count so they aren't immediately flagged dead again.
        dead = self.ema_cluster_size < self.ema_dead_threshold
        num_dead = int(dead.sum().item())
        self.restarted_codes = torch.tensor(float(num_dead), device=z_flattened.device)
        if num_dead > 0:
            rand_idx = torch.randint(0, z_flattened.size(0), (num_dead,), device=z_flattened.device)
            new_codes = z_flattened[rand_idx]
            avg_size = self.ema_cluster_size.mean().clamp(min=1.0)
            self.embedding.weight.data[dead] = new_codes
            self.ema_embed_sum[dead] = new_codes * avg_size
            self.ema_cluster_size[dead] = avg_size

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
            if self.l2_normalize:
                z_cmp = F.normalize(z_flattened, dim=-1)
                cb_cmp = F.normalize(centroids, dim=-1)
                distances = 2.0 - 2.0 * torch.matmul(z_cmp, cb_cmp.t())
            else:
                distances = (
                    torch.sum(z_flattened ** 2, dim=-1, keepdim=True)
                    + torch.sum(centroids ** 2, dim=-1).unsqueeze(0)
                    - 2 * torch.matmul(z_flattened, centroids.t())
                )
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
        # the pre-init state (e_k = m_k / N_k must hold at handoff).
        if self.use_ema:
            counts = torch.bincount(assignments, minlength=self.num_embeddings).to(centroids.dtype).clamp(min=1.0)
            self.ema_cluster_size.copy_(counts)
            self.ema_embed_sum.copy_(centroids * counts.unsqueeze(1))

    def forward(self, z):
        b, c, h, w = z.shape
        z_channel_last = z.permute(0, 2, 3, 1) # (B, H, W, C)
        z_flattened = z_channel_last.reshape(b*h*w, self.embedding_dim)

        codebook = self.embedding.weight

        if self.l2_normalize:
            # Cosine-similarity nearest-neighbor search only: for unit vectors
            # ||a-b||^2 = 2 - 2*a.b. The returned z_q below is still the RAW
            # codeword and z is untouched, so the rest of the pipeline (residual
            # addition, SWD, decoder) sees the same raw-scale space as when
            # l2_normalize is off -- only which code gets picked changes.
            z_cmp = F.normalize(z_flattened, dim=-1)
            cb_cmp = F.normalize(codebook, dim=-1)
            distances = 2.0 - 2.0 * torch.matmul(z_cmp, cb_cmp.t())
        else:
            # Calculate distances between z and the codebook embeddings |a-b|²
            # Efficient computation of Euclidean distances between the input vectors and codebook entries using the identity
            distances = (
                torch.sum(z_flattened ** 2, dim=-1, keepdim=True)                 # a²
                + torch.sum(codebook.t() ** 2, dim=0, keepdim=True)  # b²
                - 2 * torch.matmul(z_flattened, codebook.t())        # -2ab
            )

        # Get the index with the smallest distance
        # Vector Quantization: Selects the index of the closest codebook vector for each input patch (quantization step).
        encoding_indices = torch.argmin(distances, dim=-1)

        # Codebook health diagnostics, detached: how many distinct codes fired this
        # batch, and how uniformly. Cheap to compute (num_embeddings-sized histogram).
        with torch.no_grad():
            one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
            avg_probs = one_hot.mean(dim=0)
            self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            self.codebook_usage = (avg_probs > 0).float().mean()

        # Get the quantized vector
        # Codebook Lookup & Reshape
        # Codebook loss: Encourages codebook embeddings to match encoder outputs
        # Commitment loss: Encourages encoder outputs to commit to codebook entries
        # Retrieves quantized vectors (z_q) from the codebook using the selected indices and reshapes them to the original z format.
        z_q = codebook[encoding_indices]
        z_q = z_q.reshape(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)

        # EMA codebook step (training mode only; uses this batch's assignments). Runs in
        # fp32 regardless of autocast so the running statistics don't accumulate bf16
        # rounding error. Eval mode never touches the statistics.
        if self.use_ema and self.training:
            self._ema_update(z_flattened.detach().float(), encoding_indices)

        # Calculate the commitment loss
        mse_loss = nn.MSELoss(reduction=self.reduction)

        commitment_loss = self.commitment_cost * mse_loss(z_q.detach(), z)
        if self.use_ema:
            # Codes are trained by the EMA update, not by gradients: codebook_loss is
            # returned as a detached diagnostic (quantization error, same scale as
            # before so runs stay comparable) but EXCLUDED from the optimized loss.
            codebook_loss = mse_loss(z_q.detach(), z.detach())
            loss = commitment_loss
        else:
            codebook_loss = mse_loss(z_q, z.detach())
            loss = codebook_loss + commitment_loss

        # Straight-through estimator trick for gradient backpropagation
        # Ensures gradients flow from z_q to z during backpropagation while using quantized values for the forward pass.
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices, commitment_loss, codebook_loss