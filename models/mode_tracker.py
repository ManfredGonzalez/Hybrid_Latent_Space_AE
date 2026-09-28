"""Image-level modes for a mixture (MM-FM) flow source, obtained online and without a loss.

WHAT THIS IS FOR. MM-FM (Luo et al., CVPR 2026) replaces the standard N(0, I) flow source with
a Gaussian mixture fitted to the data and pairs each data latent with a sample from ITS OWN
mode. That needs three things per mode m: a weight c_m, a mean mu_m and a (diagonal) variance
sigma2_m -- all in the space the flow trains in -- plus an assignment rule that is consistent
between training and sampling.

WHY THERE IS NO LOSS. With an EMA codebook the codewords are a running mean of the vectors
assigned to them; the only gradient term in a VQ bottleneck is the COMMITMENT loss, and that
one acts on the encoder. Here the input is the frozen DINOv2 [CLS] token, so there is no
encoder to pull: no commitment loss, no straight-through estimator, nothing added to the
objective. This module is a statistics tracker that happens to live inside the training loop.
The update rule below is exactly one step of mini-batch k-means.

WHY THE ASSIGNMENT IS CONSISTENT. MM-FM's coupling is only valid if the distribution of source
samples at sampling time equals their marginal under the training coupling. With a
deterministic hard assignment and c_m defined as the RUNNING FREQUENCY of assignments, that
holds by construction -- which is the reason a quantizer fits this job.

MEASURED CONTEXT (reports/mmfm_mode_tests/results.json, 256k ImageNet images):
  * Streamed EMA-VQ over [CLS] matches OFFLINE k-means: R_cls 0.538 vs 0.538 at M=1024,
    0.415 vs 0.397 at M=8192, NMI 0.91-0.93. One pass is enough; 3 epochs barely moved it.
  * ONLY with the dead-code threshold scaled to M. At the VQEmbedding default (1.0) the
    codebook thrashed -- 304,500 restarts in one epoch at M=1024 -- and usage perplexity
    collapsed to 3-15% of M. At 0.1 * batch / M it settled (120 restarts).
  * On the CURRENT (pre-REPA-E) latent these modes explain ~0% of the latent's variance, so
    the mixture source is not expected to help YET. This module exists so that the statistics
    are already tracked when the decision gate (see reports/mmfm_repae_design.pdf) is reached.

TWO TRACKERS, TWO TIME CONSTANTS. The codebook lives in 768-d and every code is hit often
enough to use a short decay. The per-mode LATENT statistics live in 8192-d and each mode is hit
~0.03 times per step (batch 256, M=8192), so they use a much slower decay -- or, by default,
exact cumulative averaging, which has no time constant to tune and is what you want for
statistics of a slowly drifting encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.distributed import all_reduce_sum_, is_main_process


class ModeTracker(nn.Module):
    """EMA-VQ over a frozen global descriptor + per-mode statistics of the flow latent.

    Everything is a persistent buffer, so the checkpoint IS the fitted mixture: no offline
    clustering pass, no post-hoc fit, and resuming never re-derives the modes (which would
    silently redefine every mode and invalidate the statistics accumulated so far).

    Args:
        num_modes: M, the number of image-level modes.
        descriptor_dim: dimension of the global descriptor (768 for DINOv2-B [CLS]).
        latent_shape: (C, H, W) of the latent whose statistics are tracked.
        decay: EMA decay for the CODEBOOK.
        dead_threshold_frac: dead-code threshold as a fraction of the expected per-step count
            (batch / M). 0.1 is the measured-good value; see the module docstring.
        stats_decay: EMA decay for the per-mode latent statistics; None (default) means exact
            cumulative averaging.
        var_shrinkage: kappa in sigma2_m = (n_m v_m + kappa v_global) / (n_m + kappa). Keeps a
            mode with three samples from reporting a fake-tiny variance.
        num_classes: size of the p(m | y) table, or 0 to disable it. The table is what makes
            class-conditional sampling draw a mode consistent with the requested label.
    """

    def __init__(self, num_modes, descriptor_dim, latent_shape, decay=0.99,
                 dead_threshold_frac=0.1, stats_decay=None, var_shrinkage=10.0,
                 num_classes=1000, eps=1e-5, init_buffer_factor=4):
        super().__init__()
        if num_modes < 2:
            raise ValueError(f"num_modes must be >= 2, got {num_modes}.")
        self.num_modes = int(num_modes)
        self.descriptor_dim = int(descriptor_dim)
        self.latent_shape = tuple(latent_shape)
        self.latent_dim = int(torch.tensor(self.latent_shape).prod())
        self.decay = decay
        self.stats_decay = stats_decay
        self.dead_threshold_frac = dead_threshold_frac
        self.var_shrinkage = var_shrinkage
        self.num_classes = int(num_classes)
        self.eps = eps
        self.init_buffer_target = int(init_buffer_factor * num_modes)

        # --- codebook over the global descriptor (cosine space: rows are unit-norm) --------
        self.register_buffer("codebook", torch.zeros(self.num_modes, self.descriptor_dim))
        self.register_buffer("ema_count", torch.zeros(self.num_modes))
        self.register_buffer("ema_sum", torch.zeros(self.num_modes, self.descriptor_dim))

        # --- per-mode statistics of the (normalized) latent --------------------------------
        self.register_buffer("mode_count", torch.zeros(self.num_modes))
        self.register_buffer("mode_sum", torch.zeros(self.num_modes, self.latent_dim))
        self.register_buffer("mode_sumsq", torch.zeros(self.num_modes, self.latent_dim))
        self.register_buffer("global_count", torch.zeros(()))
        self.register_buffer("global_sum", torch.zeros(self.latent_dim))
        self.register_buffer("global_sumsq", torch.zeros(self.latent_dim))

        # --- class-conditional mode table --------------------------------------------------
        if self.num_classes > 0:
            self.register_buffer("class_mode_count", torch.zeros(self.num_classes, self.num_modes))

        # --- state flags (buffers so they survive checkpoint/resume) -----------------------
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.register_buffer("restarts_last_update", torch.zeros(()))
        self._init_buffer = []          # transient, dropped after initialization

    # ------------------------------------------------------------------ views for sampling #

    @property
    def weights(self):
        """c_m = running frequency of assignments. Zero for modes never used."""
        n = self.mode_count
        return n / n.sum().clamp(min=1e-12)

    @property
    def means(self):
        """(M, D) mu_m; modes with no samples fall back to the global mean."""
        n = self.mode_count.clamp(min=1e-12).unsqueeze(1)
        mu = self.mode_sum / n
        empty = (self.mode_count == 0).unsqueeze(1)
        return torch.where(empty, self.global_mean.unsqueeze(0).expand_as(mu), mu)

    @property
    def global_mean(self):
        return self.global_sum / self.global_count.clamp(min=1e-12)

    @property
    def global_var(self):
        m = self.global_mean
        return (self.global_sumsq / self.global_count.clamp(min=1e-12) - m ** 2).clamp(min=1e-8)

    @property
    def variances(self):
        """(M, D) sigma2_m, shrunk toward the global variance by `var_shrinkage`."""
        n = self.mode_count.clamp(min=1e-12).unsqueeze(1)
        mu = self.mode_sum / n
        raw = (self.mode_sumsq / n - mu ** 2).clamp(min=0.0)
        k = self.var_shrinkage
        v_glob = self.global_var.unsqueeze(0)
        return (self.mode_count.unsqueeze(1) * raw + k * v_glob) / (self.mode_count.unsqueeze(1) + k)

    def class_mode_probs(self, smoothing=1e-3):
        """(num_classes, M) p(m | y). Smoothed so an unseen (class, mode) pair is merely
        unlikely rather than impossible -- a hard zero would make some labels unsamplable."""
        if self.num_classes <= 0:
            raise RuntimeError("ModeTracker was built with num_classes=0; no p(m|y) table.")
        table = self.class_mode_count + smoothing
        return table / table.sum(dim=1, keepdim=True)

    def mixture_params(self):
        """(c, mu, sigma2) ready for MM-FM sampling. Detached; these are estimates, not
        parameters, and nothing downstream should backpropagate into them."""
        return self.weights.detach(), self.means.detach(), self.variances.detach()

    # -------------------------------------------------------------------------- assignment #

    @staticmethod
    def _normalize(descriptor):
        return F.normalize(descriptor.float(), dim=-1)

    @torch.no_grad()
    def assign(self, descriptor):
        """(B, descriptor_dim) -> (B,) mode index, by cosine similarity. Hard assignment.

        MM-FM's default is soft assignment, but they report the two behave similarly in high
        dimensions and their theory is written for the hard version; hard assignment is also
        what makes the training/sampling consistency exact.
        """
        if not bool(self.initialized):
            raise RuntimeError("ModeTracker.assign() called before initialization.")
        d = self._normalize(descriptor)
        return (d @ self.codebook.t()).argmax(dim=1)

    # ---------------------------------------------------------------------- initialization #

    @torch.no_grad()
    def observe_for_init(self, descriptor):
        """Buffer descriptors until there are enough to seed the codebook; then seed it.

        Returns True once initialization has happened. Deliberately seeded from the DATA
        STREAM (the first ~4M vectors, i.e. ~64 steps at M=4096 and batch 256) rather than
        from a pass over the dataset -- the point of this module is that no offline pass is
        needed. Rank 0 seeds and broadcasts, exactly like the trainers' k-means init, so all
        ranks share one codebook (otherwise the all-reduced counts below are meaningless).
        """
        if bool(self.initialized):
            return True
        # Gather across ranks BEFORE buffering. Buffering only this rank's shard would need
        # world_size times as many steps to reach the target (at 32 per rank and M=4096:
        # 512 steps instead of 64), which in a short run means the tracker never starts at all.
        self._init_buffer.append(_all_gather_cat(self._normalize(descriptor)).cpu())
        have = sum(t.shape[0] for t in self._init_buffer)
        if have < self.init_buffer_target:
            return False

        buf = torch.cat(self._init_buffer, dim=0).to(self.codebook.device)
        self._init_buffer = []
        codes = _kmeans_plusplus_cosine(buf, self.num_modes, iters=10)
        self.codebook.copy_(F.normalize(codes, dim=-1))
        self.ema_count.fill_(1.0)
        self.ema_sum.copy_(self.codebook)
        _broadcast_(self.codebook, self.ema_count, self.ema_sum)
        self.initialized.fill_(True)
        if is_main_process():
            print(f"[modes] codebook seeded from {buf.shape[0]} descriptors "
                  f"(M={self.num_modes}, dim={self.descriptor_dim})")
        return True

    # ---------------------------------------------------------------------------- updating #

    @torch.no_grad()
    def update(self, descriptor, latent, labels=None, batch_size_hint=None):
        """One tracker step. `latent` must be ALREADY NORMALIZED and DETACHED.

        Order matters: assignment uses the codebook as it was at the start of the step, the
        latent statistics are keyed on that assignment, and only then does the codebook move.
        Returns a dict of diagnostics (cheap scalars only).
        """
        if not bool(self.initialized):
            raise RuntimeError("ModeTracker.update() called before initialization.")
        # GATHER THE BATCH, DO NOT REDUCE THE BUFFERS. The natural implementation accumulates
        # into (M, latent_dim) temporaries and all-reduces those; at M=4096 and 8192 dims that
        # is 134 MB per tensor per step. Gathering the batch instead moves B x 8192 floats
        # (8 MB at B=256) and every rank then applies the identical full-batch update, which
        # keeps the buffers bit-identical across ranks without ever putting one on the wire.
        d = _all_gather_cat(self._normalize(descriptor))
        z = _all_gather_cat(latent.detach().float().reshape(latent.shape[0], -1))
        idx = (d @ self.codebook.t()).argmax(dim=1)
        if self.num_classes > 0 and labels is not None:
            labels = _all_gather_cat(labels.long())

        counts = torch.bincount(idx, minlength=self.num_modes).float()
        batch_total = float(idx.numel())
        ones = torch.ones(idx.numel(), device=z.device, dtype=z.dtype)

        # --- latent statistics (cumulative by default; see the module docstring) -----------
        # index_add_ straight into the buffers: no (M, latent_dim) temporary is ever allocated.
        if self.stats_decay is None:
            self.mode_count.index_add_(0, idx, ones)
            self.mode_sum.index_add_(0, idx, z)
            self.mode_sumsq.index_add_(0, idx, z ** 2)
            self.global_count.add_(batch_total)
            self.global_sum.add_(z.sum(dim=0))
            self.global_sumsq.add_((z ** 2).sum(dim=0))
        else:
            s = self.stats_decay
            self.mode_count.mul_(s).index_add_(0, idx, ones * (1 - s))
            self.mode_sum.mul_(s).index_add_(0, idx, z * (1 - s))
            self.mode_sumsq.mul_(s).index_add_(0, idx, (z ** 2) * (1 - s))
            self.global_count.mul_(s).add_(batch_total * (1 - s))
            self.global_sum.mul_(s).add_(z.sum(dim=0), alpha=1 - s)
            self.global_sumsq.mul_(s).add_((z ** 2).sum(dim=0), alpha=1 - s)
        if self.num_classes > 0 and labels is not None:
            self.class_mode_count.index_put_((labels, idx),
                                             torch.ones_like(ones), accumulate=True)

        # The codebook side is only (M, 768) = 12 MB, so a dense temporary is fine here.
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, idx, d)

        # --- codebook (skipped once frozen: the descriptor is frozen, so the partition
        #     converges in ~1 epoch and Phase 2 wants it to stop moving) --------------------
        restarts = 0.0
        if not bool(self.frozen):
            dec = self.decay
            self.ema_count.mul_(dec).add_(counts, alpha=1 - dec)
            self.ema_sum.mul_(dec).add_(sums, alpha=1 - dec)
            n = self.ema_count.clamp(min=self.eps).unsqueeze(1)
            self.codebook.copy_(F.normalize(self.ema_sum / n, dim=-1))
            restarts = self._restart_dead_codes(d, batch_size_hint or batch_total)
        self.restarts_last_update.fill_(restarts)

        return {
            "modes/batch_unique": float(torch.unique(idx).numel()),
            "modes/restarts": restarts,
            "modes/usage_perplexity": float(self._usage_perplexity()),
            "modes/nonempty_frac": float((self.mode_count > 0).float().mean()),
        }

    @torch.no_grad()
    def _restart_dead_codes(self, descriptors, batch_size):
        """Re-seed codes whose EMA count fell below 0.1 * (batch / M) with real descriptors.

        The threshold MUST scale with M. At a fixed 1.0 (the VQEmbedding default) essentially
        every code looks dead -- each one is expected to win only batch/M assignments per step
        -- and the codebook thrashes; measured: 304,500 restarts/epoch vs 120 once scaled.
        """
        expected = max(batch_size, 1.0) / self.num_modes
        threshold = self.dead_threshold_frac * expected
        dead = (self.ema_count < threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return 0.0
        n_take = min(dead.numel(), descriptors.shape[0])
        if n_take == 0:
            return 0.0
        dead = dead[:n_take]
        # Rank 0 chooses, then broadcasts: independent choices per rank would fork the
        # codebooks that the all-reduce above assumes are identical.
        pick = torch.randperm(descriptors.shape[0], device=descriptors.device)[:n_take]
        new_codes = descriptors[pick]
        _broadcast_(new_codes)
        self.codebook[dead] = F.normalize(new_codes, dim=-1)
        self.ema_sum[dead] = self.codebook[dead]
        self.ema_count[dead] = expected
        return float(n_take)

    def _usage_perplexity(self):
        p = self.weights
        return torch.exp(-(p * (p + 1e-12).log()).sum())

    @torch.no_grad()
    def freeze_codebook(self):
        """Stop moving the codes; keep assigning and keep updating the latent statistics."""
        self.frozen.fill_(True)

    @torch.no_grad()
    def reset_latent_statistics(self):
        """Drop the accumulated latent statistics, keeping the partition.

        Use after the encoder has moved a lot (e.g. at the end of end-to-end training) to
        re-measure mu_m and sigma2_m on the CURRENT latent instead of an average over the
        whole trajectory.
        """
        self.mode_count.zero_(); self.mode_sum.zero_(); self.mode_sumsq.zero_()
        self.global_count.zero_(); self.global_sum.zero_(); self.global_sumsq.zero_()


# --------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------- #

def _broadcast_(*tensors, src=0):
    """Broadcast in place from rank `src`; a no-op outside DDP."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.broadcast(t, src=src)


def _all_gather_cat(x):
    """Concatenate `x` across ranks (a no-op outside DDP).

    Every rank ends up with the whole batch and applies the same update, which is what keeps
    the tracker's buffers identical across ranks. Requires equal shapes per rank -- true here
    because the training loader uses drop_last=True.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return x
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def _kmeans_plusplus_cosine(x, k, iters=10, chunk=8192):
    """Cosine k-means with k-means++ seeding on unit-norm rows. Returns (k, d) centroids.

    Chunked so the (N, k) similarity matrix never materializes in full: at N=16k and k=8192
    that would already be 0.5 GB in fp32, and this runs inside the training process.
    """
    n, d = x.shape
    if n < k:
        raise ValueError(f"need at least k={k} descriptors to seed the codebook, got {n}.")
    centroids = torch.empty(k, d, device=x.device, dtype=x.dtype)
    centroids[0] = x[torch.randint(n, (1,), device=x.device)]
    # k-means++: each new centre is drawn with probability proportional to its squared
    # distance from the nearest existing centre, which spreads the seeds over the data
    # instead of clumping them where the density is highest.
    closest = 1.0 - (x @ centroids[0].unsqueeze(1)).squeeze(1)
    for i in range(1, k):
        probs = closest.clamp(min=0) + 1e-12
        centroids[i] = x[torch.multinomial(probs, 1)]
        closest = torch.minimum(closest, 1.0 - (x @ centroids[i].unsqueeze(1)).squeeze(1))

    for _ in range(iters):
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            assign[start:stop] = (x[start:stop] @ centroids.t()).argmax(dim=1)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=x.device, dtype=x.dtype)
        sums.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones(n, device=x.device, dtype=x.dtype))
        empty = counts == 0
        centroids = F.normalize(sums / counts.clamp(min=1).unsqueeze(1), dim=-1)
        if empty.any():                      # re-seed empties from random points
            centroids[empty] = x[torch.randint(n, (int(empty.sum()),), device=x.device)]
    return centroids
