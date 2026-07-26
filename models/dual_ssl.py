"""
DualSSL -- the GMM-DualVAE codebook turned into a SwAV self-supervised learner.

The whole point of this file: *with very few changes* the DualVAE architecture becomes an
SSL algorithm. We reuse, unchanged, two pieces of the reconstruction model:

  * DUALVAE_Encoder  -- the exact same encoder trunk.
  * VQEmbedding      -- the exact same codebook, now read as a bank of SwAV *prototypes*.

Everything reconstruction-specific (decoder, KL, commitment, reconstruction loss) is simply
DROPPED. In its place we add SwAV (Caron et al. 2020): two augmented views of an image are
each soft-assigned to the prototypes, and we predict one view's assignment from the other
("swapped prediction"). Anti-collapse -- which reconstruction used to provide implicitly --
now comes from Sinkhorn-Knopp equipartition of the assignments. No decoder, no negatives.

Two embedding modes (config `embedding_mode`):
  * "global" (default): mean-pool the encoder feature map to one vector per image, then
    assign that vector to the prototypes -> one distribution over codes per image.
  * "dense": assign every spatial location to the prototypes and average the resulting
    distributions -> a soft "bag-of-codes" per image (pooling in assignment space instead of
    feature space). Avoids the cross-view spatial-correspondence problem of dense SSL.

The codebook (VQEmbedding.embedding.weight) is used only as the K x d prototype matrix; the
VQ hard-quantization / EMA machinery is NOT invoked here -- assignment is soft (cosine +
softmax), which is what SwAV needs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import DUALVAE_Encoder
from .modules.embedding import VQEmbedding

EMBEDDING_MODES = {"global", "dense"}
ASSIGNMENTS = {"cosine", "gaussian"}


class DualSSL(nn.Module):
    def __init__(self, latent_channels=8, downsample_factor=8, num_prototypes=256,
                 embedding_mode="global", proj_dim=None, proj_hidden_dim=256,
                 l2_normalize_codes=True, temperature=0.1,
                 sinkhorn_eps=0.05, sinkhorn_iters=3,
                 assignment="cosine", ema_decay=0.99,
                 sigma2_floor=0.1, sigma2_ceil=10.0):
        super().__init__()
        if embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"embedding_mode must be one of {sorted(EMBEDDING_MODES)}, got {embedding_mode!r}.")
        if assignment not in ASSIGNMENTS:
            raise ValueError(f"assignment must be one of {sorted(ASSIGNMENTS)}, got {assignment!r}.")
        self.embedding_mode = embedding_mode
        self.temperature = temperature
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        # --- NOVEL: per-prototype variance (GMM) assignment ---
        # "cosine" = plain SwAV (single global temperature tau).
        # "gaussian" = per-prototype sigma_k^2 sharpens the prediction softmax
        #   (logit_k = cos_k / sigma_k^2, normalized so all-equal sigma == cosine/tau).
        #   sigma_k^2 is an EMA of the within-cluster (chord) spread -- the SAME per-code
        #   variance the DualVAE codebook models, now controlling the SSL assignment. This
        #   is what lets one GMM codebook serve both the generative prior and the SSL
        #   prototypes (the duality). sigma2_floor is the anti-collapse guardrail.
        self.assignment = assignment
        self.ema_decay = ema_decay
        self.sigma2_floor = sigma2_floor
        self.sigma2_ceil = sigma2_ceil

        trunk_channels = 2 * latent_channels
        # --- REUSED, unchanged from DUALVAE ---
        self.encoder = DUALVAE_Encoder(downsample_factor=downsample_factor, out_channels=trunk_channels)
        self.bottle_neck = nn.Conv2d(trunk_channels, latent_channels, kernel_size=1, padding=0)

        # Optional SwAV-style projection head. Off by default => the C-dim codebook is used
        # directly as prototypes (maximal reuse). Set proj_dim to add a small MLP head.
        feat_dim = latent_channels
        if proj_dim:
            self.projector = nn.Sequential(
                nn.Linear(latent_channels, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim), nn.GELU(),
                nn.Linear(proj_hidden_dim, proj_dim),
            )
            feat_dim = proj_dim
        else:
            self.projector = None
        self.feat_dim = feat_dim

        # --- REUSED codebook, now the SwAV prototype bank (gradient-trained: use_ema=False) ---
        self.prototypes = VQEmbedding(num_embeddings=num_prototypes, embedding_dim=feat_dim,
                                      l2_normalize=l2_normalize_codes, use_ema=False)
        self.num_prototypes = num_prototypes
        # Per-prototype variance (EMA of within-cluster chord spread). Init to 1.0 so the
        # gaussian assignment starts exactly equal to plain cosine/tau, then differentiates.
        self.register_buffer("proto_sigma2", torch.ones(num_prototypes))
        self.register_buffer("proto_count", torch.zeros(num_prototypes))

    # ------------------------------------------------------------------ #
    def _backbone(self, x):
        """Encoder trunk -> C-channel latent grid (B, C, h, w). Same path as DualVAE."""
        return self.bottle_neck(self.encoder(x))

    def _norm_prototypes(self):
        return F.normalize(self.prototypes.embedding.weight, dim=1)   # (K, feat_dim)

    def forward_scores(self, x):
        """One view -> (B, K) prototype scores (cosine similarity), image-level.

        global: pool features then assign. dense: assign per location then pool the scores.
        """
        z = self._backbone(x)                                   # (B, C, h, w)
        b, c, h, w = z.shape
        protos = self._norm_prototypes()                        # (K, feat_dim)
        if self.embedding_mode == "global":
            feat = z.mean(dim=(2, 3))                           # (B, C)
            if self.projector is not None:
                feat = self.projector(feat)
            feat = F.normalize(feat, dim=1)
            return feat @ protos.t()                            # (B, K)
        # dense: per-location cosine, averaged to an image-level score.
        loc = z.permute(0, 2, 3, 1).reshape(b * h * w, c)
        if self.projector is not None:
            loc = self.projector(loc)
        loc = F.normalize(loc, dim=1)
        loc_scores = loc @ protos.t()                           # (B*hw, K)
        return loc_scores.reshape(b, h * w, -1).mean(dim=1)     # (B, K)

    @torch.no_grad()
    def sinkhorn(self, scores):
        """Sinkhorn-Knopp: turn scores (B, K) into balanced soft assignments (B, K).
        Equipartition over prototypes is the anti-collapse valve reconstruction used to give."""
        Q = torch.exp(scores.float() / self.sinkhorn_eps).t()   # (K, B)
        Q /= Q.sum().clamp(min=1e-12)
        K, B = Q.shape
        for _ in range(self.sinkhorn_iters):
            Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-12)     # rows (prototypes) -> uniform
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-12)     # cols (samples) -> uniform
            Q /= B
        Q *= B                                                  # columns sum to 1
        return Q.t()                                            # (B, K)

    def _inv_var(self):
        """Per-prototype inverse variance, normalized so its mean equals 1/temperature.
        With all sigma_k equal this returns a uniform 1/tau, i.e. plain SwAV exactly."""
        s2 = self.proto_sigma2.clamp(self.sigma2_floor, self.sigma2_ceil)
        inv = 1.0 / s2
        return inv / inv.mean().clamp(min=1e-12) / self.temperature      # (K,)

    def _pred_logits(self, cos):
        """Prediction-side logits from raw cosine scores (B, K).
        cosine: cos / tau (single temperature). gaussian: cos * inv_var_k (per-prototype)."""
        if self.assignment == "gaussian":
            return cos * self._inv_var().unsqueeze(0)
        return cos / self.temperature

    @torch.no_grad()
    def _ema_update_sigma2(self, cos, q):
        """M-step: EMA the per-prototype within-cluster chord spread d2 = 2 - 2*cos, weighted
        by the (Sinkhorn) soft assignment q. Detached; clamped to [floor, ceil]."""
        d2 = (2.0 - 2.0 * cos).clamp(min=0.0)                   # (B, K) squared chord distance
        num = (q * d2).sum(0)                                   # (K,)
        den = q.sum(0)                                          # (K,) soft counts
        dec = self.ema_decay
        self.proto_count.mul_(dec).add_(den, alpha=1 - dec)
        mask = den > 1e-3
        batch_var = torch.zeros_like(self.proto_sigma2)
        batch_var[mask] = (num[mask] / den[mask])
        updated = dec * self.proto_sigma2 + (1 - dec) * batch_var
        self.proto_sigma2[mask] = updated[mask].clamp(self.sigma2_floor, self.sigma2_ceil)

    def swav_loss(self, scores_list, n_targets=None):
        """Swapped-prediction loss over a list of per-view (B, K) *raw cosine* scores.

        Targets (Sinkhorn assignments) come from the first `n_targets` views (global crops);
        every OTHER view must predict them. Sinkhorn is identical in both assignment modes
        (balanced cluster targets); only the prediction sharpness changes (per-prototype in
        gaussian mode). In gaussian mode the per-prototype sigma_k^2 is EMA-updated here from
        the target views. For 2 views this is CE(q0, p1) + CE(q1, p0).
        """
        n_views = len(scores_list)
        if n_targets is None:
            n_targets = n_views
        total, n_terms = 0.0, 0
        for i in range(n_targets):
            q = self.sinkhorn(scores_list[i])                   # target assignment (raw cos)
            if self.assignment == "gaussian" and self.training:
                self._ema_update_sigma2(scores_list[i], q)
            for j in range(n_views):
                if i == j:
                    continue
                logp = F.log_softmax(self._pred_logits(scores_list[j]), dim=1)
                total = total - torch.mean(torch.sum(q * logp, dim=1))
                n_terms += 1
        return total / max(n_terms, 1)

    @torch.no_grad()
    def sigma2_stats(self):
        """Diagnostics for the per-prototype variance (gaussian mode)."""
        s2 = self.proto_sigma2.clamp(self.sigma2_floor, self.sigma2_ceil)
        return {"Sigma2/Mean": s2.mean().item(), "Sigma2/Min": s2.min().item(),
                "Sigma2/Max": s2.max().item(),
                "Sigma2/AtFloorFrac": (s2 <= self.sigma2_floor * 1.001).float().mean().item()}

    @torch.no_grad()
    def load_codebook_from_dualvae(self, ckpt_path, device="cpu"):
        """Duality / transfer: seed the SSL prototypes (and their sigma_k^2) from a trained
        DualVAE codebook. Requires matching (num_prototypes, feat_dim) -- i.e. proj_dim off so
        the prototype dim equals the codebook's latent_channels."""
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        w = sd["vq_layer.embedding.weight"]                    # (K, C)
        if tuple(w.shape) != tuple(self.prototypes.embedding.weight.shape):
            raise ValueError(f"codebook shape {tuple(w.shape)} != prototype shape "
                             f"{tuple(self.prototypes.embedding.weight.shape)}; disable proj_dim "
                             "and match num_prototypes/latent_channels to transfer.")
        self.prototypes.embedding.weight.data.copy_(w.to(self.prototypes.embedding.weight.device))
        # Seed sigma_k^2 from the generative EMA within-component variance, if present.
        if "vq_layer.ema_res_sq" in sd and "vq_layer.ema_cluster_size" in sd:
            v = sd["vq_layer.ema_res_sq"].float()
            n = sd["vq_layer.ema_cluster_size"].float().clamp(min=1e-5)
            s2 = (v / n).clamp(self.sigma2_floor, self.sigma2_ceil)
            self.proto_sigma2.data.copy_(s2.to(self.proto_sigma2.device))
        print(f"[transfer] seeded {w.shape[0]} prototypes from {ckpt_path}"
              + (" (+sigma_k^2)" if "vq_layer.ema_res_sq" in sd else ""))

    @torch.no_grad()
    def forward_eval(self, x):
        """Frozen-feature readout for the val monitor:
        returns (representation (B, C) L2-normalized pooled backbone feature, scores (B, K)).
        The representation is taken BEFORE the projection head (standard SSL eval feature)."""
        z = self._backbone(x)
        rep = F.normalize(z.mean(dim=(2, 3)), dim=1)            # (B, C)
        scores = self.forward_scores(x)                        # (B, K)
        return rep, scores

    def forward(self, views):
        """views: list of image tensors (each (B, 3, H, W)). Returns the SwAV loss."""
        scores_list = [self.forward_scores(v) for v in views]
        return self.swav_loss(scores_list)
