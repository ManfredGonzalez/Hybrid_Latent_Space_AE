import math
import torch
import torch.nn as nn


class MaskedLatentPredictor(nn.Module):
    """A small ViT-style context predictor for masked latent modeling on the DUALVAE latent.

    It reads the (partially masked) pre-quantization latent grid z_e_vq (B, C, h, w), replaces
    the MASKED locations with a learned mask token, and -- attending over ALL locations -- emits
    a per-location distribution over the K codebook entries (logits). Those logits are consumed
    two ways by the trainer, giving the two experiments a SHARED architecture but different losses:

      * Masked Code Modeling (MCM): softmax(logits) is matched (cross-entropy) to the GMM's own
        soft assignment (responsibilities) at the masked locations -- "predict which code a hidden
        patch is, from its neighbours".
      * Masked Latent GMM Modeling (MLM): softmax(logits) = context-predicted mixture weights
        r_k(context); the TRUE masked z_e_vq is scored under the mixture
        p(z|ctx) = sum_k r_k(ctx) N(e_k, sigma_k^2 I). This is a conditional version of the
        code-centered GMM prior p(z) = sum_k pi_k N(e_k, sigma_k^2 I) -- so the head doubles as a
        native GENERATIVE PRIOR over the latent (sample k ~ r_k, then z ~ N(e_k, sigma_k^2)).

    Gradients flow from the loss through the VISIBLE tokens back into the DUALVAE encoder (that is
    what makes the codes context-predictable = semantic); the prediction TARGET is detached by the
    trainer. The reconstruction/GAN path is never touched -- this is a pure auxiliary head.
    """

    def __init__(self, latent_channels, num_embeddings, grid_hw, dim=256, depth=4, heads=8,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.C = latent_channels
        self.K = num_embeddings
        self.h, self.w = grid_hw
        self.n_loc = self.h * self.w

        self.input_proj = nn.Linear(latent_channels, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_loc, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_embeddings)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, z_e_vq, mask):
        """z_e_vq: (B, C, h, w) -- VISIBLE tokens carry gradient into the encoder.
        mask: (B, n_loc) bool, True where the location is MASKED (hidden from the predictor).
        Returns logits (B, n_loc, K)."""
        b, c, h, w = z_e_vq.shape
        x = z_e_vq.permute(0, 2, 3, 1).reshape(b, h * w, c)     # (B, n_loc, C)
        x = self.input_proj(x)                                  # (B, n_loc, dim)
        # Hide the masked locations: swap their tokens for the shared learned mask embedding
        # so the predictor cannot peek at the answer (no leakage), then let attention infer them.
        x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = x + self.pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)                                     # (B, n_loc, K)


def sample_mask(b, n_loc, ratio, device, generator=None):
    """Random per-image mask, True = masked. Guarantees >=1 visible and >=1 masked per image."""
    k = int(round(ratio * n_loc))
    k = max(1, min(n_loc - 1, k))
    noise = torch.rand(b, n_loc, device=device, generator=generator)
    idx = noise.argsort(dim=1)[:, :k]                           # k lowest-noise locations -> masked
    mask = torch.zeros(b, n_loc, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask
