from .modules.embedding import VQEmbedding
from .modules.encoder import DUALVAE_Encoder
from .modules.decoder import DUALVAE_Decoder
from .modules.attention import AttentionBlock, SpatialCrossAttentionBlock
from .modules.cont_dropout import validate_cont_dropout_p, apply_cont_dropout

import torch.nn as nn
import torch

# Enum-style set of valid continuous_mode values. Add new modes (e.g. "fixed_noise") here.
CONTINUOUS_MODES = {"learned_variance", "deterministic"}


def validate_continuous_mode(mode):
    if mode not in CONTINUOUS_MODES:
        raise ValueError(f"continuous_mode must be one of {sorted(CONTINUOUS_MODES)}, got {mode!r}.")


class SW_DUALVAE(nn.Module):
    def __init__(self, num_embeddings=512, latent_channels=8, commitment_cost=0.25, downsample_factor=8, combine_mode='cross_attention', l2_normalize_codes=False, cont_dropout_p=0.0, continuous_mode='learned_variance',
                 use_ema_codebook=False, ema_decay=0.99, ema_eps=1e-5, ema_dead_threshold=1.0,
                 rq_depth=1, residual_continuous=False, component_prior=False, sigma2_floor=1e-3, sigma2_ceil=10.0):
        super(SW_DUALVAE, self).__init__()
        validate_cont_dropout_p(cont_dropout_p)
        validate_continuous_mode(continuous_mode)
        if component_prior and not use_ema_codebook:
            raise ValueError("component_prior=True needs the EMA statistics (sigma_k^2 = v_k/N_k); set use_ema_codebook=True.")
        if residual_continuous and combine_mode == 'cross_attention':
            # Strict GMM semantics require the latent to literally be z = e_k + Delta;
            # a nonlinear combine breaks that reading.
            raise ValueError("residual_continuous=True requires combine_mode='residual_addition' (z = e_k + Delta).")
        self.cont_dropout_p = cont_dropout_p
        self.continuous_mode = continuous_mode
        self.last_drop_fraction = 0.0
        self.downsample_factor = downsample_factor
        self.combine_mode = combine_mode
        self.latent_channels = latent_channels
        # GMM wiring flags -- see models/dual_vae.py for the full rationale.
        self.residual_continuous = residual_continuous
        self.component_prior = component_prior
        trunk_channels = 2 * latent_channels
        self.encoder = DUALVAE_Encoder(downsample_factor=self.downsample_factor, out_channels=trunk_channels)  # (Batch_Size, 2C, Height / 8, Width / 8) -- sized to feed the vanilla branch's mean/logvar head at full rank
        # These are the VQ and the vanilla VAE bottlenecks. Both branches, the VQ
        # lookup, and the decoder now share the same C-channel space (no separate
        # inflated codebook_dim), so the quantization metric matches what the
        # decoder actually consumes.
        self.bottle_neck_VQ = nn.Conv2d(trunk_channels, latent_channels, kernel_size=1, padding=0)
        # Residual wiring reads the C-channel quantization residual; parallel wiring
        # reads the 2C-channel trunk.
        vanilla_in_channels = latent_channels if residual_continuous else trunk_channels
        if self.continuous_mode == 'deterministic':
            self.vanilla_VAE_bottle_neck = nn.Conv2d(vanilla_in_channels, latent_channels, kernel_size=1, padding=0)
        else:
            self.vanilla_VAE_bottle_neck = nn.Conv2d(vanilla_in_channels, 2 * latent_channels, kernel_size=1, padding=0)
        if residual_continuous:
            self._init_identity_residual_head()

        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=latent_channels, commitment_cost=commitment_cost, reduction='mean', l2_normalize=l2_normalize_codes,
                                    use_ema=use_ema_codebook, ema_decay=ema_decay, ema_eps=ema_eps, ema_dead_threshold=ema_dead_threshold,
                                    rq_depth=rq_depth, sigma2_floor=sigma2_floor, sigma2_ceil=sigma2_ceil)

        if self.combine_mode == 'cross_attention':
            self.cross_attention = SpatialCrossAttentionBlock(channels=latent_channels, num_groups=2)
        else:
            self.attention = AttentionBlock(channels=latent_channels, num_groups=2)

        self.decoder = DUALVAE_Decoder(downsample_factor=self.downsample_factor, latent_channels=latent_channels)

    @torch.no_grad()
    def _init_identity_residual_head(self):
        """Start the residual head as Delta ~= r (identity mu weights; for the
        learned-variance mode also logvar bias = -4, std ~0.14). At step 0 the model is
        then near-lossless (e_k + Delta ~= z_e_vq)."""
        w = self.vanilla_VAE_bottle_neck.weight
        b = self.vanilla_VAE_bottle_neck.bias
        w.zero_()
        b.zero_()
        for i in range(self.latent_channels):
            w[i, i, 0, 0] = 1.0
        if self.continuous_mode != 'deterministic':
            b[self.latent_channels:] = -4.0

    def _component_prior_var(self, encoding_indices, batch_size, lh, lw):
        """(B, 1, H, W) per-location prior variance sigma_k^2 from the depth-1 codes.
        Detached: the prior is EMA-estimated, never gradient-trained."""
        sigma2 = self.vq_layer.sigma2[encoding_indices]
        return sigma2.reshape(batch_size, lh, lw).unsqueeze(1).detach()

    def forward_vanilla_z(self, x, noise):

        # (Batch_Size, 2C, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, C, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8.
        # (Batch_Size, C, Height / 8, Width / 8) -> (Batch_Size, C, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, C, Height / 8, Width / 8) -> (Batch_Size, C, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, C, Height / 8, Width / 8) -> (Batch_Size, C, Height / 8, Width / 8)
        stdev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev)
        # (Batch_Size, C, Height / 8, Width / 8) -> (Batch_Size, C, Height / 8, Width / 8)
        x = mean + stdev * noise

        return x, mean, log_variance

    def forward_vanilla_z_deterministic(self, x):
        # Plain deterministic projection: no split, no noise, no reparameterization.
        # The SWD loss alone shapes the aggregate distribution (WAE/SWAE-style).
        return x

    def _combine(self, z_vq, z_vanilla_post):
        if self.combine_mode == 'cross_attention':
            # Q = z_vq (low frequency prior), K=V = z_vanilla_post (high frequency details)
            # The residual connection inside SpatialCrossAttentionBlock ensures: out = z_vq + Attn(z_vq, z_vanilla)
            return self.cross_attention(q=z_vq, kv=z_vanilla_post)
        # Fallback to simple residual addition + self attention
        z_combined = z_vq + z_vanilla_post
        return self.attention(z_combined)

    def forward(self, x, ablation_mode=-1):
        z_e = self.encoder(x) # (Batch_Size, 2C, Height / 8, Width / 8)
        z_e_vq = self.bottle_neck_VQ(z_e) # (Batch_Size, C, Height / 8, Width / 8)
        #z_q, loss, encoding_indices (depth-1), commitment_loss, codebook_loss
        z_vq, vq_loss, encoding_indices, commitment_loss, codebook_loss = self.vq_layer(z_e_vq) # (Batch_Size, C, Height / 8, Width / 8)
        # z_e = torch.Size([4, 8, 32, 32])
        if self.residual_continuous:
            # GMM wiring: continuous branch encodes the quantization residual
            # (z_vq.detach() = raw quantized values; grads still reach z_e_vq via r).
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach()) # (Batch_Size, C or 2C, ...)
        else:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e) # (Batch_Size, C or 2C, Height / 8, Width / 8)
        if self.continuous_mode == 'deterministic':
            z_vanilla_post = self.forward_vanilla_z_deterministic(z_e_vanilla) # (Batch_Size, C, Height / 8, Width / 8)
            mean, log_variance = z_vanilla_post, None
        else:
            # The encoder expects noise with shape (Batch_Size, C, Height/8, Width/8).
            batch_size, _, height, width = x.shape
            noise = torch.randn((batch_size, self.latent_channels, height // self.downsample_factor, width // self.downsample_factor), device=x.device)
            z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise) # (Batch_Size, C, Height / 8, Width / 8)

        # --- Ablation Logic ---
        if ablation_mode == 0:
            # VQ only: Kill Vanilla
            z_vanilla_post = z_vanilla_post * 0
        elif ablation_mode == 1:
            # Vanilla only: Kill VQ
            z_vq = z_vq * 0

        # --- Continuous-branch dropout (combine path only; ablation_mode takes precedence) ---
        # z_vanilla_post itself (and thus the returned log_variance) stays the full, un-dropped
        # posterior -- only the copy fed into the combine step gets masked, so the SWD/variance-
        # budget losses keep seeing the true posterior.
        effective_dropout_p = self.cont_dropout_p if ablation_mode == -1 else 0.0
        z_vanilla_combine, self.last_drop_fraction, keep_mask = apply_cont_dropout(z_vanilla_post, effective_dropout_p, self.training)

        # --- Latent Combination Logic ---
        z_combined = self._combine(z_vq, z_vanilla_combine)
        # resolve any spatial inconsistencies between the VQ tokens and the continuous noise before feeding it into the decoder.
        x_recon = self.decoder(z_combined) 

        vq_related_losses = {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "z_vq": z_vq
        }
        # Per-location prior variance (None unless component_prior). The SWD/variance
        # regularizers whiten Delta by sigma_k before matching against N(0, I).
        prior_var = None
        if self.component_prior:
            prior_var = self._component_prior_var(encoding_indices, x.shape[0],
                                                  z_e_vq.shape[2], z_e_vq.shape[3])

        vanilla_vae_related_loss_terms = {
            "z_vanilla_post": z_vanilla_post,
            "log_variance": log_variance,
            # (B, 1, 1, 1) 0/1 dropout keep-mask (None when no dropout was applied).
            # Lets the trainer restrict the SWD / variance-budget regularizers to the
            # samples whose continuous branch actually reached the decoder.
            "keep_mask": keep_mask,
            # (B, 1, H, W) sigma_k^2 map (None unless component_prior).
            "prior_var": prior_var
        }
        return x_recon, vq_related_losses, vanilla_vae_related_loss_terms
    
    def encode_for_diffusion(self, x, noise=None):
        z_e = self.encoder(x)

        # 1. VQ Branch
        z_e_vq = self.bottle_neck_VQ(z_e)
        z_vq, _, _, _, _ = self.vq_layer(z_e_vq)

        # 2. Vanilla Branch (same wiring as forward())
        if self.residual_continuous:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
        else:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e)
        if self.continuous_mode == 'deterministic':
            z_vanilla_post = self.forward_vanilla_z_deterministic(z_e_vanilla)
        else:
            if noise is None:
                batch_size, _, height, width = x.shape
                noise = torch.randn((batch_size, self.latent_channels, height // 8, width // 8), device=x.device)
            z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise)

        # 3. Combine (uses the same combine_mode-aware path as forward(), so this
        # no longer crashes when combine_mode='cross_attention')
        return self._combine(z_vq, z_vanilla_post)