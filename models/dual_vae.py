from .modules.embedding import VQEmbedding
from .modules.fsq import FSQEmbedding
from .modules.encoder import DUALVAE_Encoder
from .modules.decoder import DUALVAE_Decoder
from .modules.attention import AttentionBlock
from .modules.cont_dropout import validate_cont_dropout_p, apply_cont_dropout
from .modules.wavelet import HaarDWT, NUM_BANDS
from .modules.detail_encoder import WaveletDetailEncoder

import torch.nn as nn
import torch

class DUALVAE(nn.Module):
    def __init__(self, num_embeddings=512, latent_channels=8, commitment_cost=0.25, downsample_factor=8, l2_normalize_codes=False, cont_dropout_p=0.0,
                 use_ema_codebook=False, ema_decay=0.99, ema_eps=1e-5, ema_dead_threshold=1.0,
                 rq_depth=1, residual_continuous=False, component_prior=False, sigma2_floor=1e-3, sigma2_ceil=10.0,
                 wavelet_detail=False, wavelet_band_channels=None,
                 hierarchical_semantic=False, coarse_factor=4, coarse_num_embeddings=64,
                 quantizer="vq", fsq_levels=None):
        super(DUALVAE, self).__init__()
        validate_cont_dropout_p(cont_dropout_p)
        # quantizer: "vq" (learned codebook, EMA or gradient) or "fsq" (fixed scalar grid,
        # Mentzer et al. 2023). FSQ maintains the SAME per-code EMA statistics -- it just
        # never moves the codes -- so the code-centered prior works under both.
        quantizer = (quantizer or "vq").lower()
        if quantizer not in ("vq", "fsq"):
            raise ValueError(f"quantizer must be 'vq' or 'fsq', got {quantizer!r}.")
        self.quantizer = quantizer
        if component_prior and quantizer == "vq" and not use_ema_codebook:
            # FSQ is exempt: its sigma_k^2/mu_k come from its own EMA statistics, which exist
            # unconditionally -- there is no gradient-trained-codebook variant to guard against.
            raise ValueError("component_prior=True needs the EMA statistics (sigma_k^2 = v_k/N_k); set use_ema_codebook=True.")
        self.cont_dropout_p = cont_dropout_p
        self.last_drop_fraction = 0.0
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels
        # --- Wavelet detail branch (single-level Haar; default OFF) ---
        # When True, the CONTINUOUS branch's INPUT changes from the (post-compression)
        # quantization residual to the PRE-compression high-frequency Haar subbands
        # (LH/HL/HH), so Delta finally carries fine detail the strided encoder would have
        # averaged away -- while the regularizer stays the SAME code-centered KL
        # (N(0, sigma_k^2) via component_prior). It still predicts (mean, logvar) and
        # reparameterizes, so the loss/KL machinery is unchanged. residual_continuous is
        # ignored for the input when this is on (the wavelet detail replaces it).
        self.wavelet_detail = wavelet_detail
        # --- GMM wiring flags (all default-off, for ablations) ---
        # residual_continuous: the continuous branch encodes the quantization residual
        #   r = z_e_vq - z_q (detached codes) instead of reading the trunk in parallel.
        #   The latent is then literally z = e_k + Delta: code = component mean,
        #   continuous = within-component offset. Branch roles are set by construction,
        #   so the two branches can no longer compete for the same information.
        # component_prior: the KL prior on Delta becomes N(0, sigma_k^2) with the
        #   per-code EMA variance (see VQEmbedding.sigma2) instead of N(0, I).
        self.residual_continuous = residual_continuous
        self.component_prior = component_prior
        trunk_channels = 2 * latent_channels
        self.encoder = DUALVAE_Encoder(downsample_factor=self.downsample_factor, out_channels=trunk_channels)  # (Batch_Size, 2C, Height / 8, Width / 8) -- sized to feed the vanilla branch's mean/logvar head at full rank
        # These are the VQ and the vanilla VAE bottlenecks. Both branches, the VQ
        # lookup, and the decoder now share the same C-channel space (no separate
        # inflated codebook_dim), so the quantization metric matches what the
        # decoder actually consumes.
        self.bottle_neck_VQ = nn.Conv2d(trunk_channels, latent_channels, kernel_size=1, padding=0)
        if self.wavelet_detail:
            # Detail branch = Haar DWT front-end + per-band encoder -> C-channel detail
            # features, then a mean/logvar head so the KL/reparam path is unchanged.
            if wavelet_band_channels is None:
                wavelet_band_channels = self._default_band_channels(latent_channels)
            if sum(wavelet_band_channels) != latent_channels:
                raise ValueError(f"wavelet_band_channels {wavelet_band_channels} must sum to latent_channels ({latent_channels}).")
            self.wavelet_band_channels = list(wavelet_band_channels)
            self.dwt = HaarDWT(in_channels=3)
            detail_down = max(1, downsample_factor // 2)          # DWT halves; detail enc does the rest
            self.detail_encoder = WaveletDetailEncoder(in_per_band=3, band_channels=self.wavelet_band_channels, down=detail_down)
            self.wavelet_meanvar = nn.Conv2d(latent_channels, 2 * latent_channels, kernel_size=1, padding=0)
        else:
            # Residual wiring: the continuous head reads the C-channel quantization residual;
            # parallel (original) wiring: it reads the 2C-channel trunk.
            vanilla_in_channels = latent_channels if residual_continuous else trunk_channels
            self.vanilla_VAE_bottle_neck = nn.Conv2d(vanilla_in_channels, 2 * latent_channels, kernel_size=1, padding=0)
            if residual_continuous:
                self._init_identity_residual_head()

        if quantizer == "fsq":
            # d == latent_channels by construction (FSQEmbedding enforces it), so there are
            # no projections: the code, the residual, Delta and the decoder input all live
            # in one space and the GMM statistics describe exactly what Delta models.
            self.vq_layer = FSQEmbedding(levels=fsq_levels, embedding_dim=latent_channels,
                                         ema_decay=ema_decay, ema_eps=ema_eps,
                                         sigma2_floor=sigma2_floor, sigma2_ceil=sigma2_ceil,
                                         rq_depth=rq_depth)
        else:
            self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=latent_channels, commitment_cost=commitment_cost, l2_normalize=l2_normalize_codes,
                                        use_ema=use_ema_codebook, ema_decay=ema_decay, ema_eps=ema_eps, ema_dead_threshold=ema_dead_threshold,
                                        rq_depth=rq_depth, sigma2_floor=sigma2_floor, sigma2_ceil=sigma2_ceil)

        # --- Hierarchical SEMANTIC level (coarse code-centered GMM; default OFF) ---
        # Adds a COARSE quantization level with a large receptive field, so its codes
        # summarize object-scale regions (concepts) instead of 8x8 patches (textures). The
        # coarse level carries the code-centered GMM at the SEMANTIC scale (coarse code =
        # component mean, small coarse Delta = within-concept offset); the existing FINE VQ +
        # continuous branch become the DETAIL/texture path, so pixel fidelity is preserved.
        # The fine level encodes the RESIDUAL after the (broadcast) coarse layout, which forces
        # the coarse codes to carry the dominant structure (VQVAE-2 / RQ principle across scales).
        self.hierarchical_semantic = hierarchical_semantic
        if hierarchical_semantic:
            import math as _math
            n_steps = int(round(_math.log2(coarse_factor)))
            if 2 ** n_steps != coarse_factor or n_steps < 1:
                raise ValueError(f"coarse_factor must be a power of 2 >= 2, got {coarse_factor}.")
            self.coarse_factor = coarse_factor
            C = latent_channels
            downs = []
            for i in range(n_steps):
                downs.append(nn.Conv2d(C, C, 3, stride=2, padding=1))
                if i != n_steps - 1:
                    downs += [nn.GroupNorm(2, C), nn.SiLU()]
            self.coarse_down = nn.Sequential(*downs)      # z_e_vq (32x32) -> coarse (32/cf)
            ups = []
            for i in range(n_steps):
                ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
                ups.append(nn.Conv2d(C, C, 3, stride=1, padding=1))
                if i != n_steps - 1:
                    ups += [nn.GroupNorm(2, C), nn.SiLU()]
            self.coarse_up = nn.Sequential(*ups)          # coarse -> 32x32 (broadcast layout)
            # coarse codebook = the SEMANTIC GMM (fewer codes -> more concept-like). Uses the
            # same EMA/sigma2 machinery so it has pi_k, sigma_k^2, and the code-centered prior.
            self.vq_layer_coarse = VQEmbedding(
                num_embeddings=coarse_num_embeddings, embedding_dim=C, commitment_cost=commitment_cost,
                l2_normalize=l2_normalize_codes, use_ema=use_ema_codebook, ema_decay=ema_decay,
                ema_eps=ema_eps, ema_dead_threshold=ema_dead_threshold, rq_depth=1,
                sigma2_floor=sigma2_floor, sigma2_ceil=sigma2_ceil)
            # coarse continuous offset (code-centered GMM at the coarse scale)
            self.coarse_vanilla_bottleneck = nn.Conv2d(C, 2 * C, 1)
            # Zero-init the last coarse_up conv so coarse_up = 0 at step 0: the fine level then
            # sees the untouched z_e_vq (matches the k-means codebook init) and the coarse
            # contribution ramps in as reconstruction gradient trains coarse_up -- a clean warm start.
            with torch.no_grad():
                last_conv = [m for m in self.coarse_up if isinstance(m, nn.Conv2d)][-1]
                last_conv.weight.zero_(); last_conv.bias.zero_()

        self.attention = AttentionBlock(channels=latent_channels, num_groups=2)

        self.decoder = DUALVAE_Decoder(downsample_factor=self.downsample_factor, latent_channels=latent_channels)

    @staticmethod
    def _default_band_channels(latent_channels):
        """Split latent_channels into NUM_BANDS groups, remainder to earliest bands
        (C=8, 3 bands -> [3, 3, 2])."""
        base = latent_channels // NUM_BANDS
        rem = latent_channels % NUM_BANDS
        return [base + (1 if i < rem else 0) for i in range(NUM_BANDS)]

    @torch.no_grad()
    def _init_identity_residual_head(self):
        """Start the residual head as Delta ~= r: mu weights = identity, logvar bias = -4
        (std ~0.14). At step 0 the model is then near-lossless (e_k + Delta ~= z_e_vq),
        so training starts from 'refine the codes' instead of 'rebuild the latent'."""
        w = self.vanilla_VAE_bottle_neck.weight  # (2C, C, 1, 1)
        b = self.vanilla_VAE_bottle_neck.bias
        w.zero_()
        b.zero_()
        for i in range(self.latent_channels):
            w[i, i, 0, 0] = 1.0
        b[self.latent_channels:] = -4.0

    def _gather_per_location(self, table, encoding_indices, batch_size, lh, lw):
        """(K,) or (K, C) statistic -> (B, 1, H, W) or (B, C, H, W), gathered from the
        depth-1 code assignments. Detached: the prior is EMA-estimated, never trained.

        Two shapes because the two quantizers hold different statistics. VQ tracks one
        isotropic sigma_k^2 per code, which broadcasts over channels; FSQ tracks a diagonal
        (K, C) covariance and a (K, C) mean offset, because a within-cell displacement is
        directional and collapsing it to a scalar would discard most of it.
        """
        if table is None:
            return None
        v = table[encoding_indices]
        if v.dim() == 1:
            return v.reshape(batch_size, lh, lw).unsqueeze(1).detach()
        return v.reshape(batch_size, lh, lw, -1).permute(0, 3, 1, 2).contiguous().detach()

    def _component_prior(self, encoding_indices, batch_size, lh, lw):
        """(prior_var, prior_mean) for the KL. prior_mean is None unless the quantizer
        exposes mu_k -- an EMA codebook's residual is already zero-mean, so only FSQ needs it."""
        var = self._gather_per_location(self.vq_layer.sigma2, encoding_indices, batch_size, lh, lw)
        mean = self._gather_per_location(getattr(self.vq_layer, 'mu', None),
                                         encoding_indices, batch_size, lh, lw)
        return var, mean

    def _component_prior_var(self, encoding_indices, batch_size, lh, lw):
        """Back-compat wrapper: variance only."""
        return self._component_prior(encoding_indices, batch_size, lh, lw)[0]

    def encode_zevq(self, x):
        """Encoder + VQ bottleneck only -> z_e_vq (B, C, h, w). Cheap path for the
        augmented view in the cross-view code-invariance aux loss (no decode)."""
        return self.bottle_neck_VQ(self.encoder(x))

    def code_soft_assign(self, z_e_vq, tau=0.5):
        """(B, C, h, w) -> (B, h*w, K) PER-LOCATION soft assignment over the depth-1 codebook
        (softmax over -||z_e_vq - e_k||^2 / tau). This KEEPS the spatial structure that
        code_histogram averages away -- required by the spatially-aligned CVI loss, where
        location i of the clean view must be matched against location i of a photometrically
        augmented view (no crop/flip, so the grids line up)."""
        import torch.nn.functional as F
        b, c, h, w = z_e_vq.shape
        zf = z_e_vq.permute(0, 2, 3, 1).reshape(b * h * w, c)
        cb = self.vq_layer.codebook                                # (K, C)
        if self.vq_layer.l2_normalize:                            # cosine (matches VQ lookup)
            logits = (F.normalize(zf, dim=-1) @ F.normalize(cb, dim=-1).t()) / tau
        else:
            d2 = (zf ** 2).sum(-1, keepdim=True) + (cb ** 2).sum(-1) - 2 * zf @ cb.t()
            logits = -d2 / tau
        return F.softmax(logits, dim=-1).reshape(b, h * w, -1)    # (B, HW, K)

    def code_histogram(self, z_e_vq, tau=0.5):
        """(B, C, h, w) -> (B, K) image-level soft bag-of-codes over the depth-1 codebook
        (per-location soft assignment averaged over spatial locations). Position-invariant
        (crops don't misalign it), but the pooling washes out per-patch identity -- see
        code_soft_assign for the spatially-resolved version used by the aligned CVI loss."""
        return self.code_soft_assign(z_e_vq, tau).mean(dim=1)   # (B, K)

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
    def forward(self, x, ablation_mode=-1):
        # The encoder expects noise with shape (Batch_Size, C, Height/8, Width/8).
        batch_size, _, height, width = x.shape
        noise = torch.randn((batch_size, self.latent_channels, height // self.downsample_factor, width // self.downsample_factor), device=x.device)
        z_e = self.encoder(x) # (Batch_Size, 2C, Height / 8, Width / 8)
        z_e_vq = self.bottle_neck_VQ(z_e) # (Batch_Size, C, Height / 8, Width / 8)

        # --- Coarse SEMANTIC level (hierarchical) ---
        # Quantize a downsampled z_e_vq (large receptive field -> concepts) into the coarse
        # code-centered GMM, broadcast it back, and let the fine level below encode only the
        # RESIDUAL (so the coarse must carry the dominant, object-scale structure).
        coarse_up = None
        coarse_extra = {}
        if self.hierarchical_semantic:
            coarse_feat = self.coarse_down(z_e_vq)                                # (B, C, gc, gc)
            cz_vq, c_vq_loss, c_idx, c_commit, c_codebook = self.vq_layer_coarse(coarse_feat)
            c_r = coarse_feat - cz_vq.detach()                                    # within-concept residual
            c_meanvar = self.coarse_vanilla_bottleneck(c_r)
            gc = coarse_feat.shape[2]
            c_noise = torch.randn((batch_size, self.latent_channels, gc, gc), device=x.device)
            c_zcont, c_mean, c_logvar = self.forward_vanilla_z(c_meanvar, c_noise)
            coarse_latent = cz_vq + c_zcont                                       # e_k + Delta (coarse GMM)
            coarse_up = self.coarse_up(coarse_latent)                             # (B, C, H/8, W/8)
            c_prior_var = None
            if self.component_prior:
                c_prior_var = self.vq_layer_coarse.sigma2[c_idx].reshape(batch_size, gc, gc).unsqueeze(1).detach()
            z_e_vq = z_e_vq - coarse_up.detach()   # fine level now encodes the residual after coarse
            coarse_extra = {
                "coarse_commit": c_commit, "coarse_codebook_loss": c_codebook,
                "coarse_perplexity": self.vq_layer_coarse.perplexity,
                "coarse_z_vq": cz_vq, "coarse_idx": c_idx,
                "coarse_mean": c_mean, "coarse_logvar": c_logvar, "coarse_prior_var": c_prior_var,
            }

        #z_q, loss, encoding_indices (depth-1), commitment_loss, codebook_loss
        z_vq, vq_loss, encoding_indices, commitment_loss, codebook_loss = self.vq_layer(z_e_vq) # (Batch_Size, C, Height / 8, Width / 8)
        # z_e = torch.Size([4, 8, 32, 32])
        if self.wavelet_detail:
            # Detail branch: encode the PRE-compression high-frequency Haar subbands
            # (LH/HL/HH) into a C-channel feature, then a mean/logvar head -> the same
            # reparameterized Delta the KL/component_prior regularizes. z = e_k + Delta,
            # but Delta now carries fine detail the strided encoder averaged away.
            _, hf = self.dwt(x)                                   # (B, 9, H/2, W/2), band-major
            delta_feat = self.detail_encoder(hf)                  # (B, C, H/8, W/8)
            z_e_vanilla = self.wavelet_meanvar(delta_feat)        # (B, 2C, H/8, W/8)
        elif self.residual_continuous:
            # GMM wiring: the continuous branch encodes the quantization residual.
            # z_vq is the straight-through output, so z_vq.detach() is the raw quantized
            # value; the detach stops the continuous branch from pushing the codes
            # around while gradients still reach z_e_vq (and the trunk) through r.
            # pre_quant() is the identity for VQ (unchanged behaviour). For FSQ it applies
            # the tanh bound, so both terms live in the grid's space -- subtracting a bounded
            # quantized value from the raw encoder output would make r track the encoder's
            # magnitude instead of the within-cell offset the prior is defined on.
            r = self.vq_layer.pre_quant(z_e_vq) - z_vq.detach()
            z_e_vanilla = self.vanilla_VAE_bottle_neck(r) # (Batch_Size, 2C, Height / 8, Width / 8)
        else:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e) # (Batch_Size, 2C, Height / 8, Width / 8)
        z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise) # (Batch_Size, C, Height / 8, Width / 8)

        # Per-location prior variance (and, under FSQ, mean) for the KL.
        prior_var = None
        prior_mean = None
        if self.component_prior:
            prior_var, prior_mean = self._component_prior(encoding_indices, batch_size,
                                                          z_e_vq.shape[2], z_e_vq.shape[3])

        # --- Ablation Logic ---
        if ablation_mode == 0:
            # VQ only: Kill Vanilla
            z_vanilla_post = z_vanilla_post * 0
        elif ablation_mode == 1:
            # Vanilla only: Kill VQ
            z_vq = z_vq * 0

        # --- Continuous-branch dropout (combine path only; ablation_mode takes precedence) ---
        # z_vanilla_post itself (and thus `mean`/`log_variance` below) stays the full, un-dropped
        # posterior -- only the copy fed into the combine step gets masked, so the KL loss keeps
        # seeing the true posterior.
        effective_dropout_p = self.cont_dropout_p if ablation_mode == -1 else 0.0
        z_vanilla_combine, self.last_drop_fraction, keep_mask = apply_cont_dropout(z_vanilla_post, effective_dropout_p, self.training)

        # simple residual addition
        z_combined = z_vq + z_vanilla_combine
        # Add the broadcast coarse SEMANTIC layout: decoder input = coarse structure + fine
        # texture-code + within-code detail. (coarse_up is 0 at init -> clean warm start.)
        if coarse_up is not None:
            z_combined = z_combined + coarse_up
        #It forces the network to treat the continuous branch as a residual detail layer.
        #attention block
        z_combined = self.attention(z_combined)
        # resolve any spatial inconsistencies between the VQ tokens and the continuous noise before feeding it into the decoder.
        x_recon = self.decoder(z_combined) 

        vq_related_losses = {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "z_vq": z_vq,
            # Pre-quantization encoder bottleneck (what the codebook quantizes). Exposed so
            # the cross-view code-invariance aux loss can reuse the clean-view assignment
            # for free (no re-encode). Extra key -> backward-compatible.
            "z_e_vq": z_e_vq,
        }
        vq_related_losses.update(coarse_extra)   # coarse_* keys only present when hierarchical_semantic
        vanilla_vae_related_losses = {
            "mean": mean,
            "log_variance": log_variance,
            "z_vanilla_post": z_vanilla_post,
            # (B, 1, 1, 1) 0/1 dropout keep-mask (None when no dropout was applied).
            # Lets the trainer mask the KL per-sample so dropped samples -- whose
            # continuous branch got no reconstruction gradient -- also get no KL
            # shrinkage: penalty and reward stay balanced under cont_dropout.
            "keep_mask": keep_mask,
            # (B, 1, H, W) per-location prior variance sigma_k^2 for the KL
            # (None unless component_prior; broadcasts over the C channels). Under FSQ this
            # is (B, C, H, W) instead -- a diagonal covariance, one value per channel.
            "prior_var": prior_var,
            # (B, C, H, W) per-location prior MEAN mu_k, or None. Only FSQ produces it: an
            # EMA codebook's residual is zero-mean by construction, a fixed grid's is not.
            "prior_mean": prior_mean
        }
        return x_recon, vq_related_losses, vanilla_vae_related_losses
    
    def encode_for_diffusion(self, x, noise=None):
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn((batch_size, self.latent_channels, height // 8, width // 8), device=x.device)

        z_e = self.encoder(x)

        # 1. VQ Branch
        z_e_vq = self.bottle_neck_VQ(z_e)
        z_vq, _, _, _, _ = self.vq_layer(z_e_vq)

        # 2. Vanilla Branch (same wiring as forward())
        if self.wavelet_detail:
            _, hf = self.dwt(x)
            z_e_vanilla = self.wavelet_meanvar(self.detail_encoder(hf))
        elif self.residual_continuous:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e_vq - z_vq.detach())
        else:
            z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e)
        z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise)

        # 3. Combine and Attention
        z_combined = z_vq + z_vanilla_post
        z_combined = self.attention(z_combined)
        
        return z_combined