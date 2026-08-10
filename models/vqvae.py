from .modules.embedding import VQEmbedding
from .modules.fsq import FSQEmbedding
from .modules.encoder import VQVAE_Encoder
from .modules.decoder import VQVAE_Decoder

import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, latent_channels=128, commitment_cost=0.25, downsample_factor=8, reduction='sum', l2_normalize_codes=False,
                 use_ema_codebook=False, ema_decay=0.99, ema_eps=1e-5, ema_dead_threshold=1.0,
                 rq_depth=1, quantizer="vq", fsq_levels=None, sigma2_floor=1e-4, sigma2_ceil=0.0278):
        super(VQVAE, self).__init__()
        # quantizer: "vq" (learned codebook, EMA or gradient) or "fsq" (fixed scalar grid,
        # Mentzer et al. 2023). FSQEmbedding is a drop-in for VQEmbedding -- same forward
        # signature, same 5-tuple, same diagnostic attribute names -- so everything below
        # and every consumer of this model is unchanged by the swap. The default is "vq",
        # which reproduces the previous constructor exactly.
        if quantizer not in ("vq", "fsq"):
            raise ValueError(f"quantizer must be 'vq' or 'fsq', got {quantizer!r}.")
        self.quantizer = quantizer
        self.encoder = VQVAE_Encoder(latent_dim=latent_channels, downsample_factor=downsample_factor)
        if quantizer == "fsq":
            # d == latent_channels by construction (FSQEmbedding enforces it and raises a
            # readable error otherwise), so the encoder head and decoder input still agree.
            # num_embeddings / commitment_cost / l2_normalize / use_ema / dead-threshold have
            # no meaning on a fixed grid and are deliberately not forwarded.
            self.vq_layer = FSQEmbedding(levels=fsq_levels, embedding_dim=latent_channels,
                                         ema_decay=ema_decay, ema_eps=ema_eps,
                                         sigma2_floor=sigma2_floor, sigma2_ceil=sigma2_ceil,
                                         rq_depth=rq_depth, reduction=reduction)
        else:
            self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=latent_channels, commitment_cost=commitment_cost, reduction=reduction, l2_normalize=l2_normalize_codes,
                                        use_ema=use_ema_codebook, ema_decay=ema_decay, ema_eps=ema_eps, ema_dead_threshold=ema_dead_threshold,
                                        rq_depth=rq_depth)
        self.decoder = VQVAE_Decoder(latent_dim=latent_channels, downsample_factor=downsample_factor)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _, commitment_loss, codebook_loss = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, commitment_loss, codebook_loss
