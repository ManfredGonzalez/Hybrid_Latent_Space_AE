from .modules.embedding import VQEmbedding
from .modules.encoder import VQVAE_Encoder
from .modules.decoder import VQVAE_Decoder

import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, latent_channels=128, commitment_cost=0.25, downsample_factor=8, reduction='sum', l2_normalize_codes=False):
        super(VQVAE, self).__init__()
        self.encoder = VQVAE_Encoder(latent_dim=latent_channels, downsample_factor=downsample_factor)
        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=latent_channels, commitment_cost=commitment_cost, reduction=reduction, l2_normalize=l2_normalize_codes)
        self.decoder = VQVAE_Decoder(latent_dim=latent_channels, downsample_factor=downsample_factor)
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _, commitment_loss, codebook_loss = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, commitment_loss, codebook_loss