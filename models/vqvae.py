from .modules.embedding import VQEmbedding
from .modules.encoder import VQVAE_Encoder
from .modules.decoder import VQVAE_Decoder

import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = VQVAE_Encoder(latent_dim=embedding_dim)
        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = VQVAE_Decoder(latent_dim=embedding_dim)
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _, commitment_loss, codebook_loss = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, commitment_loss, codebook_loss