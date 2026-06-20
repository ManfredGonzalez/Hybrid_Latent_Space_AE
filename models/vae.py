from .modules.encoder import VAE_Encoder
from .modules.decoder import VAE_Decoder

import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
        noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
        latent, mean, logvar = self.encoder(x, noise)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar

    def sample_reconstructions(self, x, n_samples=10):
        self.eval()
        with torch.no_grad():
            batch_size, _, height, width = x.shape
            reconstructions = []

            for _ in range(n_samples):
                noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
                latent, mean, logvar = self.encoder(x, noise)
                recon = self.decoder(latent)
                reconstructions.append(recon)

            # Shape: [n_samples, batch_size, C, H, W]
            reconstructions = torch.stack(reconstructions, dim=0)

            # Mean and std over the n_samples dimension
            recon_mean = reconstructions.mean(dim=0)
            recon_std = reconstructions.std(dim=0)  # This is your uncertainty estimate

        return recon_mean, recon_std