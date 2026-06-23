from .modules.encoder import VAE_Encoder
from .modules.decoder import VAE_Decoder

import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, downsample_factor=8):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.encoder = VAE_Encoder(downsample_factor=self.downsample_factor)
        self.decoder = VAE_Decoder(downsample_factor=self.downsample_factor)
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
        noise = torch.randn((batch_size, 4, height // self.downsample_factor, width // self.downsample_factor), device=x.device)
        latent, mean, logvar = self.encoder(x, noise)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar

    def sample_reconstructions(self, x, n_samples=10):
        self.eval()
        with torch.no_grad():
            batch_size, _, height, width = x.shape
            reconstructions = []

            for _ in range(n_samples):
                noise = torch.randn((batch_size, 4, height // self.downsample_factor, width // self.downsample_factor), device=x.device)
                latent, mean, logvar = self.encoder(x, noise)
                recon = self.decoder(latent)
                reconstructions.append(recon)

            # Shape: [n_samples, batch_size, C, H, W]
            reconstructions = torch.stack(reconstructions, dim=0)

            # Mean and std over the n_samples dimension
            recon_mean = reconstructions.mean(dim=0)
            recon_std = reconstructions.std(dim=0)  # This is your uncertainty estimate

        return recon_mean, recon_std