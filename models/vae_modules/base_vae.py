import torch

from torch import nn
from torch.nn import functional as F
from vae_modules.encoder import VAE_Encoder
from vae_modules.decoder import VAE_Decoder

def compute_vae_loss(original_img, reconstructed_img, mean, logvar, kl_beta=1, mse_reduction='sum'):
    batch_size = reconstructed_img.size(0)

    loss_fn = nn.MSELoss(reduction=mse_reduction)
    reconstruction_loss = loss_fn(reconstructed_img, original_img)
    KL_DIV = -torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    loss = reconstruction_loss + kl_beta * KL_DIV
    return {
        "Total_Loss": loss / batch_size, 
        "Reconstruction_Loss": reconstruction_loss / batch_size, 
        "KL": KL_DIV / batch_size}

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