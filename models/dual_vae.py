from .modules.embedding import VQEmbedding
from .modules.encoder import DUALVAE_Encoder
from .modules.decoder import DUALVAE_Decoder
from .modules.attention import AttentionBlock

import torch.nn as nn
import torch

class DUALVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super(DUALVAE, self).__init__()
        self.encoder = DUALVAE_Encoder()  # This encodes the image and gives me an initial bottleneck of (Batch_Size, 8, Height / 8, Width / 8)
        # These are the VQ and the vanilla VAE bottlenecks
        self.bottle_neck_VQ = nn.Conv2d(8, embedding_dim, kernel_size=1, padding=0)
        self.vanilla_VAE_bottle_neck = nn.Conv2d(8, 8, kernel_size=1, padding=0)

        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        self.vq_post_bottleneck = nn.Conv2d(embedding_dim, 4, kernel_size=1, padding=0) ### This is for VQ branch 
        self.vanilla_VAE_post_bottleneck = nn.Conv2d(8, 4, kernel_size=1, padding=0) ### This is for the vanilla VAE branch

        self.attention = AttentionBlock(channels=4, num_groups=2)

        self.decoder = DUALVAE_Decoder()
    def forward_vanilla_z(self, x, noise):
        
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x, mean, log_variance
    def forward(self, x, ablation_mode=-1):
        # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
        batch_size, _, height, width = x.shape
        noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
        z_e = self.encoder(x) # (Batch_Size, 8, Height / 8, Width / 8)
        z_e_vq = self.bottle_neck_VQ(z_e) # (Batch_Size, embedding_dim, Height / 8, Width / 8)
        #z_q, loss, encoding_indices, commitment_loss, codebook_loss
        z_vq, vq_loss, _, commitment_loss, codebook_loss = self.vq_layer(z_e_vq)
        z_vq = self.vq_post_bottleneck(z_vq) # (Batch_Size, 4, Height / 8, Width / 8)
        # z_e = torch.Size([4, 8, 32, 32])
        z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e) # (Batch_Size, 8, Height / 8, Width / 8)
        z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise) # (Batch_Size, 4, Height / 8, Width / 8)
        z_vanilla_post/= 0.18215 #only for the vanilla VAE branch

        # --- Ablation Logic ---
        if ablation_mode == 0:
            # VQ only: Kill Vanilla
            z_vanilla_post = z_vanilla_post * 0
        elif ablation_mode == 1:
            # Vanilla only: Kill VQ
            z_vq = z_vq * 0

        # simple residual addition
        z_combined = z_vq + z_vanilla_post
        #It forces the network to treat the continuous branch as a residual detail layer.
        #attention block
        z_combined = self.attention(z_combined)
        # resolve any spatial inconsistencies between the VQ tokens and the continuous noise before feeding it into the decoder.
        x_recon = self.decoder(z_combined) 

        vq_related_losses = {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss
        }
        vanilla_vae_related_losses = {
            "mean": mean,
            "log_variance": log_variance
        }
        return x_recon, vq_related_losses, vanilla_vae_related_losses
    
    def encode_for_diffusion(self, x, noise=None):
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
            
        z_e = self.encoder(x) 
        
        # 1. VQ Branch
        z_e_vq = self.bottle_neck_VQ(z_e) 
        z_vq, _, _, _, _ = self.vq_layer(z_e_vq)
        z_vq = self.vq_post_bottleneck(z_vq) 
        
        # 2. Vanilla Branch
        z_e_vanilla = self.vanilla_VAE_bottle_neck(z_e) 
        z_vanilla_post, mean, log_variance = self.forward_vanilla_z(z_e_vanilla, noise) 
        z_vanilla_post /= 0.18215 

        # 3. Combine and Attention
        z_combined = z_vq + z_vanilla_post
        z_combined = self.attention(z_combined)
        
        return z_combined