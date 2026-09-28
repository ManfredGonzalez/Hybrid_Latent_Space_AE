"""Frozen representation encoder (DINOv2) serving BOTH jobs REPA-E + MM-FM need.

One forward pass per batch produces:
  * patch tokens -- the REPA alignment target (Yu et al. 2024; Leng et al. 2025), matched
    one-to-one with the SiT's latent tokens;
  * the [CLS] token -- the global descriptor the image-level mode tracker quantizes.

The token grids line up by construction and this is worth checking before changing anything:
DINOv2-B/14 at 224px produces 16x16 = 256 patch tokens, and our 32x32 latent at SiT patch
size 2 produces 16x16 = 256 tokens. That correspondence is what makes the patch-wise cosine
loss meaningful; it is the same f8 geometry REPA-E uses with SD-VAE and SiT-XL/2.

PREPROCESSING DIFFERS FROM THE REFERENCE, deliberately. Their loader yields images in [0, 255]
and they divide by 255; ours yields [-1, 1] (transforms.Normalize((0.5,), (0.5,)) in
data/datasets.py), so we map [-1, 1] -> [0, 1] instead. Getting this wrong does not crash --
it quietly feeds the encoder off-distribution images and the alignment signal degrades.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV2Features(nn.Module):
    """Frozen DINOv2 wrapper. Always eval, always no_grad, never in an optimizer.

    Args:
        name: torch.hub entry point, e.g. 'dinov2_vitb14' (768-d) or 'dinov2_vitl14' (1024-d).
        image_size: side length fed to the encoder; must be a multiple of the patch size (14).
    """

    def __init__(self, name="dinov2_vitb14", image_size=224):
        super().__init__()
        if image_size % 14 != 0:
            raise ValueError(f"DINOv2 uses 14px patches; image_size {image_size} is not a multiple of 14.")
        # The weights must already be in the torch.hub cache: compute nodes usually have no
        # internet. Pre-download once on a login node with
        #   python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')"
        self.model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_size = image_size
        self.grid_size = image_size // 14
        self.embed_dim = int(self.model.embed_dim)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        # Ignore the trainer's model.train(): this encoder is frozen by contract. Letting it
        # enter train mode would change nothing today (ViT has no BN) but would silently start
        # mattering if the backbone were ever swapped for one that does.
        return super().train(False)

    def preprocess(self, images):
        """(B, 3, H, W) in [-1, 1] -> normalized (B, 3, image_size, image_size)."""
        x = (images.float() + 1.0) / 2.0
        x = (x - self._mean) / self._std
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return x

    @torch.no_grad()
    def forward(self, images):
        """Returns (patch_tokens (B, grid^2, C), cls (B, C)), both detached."""
        feats = self.model.forward_features(self.preprocess(images))
        return feats["x_norm_patchtokens"].detach(), feats["x_norm_clstoken"].detach()
