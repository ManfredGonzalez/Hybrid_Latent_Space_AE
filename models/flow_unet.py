"""Class-conditional UNet velocity field v_theta(x_t, t, y) for latent flow matching.

ADM-style (Dhariwal & Nichol 2021) UNet, sized for the (C, H/8, W/8) latents this repo's
autoencoders produce -- 8 x 32 x 32 for a 256px image at downsample_factor 8. It is the
GENERATOR being compared across latent spaces, so it is deliberately independent of which
autoencoder produced the latents: it only ever sees `in_channels`. Train it on DUALVAE
latents and on vanilla-VAE latents with the identical config and the difference in the
resulting sample quality is attributable to the latent space, not to the generator.

Conditioning is FiLM-style: the timestep embedding and the class embedding are summed into
one vector that scale/shifts every residual block's normalization. The class embedding table
has num_classes + 1 rows; the last row is the NULL class used for classifier-free guidance
(Ho & Salimans 2022) -- training drops the label to it with probability cfg_dropout_prob and
sampling extrapolates between the two.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a (B,) tensor of continuous flow times t in [0, 1].

    t is scaled by 1000 first so the usable frequency band matches the diffusion
    literature's discrete 0..1000 step index (a raw t in [0,1] would leave most of the
    sinusoidal basis unused).
    """
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def zero_module(module):
    """Zero out a module's parameters (ADM's trick for residual output projections: each
    block starts as the identity, so a deep UNet is stable at step 0)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _groups(channels, max_groups=32):
    """GroupNorm group count that always divides `channels`."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Residual block with FiLM (scale, shift) conditioning from the time+class embedding."""

    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.0):
        super().__init__()
        self.in_norm = nn.GroupNorm(_groups(in_channels), in_channels)
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # 2 * out_channels -> (scale, shift)
        self.emb_proj = nn.Linear(emb_channels, 2 * out_channels)
        self.out_norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_conv(self.dropout(F.silu(h)))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """Multi-head self-attention over the spatial grid (used at the low-resolution levels,
    where the token count is small enough for global mixing to be affordable)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        # (B, heads, HW, head_dim) for scaled_dot_product_attention
        q, k, v = (t.transpose(-2, -1).contiguous() for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LatentFlowUNet(nn.Module):
    """v_theta(x_t, t, y) -> velocity, same shape as x_t.

    Args:
        in_channels: latent width C (8 for both models compared here).
        latent_size: spatial side of the latent grid (32 for 256px / downsample 8). Only used
            to decide at which levels attention is inserted, via attention_resolutions.
        base_channels: width of the first level; each level i is base_channels * channel_mults[i].
        channel_mults: one entry per resolution level (level i runs at latent_size / 2**i).
        num_res_blocks: residual blocks per level (per side of the UNet).
        attention_resolutions: spatial sizes at which to insert self-attention.
        num_classes: number of real classes; the embedding table gets one extra NULL row.
    """

    def __init__(
        self,
        in_channels=8,
        latent_size=32,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        num_classes=10,
        dropout=0.0,
        num_heads=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        # Index num_classes == the NULL class (classifier-free guidance).
        self.null_class = num_classes

        emb_channels = base_channels * 4
        self.emb_channels = emb_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )
        self.time_embed_dim = base_channels
        self.class_emb = nn.Embedding(num_classes + 1, emb_channels)
        nn.init.normal_(self.class_emb.weight, std=0.02)

        attention_resolutions = set(attention_resolutions)

        # ---- down path ----
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        res = latent_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, emb_channels, dropout)])
                ch = out_ch
                if res in attention_resolutions:
                    block.append(SelfAttention(ch, num_heads))
                self.down_blocks.append(block)
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                res //= 2

        # ---- middle ----
        self.mid_block1 = ResBlock(ch, ch, emb_channels, dropout)
        self.mid_attn = SelfAttention(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, emb_channels, dropout)

        # ---- up path ----
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [ResBlock(ch + skip_channels.pop(), out_ch, emb_channels, dropout)]
                )
                ch = out_ch
                if res in attention_resolutions:
                    block.append(SelfAttention(ch, num_heads))
                if level != 0 and i == num_res_blocks:
                    block.append(Upsample(ch))
                    res *= 2
                self.up_blocks.append(block)

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_conv = zero_module(nn.Conv2d(ch, in_channels, 3, padding=1))

    def forward(self, x, t, y):
        """x: (B, C, H, W) noisy/interpolated latent. t: (B,) in [0, 1]. y: (B,) int64 class
        ids, where `num_classes` means the NULL (unconditional) class."""
        emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim)) + self.class_emb(y)

        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, emb)), emb)

        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_flow_model(args, in_channels, latent_size, num_classes):
    """Instantiate the UNet from a config Namespace (keys all optional -> defaults above)."""
    return LatentFlowUNet(
        in_channels=in_channels,
        latent_size=latent_size,
        base_channels=getattr(args, "unet_base_channels", 128),
        channel_mults=tuple(getattr(args, "unet_channel_mults", (1, 2, 2, 2))),
        num_res_blocks=getattr(args, "unet_num_res_blocks", 2),
        attention_resolutions=tuple(getattr(args, "unet_attention_resolutions", (16, 8))),
        num_classes=num_classes,
        dropout=getattr(args, "unet_dropout", 0.0),
        num_heads=getattr(args, "unet_num_heads", 4),
    )
