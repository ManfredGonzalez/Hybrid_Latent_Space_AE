"""Per-band detail encoder for the wavelet detail branch.

Takes the three high-frequency Haar subbands (LH, HL, HH), each with `in_per_band`
channels (3 for RGB), and encodes EACH band SEPARATELY through its own small conv stack.
The per-band outputs are concatenated so that the resulting Delta channels are grouped by
frequency band -- this is what makes "frequency-band-dependent variance" well defined: a
known channel slice belongs to LH, the next to HL, the last to HH, so a per-band prior
sigma0 (or a per-(code,band) variance diagnostic) can be applied to the right channels.

Output: (B, sum(band_channels), H_in/down, W_in/down). With band_channels=[3,3,2] the
8 output channels partition as LH->0:3, HL->3:6, HH->6:8, matching latent_channels=8.

Downsampling: `down` = 4 by default takes the 128x128 HF subbands (from a 256px image,
one DWT level) to 32x32, aligning with the base VQ latent grid so z = e_k + Delta adds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import ResidualBlock


class _BandEncoder(nn.Module):
    """One band: (B, in_ch, H, W) -> (B, out_ch, H/down, W/down). Internal width `width`
    is kept >= 32 and divisible by 32 for the ResidualBlock GroupNorm; the final 1x1 conv
    projects to the (small) out_ch without a GroupNorm."""

    def __init__(self, in_ch, out_ch, down=4, width=64):
        super().__init__()
        assert width % 32 == 0, "width must be divisible by 32 (GroupNorm groups)"
        # down is a power of 2; number of stride-2 stages = log2(down).
        n_down = 0
        d = down
        while d > 1:
            d //= 2
            n_down += 1

        layers = [nn.Conv2d(in_ch, width, kernel_size=3, padding=1), ResidualBlock(width, width)]
        for _ in range(n_down):
            layers.append(nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1))
            layers.append(ResidualBlock(width, width))
        layers += [nn.GroupNorm(32, width), nn.SiLU(), nn.Conv2d(width, out_ch, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WaveletDetailEncoder(nn.Module):
    def __init__(self, in_per_band=3, band_channels=(3, 3, 2), down=4, width=64):
        super().__init__()
        self.in_per_band = in_per_band
        self.band_channels = list(band_channels)
        self.num_bands = len(self.band_channels)
        self.encoders = nn.ModuleList([
            _BandEncoder(in_per_band, ch, down=down, width=width) for ch in self.band_channels
        ])

    def band_channel_index(self):
        """Return a (sum band_channels,) long tensor mapping each output channel to its
        band id (0=LH, 1=HL, 2=HH). Used to broadcast per-band sigma0 / variance stats
        onto the flat channel axis."""
        ids = []
        for b, ch in enumerate(self.band_channels):
            ids += [b] * ch
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, hf):
        """hf: (B, num_bands * in_per_band, H, W), band-major [LH, HL, HH].
        Returns Delta: (B, sum(band_channels), H/down, W/down)."""
        outs = []
        for b, enc in enumerate(self.encoders):
            start = b * self.in_per_band
            band = hf[:, start:start + self.in_per_band]
            outs.append(enc(band))
        return torch.cat(outs, dim=1)
