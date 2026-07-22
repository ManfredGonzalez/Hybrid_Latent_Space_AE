"""Single-level Haar Discrete Wavelet Transform as a fixed (non-learned) conv.

Purpose (see the frequency-band-variance design): a strided convolution downsamples by
averaging, which DESTROYS the high-frequency detail before it can reach the bottleneck.
A DWT downsamples 2x LOSSLESSLY -- the detail a strided conv would have thrown away is
preserved explicitly in three high-frequency subbands. Feeding those subbands to the
continuous ("detail") branch is what finally gives it access to pre-compression detail.

One level on an (B, 3, H, W) image yields four (B, 3, H/2, W/2) subbands:
  LL  low-low   : coarse approximation (the "what") -> routed to the VQ/base path
  LH  low-high  : horizontal-edge detail            -> continuous detail branch
  HL  high-low  : vertical-edge detail              -> continuous detail branch
  HH  high-high : diagonal / finest texture         -> continuous detail branch

The transform is orthonormal (0.5 * Haar filters), so it is exactly invertible and
preserves total energy; forward() returns LL separately and the three HF bands stacked
as (B, 9, H/2, W/2) in band-major order [LH(3), HL(3), HH(3)]. NUM_BANDS = 3.

Fixed weights (requires_grad=False): the DWT is a deterministic basis change, not
something to learn. Runs fine under autocast (it's just a conv), but callers doing the
GMM statistics should cast to fp32 as usual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BANDS = 3          # LH, HL, HH  (LL is handled separately as the base path)
BAND_NAMES = ("LH", "HL", "HH")


def _haar_filters():
    """Four 2x2 Haar analysis filters, L2-normalized (factor 0.5), as a (4,1,2,2) tensor
    ordered [LL, LH, HL, HH]."""
    h = 0.5
    ll = torch.tensor([[1., 1.], [1., 1.]])
    lh = torch.tensor([[1., 1.], [-1., -1.]])
    hl = torch.tensor([[1., -1.], [1., -1.]])
    hh = torch.tensor([[1., -1.], [-1., 1.]])
    return (h * torch.stack([ll, lh, hl, hh])).unsqueeze(1)  # (4,1,2,2)


class HaarDWT(nn.Module):
    """Single-level Haar DWT for RGB input via a fixed grouped conv (stride 2)."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        # One filter bank per input channel: groups=in_channels, 4 outputs each.
        # Output channel layout from grouped conv: for input channel c, outputs
        # [4c, 4c+1, 4c+2, 4c+3] = [LL_c, LH_c, HL_c, HH_c].
        weight = _haar_filters().repeat(in_channels, 1, 1, 1)  # (4*C, 1, 2, 2)
        self.register_buffer("weight", weight, persistent=False)

        # Gather indices to split the 4*C conv outputs back into per-band (C-channel)
        # tensors. LL = [0,4,8,...], LH = [1,5,9,...], etc.
        base = torch.arange(in_channels) * 4
        self.register_buffer("ll_idx", base + 0, persistent=False)
        self.register_buffer("lh_idx", base + 1, persistent=False)
        self.register_buffer("hl_idx", base + 2, persistent=False)
        self.register_buffer("hh_idx", base + 3, persistent=False)

    def forward(self, x):
        """x: (B, C, H, W) with even H, W. Returns (LL, HF) where
        LL: (B, C, H/2, W/2); HF: (B, 3C, H/2, W/2) as [LH, HL, HH] band-major."""
        # Pad to even spatial dims if needed (H, W are 256 here, already even).
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        w = self.weight.to(x.dtype)
        coeffs = F.conv2d(x, w, stride=2, groups=self.in_channels)  # (B, 4C, H/2, W/2)
        ll = coeffs.index_select(1, self.ll_idx)
        lh = coeffs.index_select(1, self.lh_idx)
        hl = coeffs.index_select(1, self.hl_idx)
        hh = coeffs.index_select(1, self.hh_idx)
        hf = torch.cat([lh, hl, hh], dim=1)  # (B, 3C, H/2, W/2), band-major
        return ll, hf
