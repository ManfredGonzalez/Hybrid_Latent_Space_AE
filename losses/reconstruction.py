"""Pluggable reconstruction / perceptual loss dispatcher shared by all four trainers.

The reconstruction objective is always: pixel_term + perceptual_weight * extra_term.
See build_reconstruction_criterion() for the available modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

from tools.normalization import denormalize

try:
    import lpips as lpips_pkg
except ImportError:
    lpips_pkg = None

VALID_PERCEPTUAL_LOSSES = ("none", "lpips", "msssim_l1", "ffl", "msssim_ffl")

_MSSSIM_DEFAULT_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _adaptive_msssim_kwargs(h, w):
    """Pick a kernel_size/betas pair that fits the image size.

    torchmetrics' default (kernel_size=11, 5 betas) needs >=176px on the short side
    (each scale halves the resolution). Datasets here range from CIFAR-sized 32px up
    to 256px, and tests use tiny tensors, so scale the kernel/level count down instead
    of hard-failing on small inputs.
    """
    min_dim = min(h, w)
    kernel_size = 11
    if min_dim < kernel_size:
        kernel_size = min_dim if min_dim % 2 == 1 else max(1, min_dim - 1)
    n_scales = 1
    dim = min_dim
    while n_scales < len(_MSSSIM_DEFAULT_BETAS) and dim // 2 >= kernel_size:
        dim //= 2
        n_scales += 1
    betas = _MSSSIM_DEFAULT_BETAS[:n_scales]
    betas = tuple(b / sum(betas) for b in betas)
    return kernel_size, betas


def focal_frequency_loss(recon, target, alpha=1.0, eps=1e-8):
    """Focal Frequency Loss (Jiang et al., ICCV 2021).

    Weights each frequency's squared spectral distance by its own (detached,
    per-image max-normalized) magnitude, so the loss focuses on frequencies the
    model currently reconstructs poorly instead of weighting all frequencies equally.
    """
    recon_freq = torch.fft.fft2(recon, norm="ortho")
    target_freq = torch.fft.fft2(target, norm="ortho")
    diff = recon_freq - target_freq

    freq_distance = diff.real.pow(2) + diff.imag.pow(2)
    weight = diff.abs().pow(alpha)
    weight = weight / (weight.amax(dim=(-3, -2, -1), keepdim=True) + eps)
    weight = weight.detach()

    return (weight * freq_distance).mean()


def _build_lpips_model(device):
    if lpips_pkg is None:
        raise ImportError(
            "perceptual_loss='lpips' requires the `lpips` package, which is not installed. "
            "Install it with `pip install lpips`."
        )
    try:
        model = lpips_pkg.LPIPS(net="vgg")
    except Exception as e:
        raise RuntimeError(
            "Failed to construct lpips.LPIPS(net='vgg'). Ensure the `lpips` package is "
            "installed correctly (pip install lpips) and its VGG backbone weights are "
            "reachable (torchvision downloads them to the torch hub cache on first use)."
        ) from e
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


class ReconstructionCriterion(nn.Module):
    """criterion(recon, target) -> (total_recon_loss, pixel_term, extra_term), all scalars.

    recon/target are expected in the dataset's normalized space (whatever the trainer's
    dataloader emits); denormalization to [0, 1] happens internally where needed.
    """

    def __init__(
        self,
        name,
        device,
        perceptual_weight=1.0,
        ffl_alpha=1.0,
        dataset_name="imagenette",
        perceptual_batch_fraction=1.0,
    ):
        super().__init__()
        if name not in VALID_PERCEPTUAL_LOSSES:
            raise ValueError(
                f"Unknown perceptual_loss '{name}'. Expected one of {VALID_PERCEPTUAL_LOSSES}."
            )
        if not (0.0 < perceptual_batch_fraction <= 1.0):
            raise ValueError(
                f"perceptual_batch_fraction must be in (0, 1], got {perceptual_batch_fraction}."
            )

        self.name = name
        self.device = device
        self.perceptual_weight = perceptual_weight
        self.ffl_alpha = ffl_alpha
        self.dataset_name = dataset_name
        self.perceptual_batch_fraction = perceptual_batch_fraction

        self.lpips_model = _build_lpips_model(device) if name == "lpips" else None

    def _denorm(self, recon, target):
        recon01 = denormalize(recon, self.dataset_name, self.device)
        target01 = denormalize(target, self.dataset_name, self.device)
        return recon01, target01

    def _lpips_term(self, recon01, target01):
        b = recon01.size(0)
        if self.perceptual_batch_fraction < 1.0 and b > 1:
            k = max(1, round(b * self.perceptual_batch_fraction))
            idx = torch.randperm(b, device=recon01.device)[:k]
            recon01 = recon01[idx]
            target01 = target01[idx]
        recon_lpips = recon01 * 2 - 1
        target_lpips = target01 * 2 - 1
        return self.lpips_model(recon_lpips, target_lpips).mean()

    def _msssim_l1(self, recon01, target01):
        kernel_size, betas = _adaptive_msssim_kwargs(recon01.shape[-2], recon01.shape[-1])
        msssim_val = multiscale_structural_similarity_index_measure(
            recon01, target01, data_range=1.0, kernel_size=kernel_size, betas=betas
        )
        l1_val = F.l1_loss(recon01, target01, reduction="mean")
        return 0.84 * (1 - msssim_val) + 0.16 * l1_val

    def forward(self, recon, target):
        recon = recon.float()
        target = target.float()
        zero = torch.zeros((), device=recon.device, dtype=recon.dtype)

        if self.name == "none":
            pixel_term = F.mse_loss(recon, target, reduction="mean")
            return pixel_term, pixel_term, zero

        if self.name == "lpips":
            pixel_term = F.mse_loss(recon, target, reduction="mean")
            recon01, target01 = self._denorm(recon, target)
            extra_term = self._lpips_term(recon01, target01)
        elif self.name == "ffl":
            pixel_term = F.mse_loss(recon, target, reduction="mean")
            recon01, target01 = self._denorm(recon, target)
            extra_term = focal_frequency_loss(recon01, target01, alpha=self.ffl_alpha)
        elif self.name == "msssim_l1":
            recon01, target01 = self._denorm(recon, target)
            pixel_term = self._msssim_l1(recon01, target01)
            extra_term = zero
        elif self.name == "msssim_ffl":
            recon01, target01 = self._denorm(recon, target)
            pixel_term = self._msssim_l1(recon01, target01)
            extra_term = focal_frequency_loss(recon01, target01, alpha=self.ffl_alpha)
        else:
            raise AssertionError(f"unreachable: {self.name}")

        total = pixel_term + self.perceptual_weight * extra_term
        return total, pixel_term, extra_term


def build_reconstruction_criterion(
    name,
    device,
    perceptual_weight=1.0,
    ffl_alpha=1.0,
    dataset_name="imagenette",
    perceptual_batch_fraction=1.0,
):
    return ReconstructionCriterion(
        name=name,
        device=device,
        perceptual_weight=perceptual_weight,
        ffl_alpha=ffl_alpha,
        dataset_name=dataset_name,
        perceptual_batch_fraction=perceptual_batch_fraction,
    )
