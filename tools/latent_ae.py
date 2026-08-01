"""Uniform, frozen encode/decode interface over this repo's trained autoencoders.

The latent flow trainer must treat the DUALVAE and the vanilla VAE as interchangeable
latent providers: same call signature, same tensor layout, same normalization protocol.
Everything model-specific is confined here.

The architecture flags are read from the `config_used.yaml` that every training run drops
next to its checkpoint, so a run is identified by its checkpoint path alone -- no chance of
rebuilding a DUALVAE with the wrong rq_depth / residual_continuous / EMA flags and getting a
silent load_state_dict failure (or worse, a partial one).

Latent conventions:
  vae     -- VAE_Encoder already multiplies by the 0.18215 SD scale factor and VAE_Decoder
             divides it back out, so the latent handed to the flow model is the SCALED one
             and decode() takes the same.
  dualvae -- z = attention(z_vq + Delta), i.e. exactly what DUALVAE.forward feeds its
             decoder (DUALVAE_Decoder applies no scaling of its own).
Both are (B, latent_channels, H/downsample, W/downsample).
"""

import os

import torch
import torch.nn as nn

from models.dual_vae import DUALVAE
from models.vae import VAE


# Preference order when `ae_checkpoint` names a run DIRECTORY rather than a file.
# LAST-epoch weights first: `best.pt` in the autoencoder trainers is selected on the TRAIN
# loss (see save_if_best_val in train_vae.py / train_dualvae.py), and with a GAN active the
# train loss is not a meaningful model-selection signal -- the last epoch is the one whose
# codebook EMA statistics, discriminator balance and LR schedule all actually finished.
AE_CHECKPOINT_PREFERENCE = ("final_epoch.pt", "last.pt", "best.pt")


def resolve_ae_checkpoint(path, preference=AE_CHECKPOINT_PREFERENCE):
    """Turn `ae_checkpoint` into a concrete .pt file.

    * a DIRECTORY  -> the first of `preference` that exists inside it;
    * an existing FILE -> honored as-is (naming a specific .pt is an explicit override);
    * a MISSING file whose parent directory exists -> falls back to the preference order in
      that directory, so a config pointing at a checkpoint that was never written still runs.

    Raises with the directory listing when nothing usable is found.
    """
    path = os.path.abspath(path)

    if os.path.isfile(path):
        return path

    run_dir = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Autoencoder checkpoint directory not found: {run_dir}")

    for name in preference:
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            if not os.path.isdir(path):
                print(f"[AE] {os.path.basename(path)} not found in {run_dir}; using {name} instead.")
            return cand

    available = sorted(f for f in os.listdir(run_dir) if f.endswith(".pt"))
    raise FileNotFoundError(
        f"No usable autoencoder checkpoint for ae_checkpoint={path!r}. Searched {run_dir} for "
        f"{list(preference)}; it contains {available or 'no .pt files'}."
    )


def load_ae_config(ae_checkpoint, ae_config=None):
    """Read the autoencoder's training config. Defaults to the config_used.yaml that
    train_dualvae.py / train_vae.py saved in the checkpoint's own directory."""
    import yaml

    if ae_config is None:
        ae_config = os.path.join(os.path.dirname(os.path.abspath(ae_checkpoint)), "config_used.yaml")
    if not os.path.exists(ae_config):
        raise FileNotFoundError(
            f"Autoencoder config not found at {ae_config}. Pass `ae_config:` in the flow config "
            f"pointing at the YAML the autoencoder was trained with."
        )
    with open(ae_config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class FrozenLatentAE(nn.Module):
    """Frozen autoencoder exposing encode()/decode() for the latent generative model.

    Always in eval mode with requires_grad=False -- the flow model must not be able to move
    the latent space it is being measured on.
    """

    def __init__(self, model, kind, latent_channels, downsample_factor, ae_checkpoint, ae_cfg):
        super().__init__()
        self.model = model
        self.kind = kind
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.ae_checkpoint = ae_checkpoint
        self.ae_cfg = ae_cfg
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # Never leave eval(): dropout / EMA-codebook updates must stay off no matter what the
        # surrounding training loop calls.
        return super().train(False)

    def latent_shape(self, image_size):
        s = image_size // self.downsample_factor
        return (self.latent_channels, s, s)

    @torch.no_grad()
    def encode(self, x, sample=True):
        """(B, 3, H, W) image -> (B, C, H/f, W/f) latent.

        sample=True draws from the continuous posterior (what the decoder was trained on,
        and what LDM uses); sample=False uses the posterior MEAN (noise = 0), which is the
        deterministic latent the cache stores.
        """
        b, _, h, w = x.shape
        lh, lw = h // self.downsample_factor, w // self.downsample_factor
        if sample:
            noise = torch.randn((b, self.latent_channels, lh, lw), device=x.device, dtype=x.dtype)
        else:
            noise = torch.zeros((b, self.latent_channels, lh, lw), device=x.device, dtype=x.dtype)

        if self.kind == "vae":
            z, _, _ = self.model.encoder(x, noise)
            return z
        if self.kind == "dualvae":
            return self.model.encode_for_diffusion(x, noise=noise)
        raise ValueError(f"Unsupported autoencoder kind: {self.kind}")

    @torch.no_grad()
    def decode(self, z):
        """(B, C, H/f, W/f) latent -> (B, 3, H, W) image in the dataset's normalized range."""
        return self.model.decoder(z)


def load_frozen_ae(ae_checkpoint, device, ae_config=None):
    """Build + load the autoencoder named by `ae_checkpoint`, frozen, from its own config.

    `ae_checkpoint` may be a run directory (resolved via AE_CHECKPOINT_PREFERENCE) or a
    specific .pt file. The RESOLVED path is what the returned object reports, so the latent
    cache key and the flow checkpoint both record which weights were actually used.
    """
    ae_checkpoint = resolve_ae_checkpoint(ae_checkpoint)
    cfg = load_ae_config(ae_checkpoint, ae_config)
    kind = cfg.get("model", None)
    if kind not in ("vae", "dualvae"):
        raise ValueError(
            f"Latent flow training supports model: 'vae' and 'dualvae'; the config at "
            f"{ae_checkpoint} says model: {kind!r}."
        )

    downsample_factor = cfg.get("downsample_factor", 8)
    latent_channels = cfg.get("latent_channels", 8 if kind == "dualvae" else 4)

    if kind == "vae":
        model = VAE(downsample_factor=downsample_factor, latent_channels=latent_channels)
    else:
        # These flags change the state_dict (EMA buffers, the residual head's input width), so
        # they must mirror training exactly or the load below fails.
        model = DUALVAE(
            num_embeddings=cfg.get("num_embeddings", 512),
            latent_channels=latent_channels,
            commitment_cost=cfg.get("commitment_cost", 0.25),
            downsample_factor=downsample_factor,
            l2_normalize_codes=cfg.get("l2_normalize_codes", False),
            cont_dropout_p=cfg.get("cont_dropout_p", 0.0),
            use_ema_codebook=cfg.get("use_ema_codebook", False),
            ema_decay=cfg.get("ema_decay", 0.99),
            ema_eps=cfg.get("ema_eps", 1e-5),
            ema_dead_threshold=cfg.get("ema_dead_threshold", 1.0),
            rq_depth=cfg.get("rq_depth", 1),
            residual_continuous=cfg.get("residual_continuous", False),
            component_prior=cfg.get("component_prior", False),
            sigma2_floor=cfg.get("sigma2_floor", 1e-3),
            sigma2_ceil=cfg.get("sigma2_ceil", 10.0),
            wavelet_detail=cfg.get("wavelet_detail", False),
            wavelet_band_channels=cfg.get("wavelet_band_channels", None),
            hierarchical_semantic=cfg.get("hierarchical_semantic", False),
            coarse_factor=cfg.get("coarse_factor", 4),
            coarse_num_embeddings=cfg.get("coarse_num_embeddings", 64),
        )
        if model.hierarchical_semantic:
            # encode_for_diffusion() does not run the coarse level, so its latent would not be
            # the one the decoder expects. Fail loudly instead of generating garbage.
            raise NotImplementedError(
                "hierarchical_semantic DUALVAE checkpoints need a coarse-aware encode path; "
                "DUALVAE.encode_for_diffusion() only implements the single-level wiring."
            )

    state = torch.load(ae_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)

    ae = FrozenLatentAE(model, kind, latent_channels, downsample_factor, ae_checkpoint, cfg)
    print(
        f"[AE] loaded {kind} from {ae_checkpoint} "
        f"(latent_channels={latent_channels}, downsample_factor={downsample_factor}, frozen)"
    )
    return ae


class LatentStats:
    """Per-channel (or global) standardization of the AE's latents.

    Flow matching couples the data to a standard normal, so the data side must be on a
    comparable scale -- and the two latent spaces being compared are NOT on the same scale
    (the VAE's is pre-multiplied by 0.18215; the DUALVAE's carries codebook-sized codes plus
    a small offset). Standardizing each with its OWN statistics is what makes the comparison
    about the latent space's structure rather than about its arbitrary variance.

    Stats are estimated once on the training set and stored in the flow checkpoint, so
    sampling always un-normalizes with the exact values training used.
    """

    def __init__(self, mean, std, mode="per_channel"):
        self.mean = mean          # (1, C, 1, 1)
        self.std = std            # (1, C, 1, 1)
        self.mode = mode

    def to(self, device):
        return LatentStats(self.mean.to(device), self.std.to(device), self.mode)

    def normalize(self, z):
        return (z - self.mean.to(z.device, z.dtype)) / self.std.to(z.device, z.dtype)

    def denormalize(self, z):
        return z * self.std.to(z.device, z.dtype) + self.mean.to(z.device, z.dtype)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "mode": self.mode}

    @staticmethod
    def from_state_dict(d):
        return LatentStats(d["mean"], d["std"], d.get("mode", "per_channel"))

    @staticmethod
    def identity(channels):
        return LatentStats(torch.zeros(1, channels, 1, 1), torch.ones(1, channels, 1, 1), "none")

    @staticmethod
    def from_latents(latents, mode="per_channel", eps=1e-6):
        """latents: (N, C, H, W) float tensor of encoded training latents."""
        if mode == "none":
            return LatentStats.identity(latents.shape[1])
        x = latents.float()
        if mode == "per_channel":
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(eps)
        elif mode == "global":
            c = x.shape[1]
            mean = x.mean().reshape(1, 1, 1, 1).expand(1, c, 1, 1).contiguous()
            std = x.std().reshape(1, 1, 1, 1).expand(1, c, 1, 1).contiguous().clamp_min(eps)
        else:
            raise ValueError(f"latent_norm must be 'per_channel', 'global' or 'none', got {mode!r}.")
        return LatentStats(mean.cpu(), std.cpu(), mode)

    def __repr__(self):
        return (f"LatentStats(mode={self.mode}, mean={self.mean.flatten().tolist()}, "
                f"std={self.std.flatten().tolist()})")
