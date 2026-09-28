"""SiT velocity field with the REPA-E wiring: internal BatchNorm + truncated alignment pass.

Adapted from the REPA-E reference implementation (Leng, Singh et al., ICCV 2025, MIT licence,
https://github.com/End2End-Diffusion/REPA-E), which is itself derived from SiT / DiT. Two
deliberate differences:

  * NO timm DEPENDENCY. The project environment (`ddpm`) has torch but not timm, so PatchEmbed
    / Attention / Mlp are implemented here. They are the standard versions; attention uses
    F.scaled_dot_product_attention, which is what timm would dispatch to anyway.
  * `in_channels` defaults to 8 (our DualVAE latent), not 4 (SD-VAE).

WHY THE BATCHNORM LIVES INSIDE THIS MODULE (and not between the two models, as a diagram would
suggest). Two-stage latent diffusion normalizes the latent with a constant measured once on the
frozen tokenizer (SD-VAE's 0.18215). Under end-to-end training the tokenizer moves, so that
constant goes stale and re-measuring it over the dataset after every step is impossible. A
BatchNorm with no affine parameters tracks the statistics for free: batch statistics in train
mode, running statistics in eval mode. Keeping it inside the model means the checkpoint carries
its own normalization, and sampling cannot silently use different numbers from training.
Affine is DISABLED on purpose: a learnable scale here would let the model shrink the latent to
make the denoising objective easier, which is the failure mode end-to-end training must avoid.

WHY forward() COMPUTES THE LOSSES. The alignment pass needs the hidden state at
`encoder_depth` and nothing after it; the diffusion pass needs the full stack. Computing the
interpolant and both losses inside forward() is what lets `align_only=True` stop the block loop
early (`break`) instead of running 28 blocks to throw 20 of them away. It also keeps the
timestep and the noise -- which the trainer reuses across the two passes -- in one place.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------- #
# Small building blocks (timm-free)
# --------------------------------------------------------------------------------------- #

def modulate(x, shift, scale):
    """adaLN modulation: (B, T, D) scaled/shifted by per-sample (B, D) vectors."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def mean_flat(x):
    """Mean over every non-batch dimension -> (B,)."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    """REPA projector: SiT hidden state -> the representation encoder's feature dimension."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, (H/p)*(W/p), D) via a strided conv, exactly like ViT."""

    def __init__(self, input_size, patch_size, in_channels, hidden_size):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size {input_size} is not divisible by patch_size {patch_size}.")
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention. No mask: every latent token attends to every other."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden size {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                       # each (B, heads, T, head_dim)
        x = F.scaled_dot_product_attention(q, k, v)   # flash/mem-efficient kernel when available
        return self.proj(x.transpose(1, 2).reshape(b, t, d))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    """Continuous flow time t in [0, 1] -> (B, D), sinusoidal features then an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.positional_embedding(t, self.frequency_embedding_size).to(t.dtype))


class LabelEmbedder(nn.Module):
    """Class embedding with the extra NULL row used by classifier-free guidance.

    The table has num_classes + 1 rows; `token_drop` replaces a label with the last row with
    probability `dropout_prob` during training, which is what makes an unconditional branch
    available at sampling time.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + (1 if dropout_prob > 0 else 0), hidden_size)

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning on (timestep + class)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def get_2d_sincos_pos_embed(embed_dim, grid_size, device=None):
    """(grid_size^2, embed_dim) fixed sin-cos positional embedding, row-major over (y, x)."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim {embed_dim} must be divisible by 4 for 2D sin-cos embedding.")
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    def _embed_1d(pos):
        omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
        omega = 1.0 / (10000 ** (omega / (embed_dim / 4.0)))
        out = pos.reshape(-1)[:, None] * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    # x first, matching the reference implementation's ordering.
    return torch.cat([_embed_1d(grid_x), _embed_1d(grid_y)], dim=1)


# --------------------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------------------- #

class SiT(nn.Module):
    """Class-conditional velocity field over a (C, H, W) latent, with REPA projectors.

    forward() returns a dict rather than a tensor because it also computes the losses -- see
    the module docstring for why. Keys:
        denoising_loss : (B,) per-sample flow-matching loss, or None when align_only
        proj_loss      : () scalar REPA loss (negative cosine similarity, averaged)
        time_input     : (B, 1, 1, 1) the timesteps used, so the second pass can reuse them
        noises         : (B, C, H, W) the noise used, likewise
        velocity       : (B, C, H, W) the predicted velocity, or None when align_only
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=8,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        encoder_depth=8,
        class_dropout_prob=0.1,
        num_classes=1000,
        z_dims=(768,),
        projector_dim=2048,
        bn_momentum=0.1,
        bn_eps=1e-4,
    ):
        super().__init__()
        if not 1 <= encoder_depth <= depth:
            raise ValueError(f"encoder_depth must be in [1, {depth}], got {encoder_depth}.")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.z_dims = list(z_dims)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.projectors = nn.ModuleList(
            [build_mlp(hidden_size, projector_dim, z_dim) for z_dim in self.z_dims])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # affine=False: see the module docstring. track_running_stats=True is what makes the
        # eval-mode alignment pass deterministic and what sampling reads back.
        self.bn = nn.BatchNorm2d(in_channels, eps=bn_eps, momentum=bn_momentum,
                                 affine=False, track_running_stats=True)
        self.bn.reset_running_stats()
        self.initialize_weights()

    # ---------------------------------------------------------------- init / normalization #

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        self.pos_embed.data.copy_(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size).unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # adaLN-Zero: every block starts as the identity, so training begins from a clean
        # residual stream instead of a random perturbation of it.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.no_grad()
    def init_bn(self, mean, std):
        """Warm-start the BN running statistics from a one-off pass over the frozen tokenizer.

        Not strictly required -- with momentum 0.1 the running values converge in ~30 steps --
        but the ALIGNMENT pass runs in eval mode, so before they converge the tokenizer would
        receive REPA gradients computed on badly scaled latents. `mean` and `std` are
        per-channel, in the space this model is fed (our pre-attention latent z_q + Delta).
        """
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.in_channels or std.numel() != self.in_channels:
            raise ValueError(f"init_bn expects {self.in_channels} per-channel values, "
                             f"got mean {tuple(mean.shape)} std {tuple(std.shape)}.")
        self.bn.running_mean.copy_(mean.to(self.bn.running_mean.device))
        self.bn.running_var.copy_((std ** 2).to(self.bn.running_var.device))

    @torch.no_grad()
    def latent_stats(self):
        """(mean, std) currently used for normalization -- what sampling must un-normalize with."""
        return self.bn.running_mean.clone(), (self.bn.running_var + self.bn.eps).sqrt()

    def normalize_latent(self, z):
        """Apply the CURRENT running statistics without touching them (for the mode tracker)."""
        mean, std = self.latent_stats()
        return (z - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    # ------------------------------------------------------------------------- the forward #

    def unpatchify(self, x):
        p, c = self.patch_size, self.out_channels
        h = w = self.x_embedder.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, h * p, w * p)

    @torch.no_grad()
    def features(self, x, t, y, depth=None):
        """(B, T, D) hidden state after `depth` blocks, on an ALREADY NORMALIZED, ALREADY
        NOISED latent. Defaults to `encoder_depth`, the layer REPA aligns.

        Exposed for representation analysis (CKNNA). forward() returns only the PROJECTED
        features, because that is what the alignment loss consumes; the projector is trained,
        so measuring alignment through it would conflate "the features are DINOv2-like" with
        "the projector learned to make them look DINOv2-like". CKNNA is meant to ask the
        former, so it needs the hidden state itself.
        """
        depth = self.encoder_depth if depth is None else depth
        h = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t.flatten()) + self.y_embedder(y, False)
        for i, block in enumerate(self.blocks):
            h = block(h, c)
            if (i + 1) == depth:
                break
        return h

    def velocity_field(self, x, t, y, force_drop_ids=None):
        """Plain v_theta(x_t, t, y) on an ALREADY NORMALIZED latent -- the sampler's entry point.

        Kept separate from forward() so sampling never touches the loss machinery (and never
        updates the BN statistics).
        """
        h = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t.flatten()) + self.y_embedder(y, self.training, force_drop_ids)
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.final_layer(h, c))

    def forward(self, x, y, zs, align_only=False, time_input=None, noises=None,
                t_sampling="uniform", logit_normal_mean=0.0, logit_normal_std=1.0):
        """One training step's forward pass.

        Args:
            x: (B, C, H, W) UNNORMALIZED latent. The trainer passes it attached for the
               alignment pass and detached for the diffusion pass -- that detach IS the
               stop-gradient that keeps the diffusion loss out of the tokenizer.
            zs: list of (B, T, z_dim) frozen representation-encoder patch tokens.
            align_only: stop after `encoder_depth` blocks and skip the diffusion loss.
            time_input / noises: reuse the draws from a previous call, so both passes of a
               step see the identical interpolant.
        """
        normalized_x = self.bn(x)
        b = normalized_x.shape[0]

        if time_input is None:
            if t_sampling == "uniform":
                time_input = torch.rand((b, 1, 1, 1))
            elif t_sampling == "logit_normal":
                n = torch.randn((b, 1, 1, 1)) * logit_normal_std + logit_normal_mean
                time_input = torch.sigmoid(n)
            else:
                raise ValueError(f"t_sampling must be 'uniform' or 'logit_normal', got {t_sampling!r}.")
        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)
        if noises is None:
            noises = torch.randn_like(normalized_x)
        else:
            noises = noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        # Linear path x_t = (1-t) x0 + t x1 with x0 = noise, x1 = data. Its exact velocity is
        # the constant x1 - x0, which is the regression target (rectified flow / SiT 'v').
        model_input = (1.0 - time_input) * noises + time_input * normalized_x
        model_target = normalized_x - noises

        h = self.x_embedder(model_input) + self.pos_embed
        c = self.t_embedder(time_input.flatten()) + self.y_embedder(y, self.training)

        zs_tilde = None
        for i, block in enumerate(self.blocks):
            h = block(h, c)
            if (i + 1) == self.encoder_depth:
                n, t_len, d = h.shape
                zs_tilde = [p(h.reshape(-1, d)).reshape(n, t_len, -1) for p in self.projectors]
                if align_only:
                    break

        denoising_loss, velocity = None, None
        if not align_only:
            velocity = self.unpatchify(self.final_layer(h, c))
            denoising_loss = mean_flat((velocity.float() - model_target.float()) ** 2)

        # REPA: patch-wise cosine similarity between the projected hidden state and the frozen
        # encoder's tokens. Negated so that minimizing it maximizes alignment.
        proj_loss = torch.zeros((), device=normalized_x.device)
        for z, z_tilde in zip(zs, zs_tilde):
            z_tilde = F.normalize(z_tilde.float(), dim=-1)
            z = F.normalize(z.float(), dim=-1)
            proj_loss = proj_loss - (z * z_tilde).sum(dim=-1).mean()
        proj_loss = proj_loss / max(1, len(zs))

        return {
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
            "velocity": velocity,
        }


# --------------------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------------------- #

def SiT_S_2(**kwargs):
    """33M. Not a research setting -- it is what makes a CPU smoke test finish in seconds."""
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


SiT_models = {"SiT-S/2": SiT_S_2, "SiT-B/2": SiT_B_2, "SiT-L/2": SiT_L_2, "SiT-XL/2": SiT_XL_2}


def build_sit(name, **kwargs):
    if name not in SiT_models:
        raise ValueError(f"Unknown SiT variant {name!r}; available: {sorted(SiT_models)}.")
    return SiT_models[name](**kwargs)
