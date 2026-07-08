import torch


def validate_cont_dropout_p(p):
    """Validates a continuous-branch dropout rate. Valid range is [0, 1)."""
    if p == 1.0:
        raise ValueError(
            "cont_dropout_p=1.0 would train a pure VQ-VAE; use configs/vqvae.yaml instead. "
            "Valid range: [0, 1)."
        )
    if not (0.0 <= p < 1.0):
        raise ValueError(f"cont_dropout_p must be in [0, 1), got {p}.")


def apply_cont_dropout(z_vanilla_post, p, training):
    """Zeroes the continuous branch for a random per-sample subset, combine path only.

    Draws one Bernoulli keep-mask per sample (shape (B, 1, 1, 1), keep probability 1 - p)
    and multiplies it into a new tensor -- z_vanilla_post itself is left untouched so callers
    can still return the true (un-dropped) posterior in their loss dicts. No inverse-probability
    rescaling: kept samples keep their real magnitude, dropped samples are exact zeros.

    Short-circuits (no Bernoulli draw, no mask tensor) when dropout doesn't apply -- eval mode
    or p <= 0 -- returning z_vanilla_post unchanged and a drop fraction of 0.0.
    """
    if not training or p <= 0.0:
        return z_vanilla_post, 0.0

    batch_size = z_vanilla_post.shape[0]
    keep_prob = 1.0 - p
    keep_mask = torch.bernoulli(
        torch.full((batch_size, 1, 1, 1), keep_prob, device=z_vanilla_post.device)
    )
    drop_fraction = 1.0 - keep_mask.mean().item()
    return z_vanilla_post * keep_mask, drop_fraction
