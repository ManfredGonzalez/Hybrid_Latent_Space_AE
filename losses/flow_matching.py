"""Conditional flow matching (rectified flow) objective and ODE samplers.

Objective (Lipman et al. 2023, "Flow Matching for Generative Modeling"; Liu et al. 2023,
"Rectified Flow"): couple a noise sample x0 ~ N(0, I) with a data latent x1 by the straight
path

    x_t = (1 - t) * x0 + t * x1,      t ~ p(t) on [0, 1]

whose exact velocity is the constant u = dx_t/dt = x1 - x0. Training is then plain
regression of the network onto that velocity,

    L = E_{t, x0, x1} || v_theta(x_t, t, y) - (x1 - x0) ||^2 ,

with no noise schedule, no SNR weighting and no variance-preserving reparameterization to
match across models -- which is exactly what makes it a clean instrument for comparing two
latent spaces: the only thing that changes between the two runs is the distribution of x1.

Sampling integrates dx/dt = v_theta(x, t, y) from t=0 (pure noise) to t=1 (data) with a
fixed step count, optionally with classifier-free guidance.
"""

import torch


def sample_timesteps(batch_size, device, mode="logit_normal", logit_normal_mean=0.0,
                     logit_normal_std=1.0, generator=None):
    """t ~ p(t) on (0, 1).

    'uniform'      -- the textbook choice.
    'logit_normal' -- t = sigmoid(n), n ~ N(mean, std^2), i.e. logit(t) ~ N(m, s^2). Default.
                      Justification: Esser et al. 2024 ("Scaling Rectified Flow Transformers
                      for High-Resolution Image Synthesis", ICML, arXiv:2403.03206) Sec. 3.1,
                      "Tailored SNR Samplers for RF Models". The RF loss is uniform in t but
                      the DIFFICULTY is not: at t=0 the optimal velocity prediction is just the
                      mean of p1 and at t=1 the mean of p0, both trivial, so uniform sampling
                      spends capacity where there is nothing to learn. Re-weighting the
                      timestep DENSITY (rather than the loss) concentrates training
                      mid-trajectory. Their large-scale formulation sweep ranks
                      rf/lognorm(0.00, 1.00) at the top, which is the default here.

    `generator` makes the draw reproducible, which is what lets the validation pass report a
    number that is comparable across epochs rather than a fresh Monte-Carlo sample each time.
    """
    if mode == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if mode == "logit_normal":
        n = torch.randn(batch_size, device=device, generator=generator) * logit_normal_std + logit_normal_mean
        return torch.sigmoid(n)
    raise ValueError(f"t_sampling must be 'uniform' or 'logit_normal', got {mode!r}.")


def flow_matching_loss(model, x1, y, t_sampling="logit_normal", logit_normal_mean=0.0,
                       logit_normal_std=1.0, cfg_dropout_prob=0.1, generator=None):
    """One conditional-flow-matching training term.

    Args:
        model: LatentFlowUNet (exposes `null_class`).
        x1: (B, C, H, W) data latents (already normalized by the trainer's latent stats).
        y: (B,) int64 class ids.
        cfg_dropout_prob: probability of replacing a label with the NULL class, which is what
            makes classifier-free guidance available at sampling time.

    Returns:
        (loss, aux dict with per-batch diagnostics)
    """
    b = x1.shape[0]
    device = x1.device

    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=generator)
    t = sample_timesteps(b, device, t_sampling, logit_normal_mean, logit_normal_std,
                         generator=generator)
    t_broadcast = t.view(b, *([1] * (x1.dim() - 1)))

    x_t = (1.0 - t_broadcast) * x0 + t_broadcast * x1
    target = x1 - x0

    if cfg_dropout_prob > 0.0:
        drop = torch.rand(b, device=device) < cfg_dropout_prob
        y = torch.where(drop, torch.full_like(y, model.null_class), y)

    pred = model(x_t, t, y)
    loss = (pred.float() - target.float()).pow(2).mean()
    return loss, {"t_mean": t.mean().detach()}


@torch.no_grad()
def _velocity(model, x, t_scalar, y, guidance_scale, null_class, bad_model=None):
    """v_theta with optional classifier-free guidance or AutoGuidance.

    CLASSIFIER-FREE GUIDANCE (bad_model is None). guidance_scale <= 1 runs a single
    conditional pass; above that the conditional and unconditional velocities are computed in
    ONE batched forward (the two halves are concatenated) and extrapolated:
        v = v_uncond + w * (v_cond - v_uncond).
    The unconditional branch is more diverse than the conditional one, so large w trades
    diversity for fidelity -- which is why FID in w is U-shaped.

    AUTOGUIDANCE (bad_model given; Karras et al. 2025, "Guiding a diffusion model with a bad
    version of itself", NeurIPS, arXiv:2406.02507). Replace the unconditional branch with a
    DEGRADED version of the same model -- same architecture, same data, same conditioning,
    just less trained:
        v = v_bad(x, t, y) + w * (v_good(x, t, y) - v_bad(x, t, y)).
    Both branches are CONDITIONAL, so this amplifies what the converged model knows that the
    under-trained one does not, without deleting class diversity the way CFG's unconditional
    branch does. Karras et al. show this improves FID where CFG only trades it against
    diversity. Cost is identical to CFG: two network evaluations per step.

    w = 1.0 is exactly the unguided model under BOTH schemes, so it is the shared baseline.
    """
    b = x.shape[0]
    t = torch.full((b,), t_scalar, device=x.device, dtype=torch.float32)

    if bad_model is not None:
        v_good = model(x, t, y)
        if guidance_scale is None or guidance_scale == 1.0:
            return v_good
        v_bad = bad_model(x, t, y)
        return v_bad + guidance_scale * (v_good - v_bad)

    if guidance_scale is None or guidance_scale <= 1.0:
        return model(x, t, y)
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    y_in = torch.cat([y, torch.full_like(y, null_class)], dim=0)
    v_cond, v_uncond = model(x_in, t_in, y_in).chunk(2, dim=0)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_flow(model, shape, y, num_steps=50, guidance_scale=1.0, device="cuda",
                solver="heun", generator=None, x0=None, dtype=torch.float32, bad_model=None):
    """Integrate dx/dt = v_theta(x, t, y) from t=0 (noise) to t=1 (data).

    Args:
        shape: (B, C, H, W) of the latents to generate.
        y: (B,) int64 class ids to condition on.
        num_steps: uniform Euler/Heun steps. Heun costs 2 network evaluations per step but is
            2nd order, so it is markedly more accurate at equal step count; report the step
            count alongside FID when comparing runs.
        guidance_scale: classifier-free guidance weight w (1.0 = no guidance).
        x0: optional starting noise (for reproducible side-by-side comparisons across models).
        bad_model: optional degraded model enabling AutoGuidance instead of CFG (see
            _velocity). Must share this model's architecture, latent shape and latent stats --
            an earlier checkpoint of the SAME run is the canonical choice.

    Returns the (B, C, H, W) generated latents at t=1.
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}.")

    was_training = model.training
    model.eval()

    x = torch.randn(shape, device=device, dtype=dtype, generator=generator) if x0 is None else x0.to(device=device, dtype=dtype)
    y = y.to(device)
    dt = 1.0 / num_steps
    null_class = model.null_class

    for i in range(num_steps):
        t = i * dt
        v = _velocity(model, x, t, y, guidance_scale, null_class, bad_model).to(dtype)
        # Heun's corrector needs v at t + dt; on the final step that lands exactly at t = 1,
        # which the logit-normal timestep sampling barely trains -- so the last step always
        # falls back to Euler.
        if solver == "euler" or i == num_steps - 1:
            x = x + dt * v
        else:
            x_pred = x + dt * v
            v2 = _velocity(model, x_pred, t + dt, y, guidance_scale, null_class, bad_model).to(dtype)
            x = x + dt * 0.5 * (v + v2)

    if was_training:
        model.train()
    return x
