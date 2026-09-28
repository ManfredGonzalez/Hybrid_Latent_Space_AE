"""Samplers matching REPA-E's evaluation protocol, ported from their samplers.py (MIT).

WHY THIS FILE EXISTS SEPARATELY FROM losses/flow_matching.py. Our own sampler integrates the
ODE with Euler/Heun and plain classifier-free guidance, which is fine for comparing our arms
against each other. It is NOT what REPA-E reports: their tables come from an SDE
(Euler-Maruyama) sampler with a guidance INTERVAL, and sampler choice moves FID by more than
most of the differences we are trying to measure. Anything meant to sit next to their numbers
has to be generated the way they generate.

THE TIME CONVENTION IS FLIPPED, AND THIS IS THE ONE THING TO GET RIGHT.
  theirs:  x_t = (1 - t) * data + t * noise      -> t = 0 is DATA,  t = 1 is NOISE
  ours:    x_t = (1 - t) * noise + t * data      -> t = 0 is NOISE, t = 1 is DATA
The two are related by t_ours = 1 - t_theirs, and their velocity is the negative of ours:
    v_theirs(x, t') = -v_ours(x, 1 - t').
Rather than re-derive their score/diffusion formulas in our convention (easy to get subtly
wrong, and the error would look like a mediocre FID rather than a crash), the integrator below
is a faithful port that runs in THEIR time units, and `_velocity_adapter` translates each model
call. Their guidance_low/guidance_high therefore keep their published meaning.
"""

import numpy as np
import torch


def expand_t_like_x(t, x):
    """(B,) -> (B, 1, 1, ...) so it broadcasts against x."""
    return t.view(-1, *([1] * (x.dim() - 1)))


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Convert a velocity prediction into a score, in REPA-E's time convention."""
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError(f"path_type {path_type!r}")

    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    return (reverse_alpha_ratio * vt - xt) / var


def compute_diffusion(t_cur):
    """REPA-E's diffusion coefficient for the SDE sampler."""
    return 2 * t_cur


def _velocity_adapter(model, x, t_theirs, y):
    """Call OUR SiT with THEIR time convention: v_theirs(x, t') = -v_ours(x, 1 - t')."""
    t_ours = (1.0 - t_theirs).to(dtype=x.dtype)
    return -model.velocity_field(x, t_ours, y)


@torch.no_grad()
def euler_maruyama_sampler(model, latents, y, num_steps=250, cfg_scale=1.0,
                           guidance_low=0.0, guidance_high=1.0, path_type="linear",
                           null_class=1000):
    """SDE sampler, a port of REPA-E's. `latents` is the starting NOISE, in normalized space.

    The guidance interval matters: classifier-free guidance is applied only while
    guidance_low <= t <= guidance_high (in their time units, where t counts DOWN from 1 to 0).
    Outside it the model runs conditionally with no extrapolation, which is what keeps guidance
    from destroying diversity at the ends of the trajectory.
    """
    if cfg_scale > 1.0:
        y_null = torch.full_like(y, null_class)

    dtype = latents.dtype
    device = latents.device
    t_steps = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.0], dtype=torch.float64)])
    x_next = latents.to(torch.float64)

    def _drift(x_cur, t_cur):
        guided = cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
        model_input = torch.cat([x_cur] * 2, dim=0) if guided else x_cur
        y_cur = torch.cat([y, y_null], dim=0) if guided else y
        t_in = torch.ones(model_input.shape[0], device=device, dtype=torch.float64) * t_cur

        v_cur = _velocity_adapter(model, model_input.to(dtype), t_in, y_cur).to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, t_in, path_type=path_type)
        d_cur = v_cur - 0.5 * compute_diffusion(t_cur) * s_cur
        if guided:
            d_cond, d_uncond = d_cur.chunk(2)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
        return d_cur

    for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
        dt = t_next - t_cur
        x_cur = x_next
        d_cur = _drift(x_cur, t_cur)
        eps = torch.randn_like(x_cur) * torch.sqrt(torch.abs(dt))
        x_next = x_cur + d_cur * dt + torch.sqrt(compute_diffusion(t_cur)) * eps

    # Final step is deterministic: no noise is injected, so the sample lands on the mean.
    t_cur, t_next = t_steps[-2], t_steps[-1]
    x_next = x_next + (t_next - t_cur) * _drift(x_next, t_cur)
    return x_next.to(dtype)


@torch.no_grad()
def euler_sampler(model, latents, y, num_steps=50, cfg_scale=1.0, guidance_low=0.0,
                  guidance_high=1.0, null_class=1000):
    """Deterministic ODE counterpart, same time convention and same guidance interval."""
    if cfg_scale > 1.0:
        y_null = torch.full_like(y, null_class)
    dtype = latents.dtype
    device = latents.device
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        dt = t_next - t_cur
        guided = cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
        model_input = torch.cat([x_next] * 2, dim=0) if guided else x_next
        y_cur = torch.cat([y, y_null], dim=0) if guided else y
        t_in = torch.ones(model_input.shape[0], device=device, dtype=torch.float64) * t_cur
        d_cur = _velocity_adapter(model, model_input.to(dtype), t_in, y_cur).to(torch.float64)
        if guided:
            d_cond, d_uncond = d_cur.chunk(2)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
        x_next = x_next + d_cur * dt
    return x_next.to(dtype)
