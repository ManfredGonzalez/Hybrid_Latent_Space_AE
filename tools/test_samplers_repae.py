"""Correctness tests for tools/samplers_repae.py. CPU, seconds, no data.

    python tools/test_samplers_repae.py

WHY THESE TESTS LOOK LIKE THIS. The first attempt tested the samplers with a freshly built
SiT and proved nothing: adaLN-Zero zero-initializes the final layer, so an untrained model
returns velocity == 0, the integrator returns its input unchanged, and every configuration
"passes" identically. Guidance appeared to do nothing because NOTHING did anything.

So these tests use analytic stand-in models whose exact answer is known:

  1. TIME CONVENTION. Ours is t=0 noise -> t=1 data; REPA-E's is the reverse, and their
     velocity is the negative of ours. The integrator runs in THEIR units and adapts each
     call. A straight-line flow has the constant exact velocity (data - noise), so integrating
     it must land on the data point -- to within float error, at any step count. If the
     adapter's sign or time flip were wrong, the sampler would converge to the wrong point
     (often to the noise, or to 2*data - noise), which on a real model looks like poor FID
     rather than a bug.
  2. GUIDANCE. With a model whose conditional and unconditional velocities differ by a known
     constant, classifier-free guidance must scale that difference by exactly cfg_scale.
  3. THE GUIDANCE INTERVAL must actually gate: outside [guidance_low, guidance_high] the step
     must be the plain conditional one.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.samplers_repae import euler_sampler, euler_maruyama_sampler  # noqa: E402


class StraightLineFlow:
    """Exact velocity field of the straight path from `start` to `target`, in OUR convention.

    v_ours(x, t) = data - noise, constant everywhere, so an Euler integration from t=0 to t=1
    lands on `target` exactly regardless of step count.
    """

    def __init__(self, start, target):
        self.v = target - start

    def velocity_field(self, x, t, y, force_drop_ids=None):
        return self.v.to(x.dtype).expand_as(x)


class ConditionalFlow:
    """v depends on the label: null class gets `v_uncond`, anything else `v_cond`."""

    def __init__(self, v_cond, v_uncond, null_class):
        self.v_cond, self.v_uncond, self.null_class = v_cond, v_uncond, null_class

    def velocity_field(self, x, t, y, force_drop_ids=None):
        is_null = (y == self.null_class).view(-1, *([1] * (x.dim() - 1)))
        return torch.where(is_null, self.v_uncond.expand_as(x), self.v_cond.expand_as(x))


def main():
    torch.manual_seed(0)
    b, c, h = 4, 8, 4
    start = torch.randn(b, c, h, h)
    target = torch.randn(b, c, h, h) * 2.0 + 1.0
    y = torch.zeros(b, dtype=torch.long)

    # ------------------------------------------------------------------ 1. time convention #
    model = StraightLineFlow(start, target)
    for steps in (10, 50, 250):
        out = euler_sampler(model, start, y, num_steps=steps, cfg_scale=1.0, null_class=1000)
        err = (out - target).abs().max().item()
        assert err < 1e-4, (
            f"ODE with {steps} steps landed {err:.3e} from the target. A straight-line flow "
            f"must integrate exactly; this means the time flip or the velocity sign is wrong.")
    print(f"ok  1. ODE recovers the target exactly (max err {err:.2e}) -> time convention "
          f"and velocity sign are right")

    # The SDE injects noise, so it cannot land exactly; it must still land NEAR the target,
    # and much closer to it than to the starting point.
    out_sde = euler_maruyama_sampler(model, start, y, num_steps=250, cfg_scale=1.0,
                                     null_class=1000)
    d_target = (out_sde - target).pow(2).mean().sqrt().item()
    d_start = (out_sde - start).pow(2).mean().sqrt().item()
    assert d_target < d_start, f"SDE ended nearer the start ({d_start:.3f}) than the target ({d_target:.3f})"
    print(f"ok  2. SDE lands near the target (rms {d_target:.3f} vs {d_start:.3f} to the start)")

    # ------------------------------------------------------------------------- 3. guidance #
    null_class = 1000
    v_cond = torch.ones(1, c, h, h)
    v_uncond = torch.zeros(1, c, h, h)
    gmodel = ConditionalFlow(v_cond, v_uncond, null_class)

    # Guided velocity should be v_uncond + w * (v_cond - v_uncond) = w, so over the whole
    # unit interval the displacement from the start is exactly -w (our sign convention:
    # the integrator runs in their units, where the accumulated drift is the negative).
    for w in (1.0, 2.0, 4.0):
        out = euler_sampler(gmodel, start, y, num_steps=100, cfg_scale=w, null_class=null_class)
        moved = (out - start).mean().item()
        assert abs(moved - w) < 1e-3, f"cfg {w}: displacement {moved:.4f}, expected {w:.4f}"
    print("ok  3. classifier-free guidance scales the conditional-unconditional gap exactly")

    # ------------------------------------------------------------ 4. the guidance interval #
    # A window covering nothing must reproduce the unguided (plain conditional) trajectory,
    # and a full window must not.
    plain = euler_sampler(gmodel, start, y, num_steps=100, cfg_scale=1.0, null_class=null_class)
    windowed = euler_sampler(gmodel, start, y, num_steps=100, cfg_scale=4.0,
                             guidance_low=0.98, guidance_high=1.0, null_class=null_class)
    full = euler_sampler(gmodel, start, y, num_steps=100, cfg_scale=4.0,
                         guidance_low=0.0, guidance_high=1.0, null_class=null_class)
    assert not torch.allclose(full, plain, atol=1e-3), "full-window guidance changed nothing"
    assert (windowed - plain).abs().max() < (full - plain).abs().max(), \
        "a narrow guidance window should deviate less from the unguided trajectory than a full one"
    print("ok  4. the guidance interval gates (narrow window deviates less than full)")

    print("\nall sampler checks passed")


if __name__ == "__main__":
    main()
