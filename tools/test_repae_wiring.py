"""Self-contained correctness test for the REPA-E gradient routing. CPU, seconds, no data.

    python tools/test_repae_wiring.py

WHY THIS EXISTS. End-to-end training has one property that must hold and that NOTHING else
reports when it breaks: the flow-matching loss must not reach the tokenizer. If the
stop-gradient is lost -- a missing .detach(), a refactor that reuses the attached latent -- the
run does not crash, the loss curves look fine, and the tokenizer quietly learns a latent that
is easy to denoise and bad to generate from (REPA-E Tab. 6: gFID 444 vs 16.3). By the time
gFID says so, the compute is spent.

The test mirrors experiments/train_repae.py's step exactly and asserts:
  1. the alignment pass DOES send gradients into the tokenizer;
  2. the SiT gradients from the alignment pass are DISCARDED before its own step, so that
     loss updates the SiT through neither path (the trainer zeroes them rather than toggling
     requires_grad, which on DDP-managed parameters can drop the reducer's hooks);
  3. the flow loss does NOT send gradients into the tokenizer (the stop-gradient);
  4. the flow loss DOES update the SiT;
  5. the alignment pass leaves the BatchNorm running statistics untouched (eval mode), while
     the diffusion pass moves them (train mode);
  6. encode_latent / decode_latent reproduce the original attention wiring exactly;
  7. both passes share one timestep and one noise draw.
"""

import os
import sys

import torch

# Runnable as `python tools/test_repae_wiring.py` from the repo root without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_vae import DUALVAE          # noqa: E402
from models.sit import build_sit             # noqa: E402


def _grad_norm(module):
    return sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)


def main():
    torch.manual_seed(0)
    # Tiny everything: 32px images -> 4x4 latent, 2 SiT blocks. The wiring is what is under
    # test, not the capacity.
    vae = DUALVAE(latent_channels=8, downsample_factor=8, quantizer="fsq", fsq_levels=[5] * 8,
                  rq_depth=1, residual_continuous=True, component_prior=True,
                  sigma2_floor=1e-4, sigma2_ceil=0.0625)
    sit = build_sit("SiT-B/2", input_size=4, in_channels=8, num_classes=10,
                    encoder_depth=1, z_dims=(16,), projector_dim=32)
    sit.blocks = sit.blocks[:2]          # 2 blocks: alignment at 1, full stack at 2
    images = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 10, (2,))
    zs = [torch.randn(2, 4, 16)]         # 2x2 latent tokens at patch 2 -> 4 tokens

    # ---------------------------------------------------------------- 6. wiring equivalence #
    vae.eval()
    with torch.no_grad():
        z_pre, _ = vae.encode_latent(images, sample=False)
        z_att = vae.encode_for_diffusion(images, noise=torch.zeros(2, 8, 4, 4))
        assert torch.equal(vae.attention(z_pre), z_att), "attention(z_pre) != encode_for_diffusion"
        assert torch.equal(vae.decode_latent(z_pre), vae.decoder(vae.attention(z_pre)))
    vae.train()
    print("ok  6. encode_latent/decode_latent reproduce the original wiring")

    # ------------------------------------------------------------------- the two-pass step #
    z, aux = vae.encode_latent(images)
    recon = vae.decode_latent(z)
    loss_recon = torch.nn.functional.mse_loss(recon, images)

    # Pre-load the SiT with a gradient, standing in for a previous micro-batch of an
    # accumulation group. The alignment pass must not disturb it -- that is precisely what
    # breaks if the surrogate below is ever replaced by a plain `loss += 1.5 * repa`.
    sit(x=z.detach(), y=labels, zs=zs, align_only=False)["denoising_loss"].mean().backward()
    sit_grads_before = {n: p.grad.clone() for n, p in sit.named_parameters() if p.grad is not None}
    assert sit_grads_before, "setup failed: the SiT has no gradients to protect"

    bn_before = sit.bn.running_var.clone()
    sit.eval()
    out_align = sit(x=z, y=labels, zs=zs, align_only=True)
    bn_after_align = sit.bn.running_var.clone()

    # The trainer's alignment path: differentiate w.r.t. the LATENT only, then re-enter the
    # tokenizer's graph through a surrogate with the same parameter gradients.
    grad_z = torch.autograd.grad(1.5 * out_align["proj_loss"], z)[0]
    (loss_recon + (z * grad_z.detach()).sum()).backward()

    vae_grad_after_align = _grad_norm(vae)
    assert vae_grad_after_align > 0, "tokenizer received no gradient from the alignment pass"
    assert torch.equal(bn_before, bn_after_align), \
        "the alignment pass moved the BatchNorm running statistics (should be eval mode)"
    print("ok  1. alignment pass feeds the tokenizer      (grad norm "
          f"{vae_grad_after_align:.4f})")
    print("ok  5a. alignment pass leaves BN running stats untouched")

    for n, g in sit_grads_before.items():
        assert torch.equal(dict(sit.named_parameters())[n].grad, g), \
            f"the alignment pass wrote into the SiT's gradients through {n}"
    print("ok  2. alignment pass leaves SiT gradients untouched (accumulation-safe)")

    # Snapshot the tokenizer's gradients, then run the diffusion pass on the DETACHED latent.
    # Nothing in the tokenizer may change as a result.
    vae_grads = {n: p.grad.clone() for n, p in vae.named_parameters() if p.grad is not None}

    sit.train()
    out_sit = sit(x=z.detach(), y=labels, zs=zs, align_only=False,
                  time_input=out_align["time_input"], noises=out_align["noises"])
    (out_sit["denoising_loss"].mean() + 0.5 * out_sit["proj_loss"]).backward()

    for n, g in vae_grads.items():
        now = dict(vae.named_parameters())[n].grad
        assert torch.equal(now, g), f"the flow loss reached the tokenizer through {n}"
    print("ok  3. flow loss does NOT reach the tokenizer  (stop-gradient holds)")

    assert _grad_norm(sit) > 0, "the SiT received no gradient from the flow loss"
    print(f"ok  4. flow loss updates the SiT               (grad norm {_grad_norm(sit):.4f})")
    assert not torch.equal(bn_after_align, sit.bn.running_var), \
        "the diffusion pass did not update the BatchNorm running statistics"
    print("ok  5b. diffusion pass updates BN running stats")

    # The shared interpolant: both passes must have seen the same t and the same noise, or the
    # two losses are computed at different points of the path.
    assert torch.equal(out_align["time_input"], out_sit["time_input"])
    assert torch.equal(out_align["noises"], out_sit["noises"])
    print("ok  7. both passes share the timestep and the noise")
    print("\nall wiring checks passed")


if __name__ == "__main__":
    main()
