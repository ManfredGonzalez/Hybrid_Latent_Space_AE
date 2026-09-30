"""Tests for the alignment arm: gradient ROUTING and the image-level term. CPU, seconds.

    python tools/test_align_routing.py

WHAT COULD SILENTLY NOT WORK, and therefore what is pinned here:

1. ROUTING. `repa_target: z_q` is supposed to send the patch-alignment gradient to the
   quantized branch only, leaving Delta to reconstruction and the KL. The mechanism is a
   one-tensor change in a surrogate term -- easy to get backwards, and a mistake produces a
   run that trains fine and answers a different question than the one asked. The test asserts
   the residual head receives EXACTLY ZERO gradient under 'z_q', and nonzero under 'z'.

2. THE CONTRASTIVE TERM. InfoNCE must actually discriminate. A sign error, a wrong label
   offset or a collapsed head all still produce a finite decreasing loss; what distinguishes
   them is retrieval accuracy. The test checks chance-level accuracy on random features and
   perfect accuracy when the head can trivially solve the task.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.align_heads import LatentCLSHead, image_alignment_loss   # noqa: E402
from models.dual_vae import DUALVAE                                   # noqa: E402


def _residual_head_grad_norm(vae):
    """Total gradient on the continuous branch's parameters (the Delta path)."""
    return sum(float(p.grad.norm()) for p in vae.vanilla_VAE_bottle_neck.parameters()
               if p.grad is not None)


def _encoder_grad_norm(vae):
    return sum(float(p.grad.norm()) for p in vae.encoder.parameters() if p.grad is not None)


def main():
    torch.manual_seed(0)
    vae = DUALVAE(latent_channels=8, downsample_factor=8, quantizer="fsq", fsq_levels=[5] * 8,
                  rq_depth=1, residual_continuous=True, component_prior=True,
                  sigma2_floor=1e-4, sigma2_ceil=0.0625)
    images = torch.randn(4, 3, 32, 32)

    # A stand-in "alignment loss" on the latent: any scalar function of z will do, since what
    # is under test is which branch the surrogate hands the gradient to.
    def run(target_name):
        vae.zero_grad(set_to_none=True)
        z, aux = vae.encode_latent(images)
        fake_align = (z ** 2).mean()
        grad_z = torch.autograd.grad(fake_align, z, retain_graph=True)[0]
        target = {"z": z, "z_q": aux["z_vq"], "delta": z - aux["z_vq"]}[target_name]
        (target * grad_z.detach()).sum().backward()
        return _residual_head_grad_norm(vae), _encoder_grad_norm(vae)

    d_zq, e_zq = run("z_q")
    assert d_zq == 0.0, f"'z_q' routing leaked gradient into the residual head ({d_zq})"
    assert e_zq > 0.0, "'z_q' routing sent no gradient to the encoder (STE/tanh path broken)"
    print(f"ok  1. target z_q  -> residual head {d_zq:.6f}, encoder {e_zq:.4f}")

    d_z, e_z = run("z")
    assert d_z > 0.0, "'z' routing should reach the residual head (reference behaviour)"
    print(f"ok  2. target z    -> residual head {d_z:.4f}, encoder {e_z:.4f}")

    d_d, _ = run("delta")
    assert d_d > 0.0, "'delta' routing should reach the residual head"
    print(f"ok  3. target delta-> residual head {d_d:.4f}")

    # ------------------------------------------------------------------- 4. contrastive term #
    head = LatentCLSHead(in_channels=8, pool=4, out_dim=64)
    z = torch.randn(16, 8, 8, 8)
    cls = torch.randn(16, 64)
    loss, acc = image_alignment_loss(head, z, cls, temperature=0.07, gather=False)
    assert torch.isfinite(loss), "InfoNCE produced a non-finite loss"
    assert acc < 0.4, f"untrained head scored {acc:.2f}; unrelated features must be near chance"
    print(f"ok  4. InfoNCE on unrelated features: loss {float(loss):.3f}, top1 {float(acc):.3f} "
          f"(chance {1/16:.3f})")

    # Solvable case: optimize the head alone on a fixed pair and check it can reach the target.
    head = LatentCLSHead(in_channels=8, pool=4, out_dim=64)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss, acc = image_alignment_loss(head, z, cls, temperature=0.07, gather=False)
        loss.backward()
        opt.step()
    assert acc > 0.9, f"the head could not solve a trivially solvable task (top1 {acc:.2f})"
    print(f"ok  5. InfoNCE is learnable: top1 {float(acc):.3f} after 300 steps")

    print("\nall alignment-routing checks passed")


if __name__ == "__main__":
    main()
