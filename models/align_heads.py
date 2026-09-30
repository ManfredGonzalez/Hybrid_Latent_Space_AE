"""Image-level alignment between the tokenizer's latent and DINOv2's [CLS] token.

WHY THIS EXISTS. REPA aligns PATCH tokens to PATCH tokens, which is local semantics. The mode
tests measured that image-level structure is what our latent lacks: DINOv2 [CLS] clusters
explain ~56% of the variance in [CLS] space and ~0% in our latent space, which is why a
mixture source (MM-FM) buys nothing on it. Making every 8x8 patch DINOv2-like does not
obviously make whole IMAGES cluster -- different classes share local appearance. This module
adds the term that targets the missing property directly.

TWO DESIGN CHOICES THAT DECIDE WHETHER IT WORKS:

1. THE HEAD IS DELIBERATELY WEAK. Give an alignment head enough capacity and it will learn to
   extract [CLS] information from a latent with no cluster structure at all -- satisfying the
   loss while changing nothing. That is already happening on the SiT side of this project: the
   projected cosine reads a healthy -0.61 while CKNNA on raw features is low. So the latent is
   pooled to a 4x4 grid (8 x 4 x 4 = 128 numbers) and mapped by a SINGLE linear layer. At rank
   128 the only way to satisfy the objective is for the latent's coarse spatial structure to
   carry the semantics.

2. THE LOSS IS CONTRASTIVE, NOT COSINE. Per-sample cosine similarity to one's own [CLS] is
   maximized just as well by every latent pointing near the MEAN [CLS] direction -- collapse,
   which is exactly the structureless latent we are trying to fix, reported as success.
   InfoNCE adds the negatives: this latent must look like ITS image's [CLS] and unlike the
   other images' in the batch. That is a direct instruction to spread images apart, which is
   the property the decision gate measures.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class LatentCLSHead(nn.Module):
    """(B, C, H, W) latent -> (B, out_dim), via coarse average pooling and ONE linear layer.

    Auxiliary: used only by the alignment loss, never by the decoder or the flow model, and
    discarded at inference. Its parameters belong to the tokenizer's optimizer, since its job
    is to make the tokenizer's latent legible rather than to model anything itself.
    """

    def __init__(self, in_channels, pool=4, out_dim=768):
        super().__init__()
        self.pool_size = pool
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.fc = nn.Linear(in_channels * pool * pool, out_dim)

    def forward(self, z):
        # The caller runs this OUTSIDE the autocast region (the alignment losses are computed
        # in fp32 alongside the reconstruction terms), so a bf16 latent would meet fp32 weights
        # and fail. Casting here rather than at the call site keeps the head usable from either
        # context; the cast is differentiable, so the gradient still reaches the encoder.
        z = z.to(self.fc.weight.dtype)
        return self.fc(self.pool(z).flatten(1))


def _gather_with_grad(x):
    """Concatenate across ranks, keeping the local slice differentiable.

    The CLIP/SimCLR pattern: all_gather itself is not differentiable, so the copies from other
    ranks act as constants (still correct negatives) and the local slice is spliced back in to
    preserve its gradient path. Without gathering, a 32-per-rank batch gives 31 negatives
    instead of 255, and contrastive objectives are strongly sensitive to that count.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x, 0
    world, rank = dist.get_world_size(), dist.get_rank()
    buf = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(buf, x.contiguous())
    buf[rank] = x
    return torch.cat(buf, dim=0), rank


def image_alignment_loss(head, z, cls_token, temperature=0.07, gather=True):
    """Symmetric InfoNCE between the pooled latent and the frozen [CLS] token.

    Returns (loss, top1_accuracy). The accuracy -- "given this latent, can you pick its own
    image's [CLS] out of the batch?" -- is the readable diagnostic: at chance it equals
    1/batch, and if it stays there the term is doing nothing no matter what the loss says.
    """
    p = F.normalize(head(z).float(), dim=-1)
    c = F.normalize(cls_token.float(), dim=-1)

    if gather:
        c_all, rank = _gather_with_grad(c)
        p_all, _ = _gather_with_grad(p)
    else:
        c_all, p_all, rank = c, p, 0

    b = p.shape[0]
    # Positives sit on the diagonal of the block this rank owns.
    labels = torch.arange(b, device=p.device) + rank * b

    logits_p = (p @ c_all.t()) / temperature          # latent -> CLS
    logits_c = (c @ p_all.t()) / temperature          # CLS -> latent
    loss = 0.5 * (F.cross_entropy(logits_p, labels) + F.cross_entropy(logits_c, labels))
    acc = (logits_p.argmax(dim=1) == labels).float().mean()
    return loss, acc
