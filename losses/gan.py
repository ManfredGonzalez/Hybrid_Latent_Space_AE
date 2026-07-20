"""VQGAN-style adversarial loss components (Esser et al. 2021, "Taming Transformers").

Opt-in via `use_gan: true` in the trainer configs. The recipe is the standard one used
by VQGAN / SD-VAE / RAE stage-1 decoders:

  * a PatchGAN discriminator (pix2pix NLayerDiscriminator) judging local realism,
  * hinge loss for the discriminator, -mean(D(fake)) for the generator,
  * VQGAN's adaptive generator weight: lambda = ||grad_L nll|| / ||grad_L g_loss||
    measured at the decoder's LAST layer, so the adversarial gradient never overwhelms
    the reconstruction gradient regardless of loss scales,
  * a warmup: the whole term is inactive before `gan_start_epoch`, letting
    reconstruction (and the codebook EMA) settle first.

Precision note: the discriminator runs in fp32 OUTSIDE the bf16 autocast region,
consistent with the project convention that all loss math is full precision. Gradients
still flow into the (autocast) generator graph through the fp32 cast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PatchDiscriminator(nn.Module):
    """pix2pix-style NLayerDiscriminator: outputs a map of per-patch realism logits.

    Judging patches (receptive field ~70px at n_layers=3) rather than whole images is
    what makes the loss a texture/detail critic instead of a global-content critic --
    exactly the role we want next to MSE/LPIPS.
    """

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        layers = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
                  nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),  # per-patch logits
        ]
        self.main = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        return self.main(x)


def hinge_d_loss(logits_real, logits_fake):
    """Hinge discriminator loss (taming-transformers default)."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def generator_g_loss(logits_fake):
    """Non-saturating hinge generator loss."""
    return -torch.mean(logits_fake)


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, max_weight=1e4, eps=1e-4):
    """VQGAN's lambda: balance adversarial vs reconstruction gradients at the decoder's
    last layer. Both grads are computed with retain_graph so the subsequent
    loss.backward() still works. Returned detached (it is a weight, not a loss path).
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + eps)
    return torch.clamp(d_weight, 0.0, max_weight).detach()


def build_gan(args, model, device):
    """Build the opt-in adversarial bundle from config args, or return None.

    Returns a dict consumed by the trainers:
      disc, opt (Adam betas 0.5/0.9), weight (gan_weight), start_epoch, adaptive,
      last_layer (decoder's final conv weight, for the adaptive lambda).
    """
    if not getattr(args, 'use_gan', False):
        return None
    disc = PatchDiscriminator(
        in_channels=3,
        ndf=getattr(args, 'gan_ndf', 64),
        n_layers=getattr(args, 'gan_layers', 3),
    ).to(device)
    return {
        'disc': disc,
        'opt': torch.optim.Adam(disc.parameters(), lr=getattr(args, 'gan_lr', args.lr), betas=(0.5, 0.9)),
        'weight': getattr(args, 'gan_weight', 0.5),
        'start_epoch': getattr(args, 'gan_start_epoch', 20),
        'adaptive': getattr(args, 'gan_adaptive_weight', True),
        'last_layer': model.decoder[-1].weight,
    }


def generator_step_terms(gan, epoch, recon, recon_loss):
    """G-side adversarial term for the current batch.

    Returns (extra_loss, g_loss_scalar, d_weight_scalar). extra_loss participates in
    the main backward; before start_epoch it is a zero tensor (term fully inactive).
    recon must still be attached to the generator graph; recon_loss is the fp32
    reconstruction (nll) loss used for the adaptive weight.
    """
    device = recon.device
    if gan is None or epoch < gan['start_epoch']:
        zero = torch.zeros((), device=device)
        return zero, 0.0, 0.0
    logits_fake = gan['disc'](recon.float())
    g_loss = generator_g_loss(logits_fake)
    if gan['adaptive']:
        d_weight = calculate_adaptive_weight(recon_loss, g_loss, gan['last_layer'])
    else:
        d_weight = torch.ones((), device=device)
    extra = gan['weight'] * d_weight * g_loss
    return extra, float(g_loss.detach()), float(d_weight)


def discriminator_step(gan, epoch, images, recon):
    """One discriminator update on (real, fake.detach()). Returns d_loss scalar (0.0
    when inactive). Call AFTER the generator's optimizer.step()."""
    if gan is None or epoch < gan['start_epoch']:
        return 0.0
    gan['opt'].zero_grad()
    logits_real = gan['disc'](images.detach().float())
    logits_fake = gan['disc'](recon.detach().float())
    d_loss = hinge_d_loss(logits_real, logits_fake)
    d_loss.backward()
    gan['opt'].step()
    return float(d_loss.detach())
