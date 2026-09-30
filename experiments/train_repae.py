"""REPA-E: end-to-end training of the DualVAE tokenizer together with a SiT flow model.

Phase 1 of reports/mmfm_repae_design.pdf. The recipe is Leng et al. (ICCV 2025), adapted to
this repo's tokenizer; the per-step structure below was verified against their public
implementation rather than inferred from the paper.

THE ONE THING THAT MUST NOT BE GOT WRONG. The diffusion (flow-matching) loss must NOT reach the
tokenizer. Left connected, it hacks the latent into something easy to denoise -- lower spatial
variance, simpler structure -- and generation quality collapses (their Table 6: gFID 444 vs
16.3). The alignment (REPA) loss is what may reach the tokenizer.

Because both losses come from the same network, the reference runs the SiT TWICE per step:

  pass A (alignment)  x = z          attached     SiT frozen + eval    first `encoder_depth`
                                                                       blocks only, then break
  pass B (diffusion)  x = z.detach() detached     SiT trainable        full stack

Pass A is what sends REPA gradients into the tokenizer; freezing the SiT there stops the same
loss from updating it twice with two different coefficients, and eval mode stops that pass from
touching the BatchNorm running statistics. The timestep and the noise are drawn in pass A and
REUSED in pass B, so both losses see the identical interpolant.

Order within a step: tokenizer (recon + LPIPS + GAN + code-centered KL + 1.5*REPA) ->
discriminator -> SiT (flow + 0.5*REPA) -> SiT EMA.

NO LATENT CACHE. The tokenizer moves every step, so latents are computed online. Caching stays
correct for the frozen-tokenizer baselines (experiments/train_latent_flow.py) and for
diagnostics, not here.
"""

import contextlib
import os
import time
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from experiments.train_dualvae import (prepare_data, build_recon_criterion,
                                       codebook_health_metrics, initialize_model)
from experiments.train_latent_flow import ModelEMA
from losses.gan import build_gan, generator_step_terms, discriminator_step
from losses.loss import dualvae_loss
from models.align_heads import LatentCLSHead, image_alignment_loss
from models.mode_tracker import ModeTracker
from models.repr_encoder import DinoV2Features
from models.sit import build_sit
from tools.distributed import (ddp_setup, ddp_cleanup, is_dist, is_main_process, world_size,
                               all_reduce_metrics)
from tools.latent_ae import load_frozen_ae
from tools.samplers_repae import euler_sampler
from tools.utils import create_directory


def _maybe_no_sync(module, is_boundary):
    """Suppress DDP's gradient all-reduce on non-final micro-batches of an accumulation group.

    DDP reduces the .grad buffers at the end of every backward it is allowed to sync on. Under
    accumulation those buffers already hold the sum over the group, so reducing once at the
    boundary is both correct and (accum - 1) fewer collectives per optimizer step. A no-op for
    a plain module or when accumulation is off.
    """
    if is_boundary or not hasattr(module, "no_sync"):
        return contextlib.nullcontext()
    return module.no_sync()


def amp_context(device, enabled):
    """bf16 autocast on CUDA, a no-op elsewhere.

    The CPU path exists so tools/test_repae_wiring.py and a laptop-scale smoke run can
    exercise this file end to end; bf16 autocast on CPU is slower than fp32, not faster.
    """
    device_type = "cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                          enabled=bool(enabled) and device_type == "cuda")


# --------------------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------------------- #

class E2ETokenizer(nn.Module):
    """Thin wrapper whose forward() is the encode/decode pair the end-to-end step needs.

    It exists for one reason: DistributedDataParallel synchronizes gradients by hooking the
    module's forward. Calling `dualvae.encode_latent(...)` on the unwrapped model would train
    each rank on its own shard and never average the tokenizer's gradients, which fails
    silently -- the run proceeds and the ranks slowly diverge.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, images, sample=True):
        z, aux = self.vae.encode_latent(images, sample=sample)
        return z, self.vae.decode_latent(z), aux


def build_tokenizer(args, device):
    """The DualVAE to tune: loaded from a run directory, or built fresh when none is given.

    FROM A CHECKPOINT (`ae_checkpoint` set). Reuses the frozen loader so the architecture flags
    (quantizer, fsq_levels, rq_depth, the residual head's width) come from the checkpoint's own
    config_used.yaml. Rebuilding them from the end-to-end config would eventually drift and
    fail on a state_dict key.

    FROM SCRATCH (`ae_checkpoint` unset). The tokenizer is built from THIS config's fields and
    trained jointly from random initialization. REPA-E supports this: their Tab. 7 reports
    gFID 4.34 from scratch vs 4.07 from a pretrained VAE at 80 epochs, both well ahead of
    vanilla REPA's 7.90 -- so the pretrained start is worth a little, not a lot.

    Two consequences worth knowing, neither of which is an error:
      * the latent moves fast early, so the BatchNorm warm-start matters less (it tracks) and
        the mode tracker's latent statistics should be reset once the tokenizer settles --
        see `mode_stats_reset_epoch`;
      * there is no before/after baseline for the decision gate. The mode tests in
        reports/mmfm_mode_tests/ were measured on the FSQ 0.1 checkpoint, so a scratch run's
        T can be compared against THAT number only as "a different tokenizer", not as
        "the same tokenizer, tuned".

    FSQ makes the scratch path simpler than VQ would: the grid is fixed, so there is no
    codebook to k-means-seed from data before training starts.
    """
    if getattr(args, "ae_checkpoint", None):
        ae = load_frozen_ae(args.ae_checkpoint, device, getattr(args, "ae_config", None))
        if ae.kind != "dualvae":
            raise ValueError(f"train_repae expects a dualvae checkpoint, got {ae.kind!r}.")
        model = ae.model                   # the loader froze it; end-to-end training unfreezes
        model.requires_grad_(True)
        model.train()
        return model, ae.latent_channels, ae.downsample_factor

    # initialize_model owns the full kwargs block; calling it keeps this path from drifting
    # out of sync with the standalone tokenizer trainer. Its optimizer is discarded -- this
    # trainer builds its own (AdamW, and the parameters also need the SiT's optimizer split).
    model, _ = initialize_model(args)
    model.train()
    if is_main_process():
        print(f"[AE] built dualvae FROM SCRATCH (quantizer={getattr(args, 'quantizer', 'vq')}, "
              f"latent_channels={args.latent_channels}, "
              f"downsample_factor={getattr(args, 'downsample_factor', 8)})")
    return model, args.latent_channels, getattr(args, "downsample_factor", 8)


def build_generator(args, latent_size, latent_channels, repr_dim, device):
    sit = build_sit(
        getattr(args, "sit_model", "SiT-B/2"),
        input_size=latent_size,
        in_channels=latent_channels,
        encoder_depth=getattr(args, "encoder_depth", 8),
        num_classes=args.num_classes,
        class_dropout_prob=getattr(args, "cfg_dropout_prob", 0.1),
        z_dims=(repr_dim,),
        projector_dim=getattr(args, "projector_dim", 2048),
        bn_momentum=getattr(args, "bn_momentum", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in sit.parameters() if p.requires_grad)
    if is_main_process():
        print(f"[sit] {getattr(args, 'sit_model', 'SiT-B/2')} "
              f"({n_params / 1e6:.1f}M params, {latent_size // sit.patch_size}^2 tokens)")
    return sit


@torch.no_grad()
def estimate_latent_stats(tokenizer, loader, device, n_batches, use_amp):
    """Per-channel mean/std of the PRE-ATTENTION latent, for warm-starting the SiT's BatchNorm.

    Not strictly required (BN converges in ~30 steps at momentum 0.1), but the alignment pass
    runs in eval mode and therefore uses the RUNNING statistics from step 0 -- before they
    converge the tokenizer would receive REPA gradients computed on badly scaled latents.

    Deliberately estimated here rather than read from the mode-test results: those statistics
    were measured on the POST-attention latent, which is a different space (48% relative).
    """
    tokenizer.eval()
    n, s, s2 = 0, None, None
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        with amp_context(device, use_amp):
            z, _ = tokenizer.encode_latent(images, sample=False)
        z = z.float()
        b = z.shape[0] * z.shape[2] * z.shape[3]
        chan_sum = z.sum(dim=(0, 2, 3))
        chan_sq = (z ** 2).sum(dim=(0, 2, 3))
        s = chan_sum if s is None else s + chan_sum
        s2 = chan_sq if s2 is None else s2 + chan_sq
        n += b
    counts = torch.tensor([float(n)], device=device)
    all_reduce_metrics_sum_(s, s2, counts)
    mean = s / counts
    std = (s2 / counts - mean ** 2).clamp(min=1e-8).sqrt()
    tokenizer.train()
    return mean.cpu(), std.cpu()


def all_reduce_metrics_sum_(*tensors):
    """Sum tensors across ranks in place (no-op single process)."""
    import torch.distributed as dist
    if is_dist():
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


# --------------------------------------------------------------------------------------- #
# one step
# --------------------------------------------------------------------------------------- #

def train_one_epoch(state, epoch):
    """One pass over the training loader. Returns a dict of epoch-mean metrics."""
    args, device = state["args"], state["device"]
    tokenizer, sit = state["tokenizer"], state["sit"]
    vae, sit_raw = state["vae"], state["sit_raw"]
    repr_encoder, modes, gan = state["repr_encoder"], state["modes"], state["gan"]
    opt_vae, opt_sit = state["opt_vae"], state["opt_sit"]
    recon_criterion, ema = state["recon_criterion"], state["ema"]
    align_head = state.get("align_head")
    balance_every = max(1, int(getattr(args, "align_balance_every", 50)))
    loader = state["trainloader"]
    use_amp = args.use_amp

    tokenizer.train()
    sit.train()
    running = {k: 0.0 for k in ("loss_vae", "recon", "kl", "vq", "repa_vae", "flow",
                                "repa_sit", "gan_g", "gan_d", "gan_w",
                                "align/lam_patch", "align/lam_img", "align/img_loss",
                                "align/img_top1", "align/tanh_grad_scale")}
    mode_logs = {}
    limit = getattr(args, "limit_train_batches", 0) or len(loader)
    n_steps = min(len(loader), limit)
    # Gradient accumulation exists so the GLOBAL batch can be held at the reference recipe's
    # value when per-GPU memory cannot reach it: global = batch_size x world_size x accum.
    # With accum > 1, `step` counts MICRO-batches and the optimizers move every `accum` of them,
    # so the optimizer-step count matches a run that used the full batch directly.
    accum = max(1, int(getattr(args, "accum_steps", 1)))

    with tqdm(total=n_steps * args.batch_size * world_size(), desc=f"Epoch {epoch}/{args.epochs}",
              unit="img", disable=not is_main_process()) as pbar:
        for step, batch in enumerate(loader):
            if step >= n_steps:
                break
            # The last micro-batch of an accumulation group: gradients are synchronized across
            # ranks and the optimizers step. On the others DDP's all-reduce is suppressed
            # (no_sync), which is the entire point -- one reduction per optimizer step, not one
            # per micro-batch.
            is_boundary = ((step + 1) % accum == 0) or (step + 1 == n_steps)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)

            # 0. Frozen representation features. One forward serves REPA (patch tokens) and
            #    the mode tracker ([CLS]) -- the reason image-level modes cost nothing here.
            with amp_context(device, use_amp):
                patch_tokens, cls_token = repr_encoder(images)

            # 1. Tokenizer forward, through the DDP wrapper. z is the PRE-ATTENTION latent;
            #    decode_latent puts attention back on the decoder side, so the reconstruction
            #    is bit-identical to what forward() would have produced.
            with _maybe_no_sync(tokenizer, is_boundary):
                with amp_context(device, use_amp):
                    z, recon, aux = tokenizer(images)

                loss_vae, recon_loss, vq_loss, kl_loss, _, _ = dualvae_loss(
                    recon.float(), images.float(), aux["vq_loss"].float(), args.kl_beta,
                    aux["mean"].float(), aux["log_variance"].float(),
                    reduction="sum", recon_criterion=recon_criterion,
                    prior_var=aux["prior_var"], prior_mean=aux["prior_mean"])

                gan_extra, g_loss_val, d_weight_val = generator_step_terms(gan, epoch, recon,
                                                                           recon_loss)
                loss_vae = loss_vae + gan_extra

                # 2. Alignment pass: the UNWRAPPED SiT in eval mode, latent ATTACHED. This is
                #    the only path by which the REPA loss reaches the tokenizer.
                #    eval() -> the BatchNorm running statistics are not moved by a pass the
                #    tokenizer drives, and normalization uses running stats, so the gradient
                #    the tokenizer receives does not depend on this batch's own statistics.
                sit_raw.eval()
                with amp_context(device, use_amp):
                    out_align = sit_raw(x=z, y=labels, zs=[patch_tokens], align_only=True,
                                        t_sampling=args.t_sampling,
                                        logit_normal_mean=getattr(args, "logit_normal_mean", 0.0),
                                        logit_normal_std=getattr(args, "logit_normal_std", 1.0))
                repa_vae = out_align["proj_loss"].float()

                # DIFFERENTIATE THE ALIGNMENT LOSS W.R.T. THE LATENT ONLY, then re-enter the
                # tokenizer's graph through a surrogate. Three things this buys, all of which
                # the obvious `loss_vae += 1.5 * repa; loss_vae.backward()` gets wrong:
                #   * autograd.grad never accumulates into .grad, so the SiT's gradients are
                #     untouched -- no need to freeze it (which risks dropping DDP's hooks) and
                #     no need to zero its gradients afterwards (which would destroy whatever
                #     accumulation had built up). This is what makes accum > 1 possible.
                #   * no parameter gradients are computed for the SiT's first `encoder_depth`
                #     blocks, which is pure saved work.
                #   * the surrogate `(z * dL/dz).sum()` has, by the chain rule, exactly the
                #     gradient w.r.t. the tokenizer's parameters that the real term would --
                #     and it flows through the DDP-wrapped forward, so it is synchronized.
                # The UNIT gradient (no coefficient): the adaptive weight below is computed
                # from its norm, so the coefficient has to be applied afterwards.
                grad_z_unit = torch.autograd.grad(repa_vae, z, retain_graph=True)[0]

                # --- image-level alignment (optional) ---------------------------------------
                # Patch-level REPA is LOCAL semantics. The mode tests measured that what our
                # latent lacks is IMAGE-level structure -- [CLS] clusters explain ~0% of its
                # variance -- and that is the property a mixture source needs. This term goes
                # after it directly. See models/align_heads.py for why it is contrastive and
                # why the head is deliberately weak.
                img_loss = torch.zeros((), device=device)
                img_acc = 0.0
                if align_head is not None:
                    img_loss, acc = image_alignment_loss(
                        align_head, aux["z_vq"], cls_token,
                        temperature=getattr(args, "img_align_temp", 0.07))
                    img_acc = float(acc.detach())

                # --- adaptive balancing ------------------------------------------------------
                # The alignment terms and the reconstruction term are not comparable as LOSSES:
                # reconstruction is sum-reduced over 3x256x256 (~1e5) while REPA is a cosine
                # similarity (~0.4). The reference recipe's VAE loss is mean-reduced and O(1),
                # so its fixed 1.5 competes; ours does not, which is why the first run's
                # end-to-end tuning was effectively nominal. Gradients ARE comparable, and both
                # are available at the latent. Recomputed every `align_balance_every` steps
                # (each one costs an extra backward) and EMA-smoothed in between.
                if getattr(args, "align_adaptive", True) and step % balance_every == 0:
                    g_recon = torch.autograd.grad(recon_loss, z, retain_graph=True)[0].norm()
                    lam_p = (args.align_target_ratio * g_recon
                             / grad_z_unit.norm().clamp(min=1e-12)).clamp(max=1e6).detach()
                    state["lam_p"] = lam_p if state.get("lam_p") is None else \
                        0.99 * state["lam_p"] + 0.01 * lam_p
                    if align_head is not None:
                        g_img = torch.autograd.grad(img_loss, aux["z_vq"],
                                                    retain_graph=True)[0].norm()
                        lam_i = (args.align_target_ratio * g_recon
                                 / g_img.clamp(min=1e-12)).clamp(max=1e6).detach()
                        state["lam_i"] = lam_i if state.get("lam_i") is None else \
                            0.99 * state["lam_i"] + 0.01 * lam_i
                lam_p = state.get("lam_p", torch.tensor(args.repa_coeff_vae, device=device))
                lam_i = state.get("lam_i", torch.tensor(1.0, device=device))

                # ROUTE the patch alignment to one branch. z = z_q + Delta is a SUM, so
                # dL/dz_q == dL/dDelta == dL/dz and choosing the recipient is just choosing
                # which tensor the surrogate multiplies. 'z_q' sends the semantic pressure to
                # the codes and leaves Delta to reconstruction and the KL, which is the
                # division of labour the code-centred mixture assumes; 'z' (the reference
                # behaviour) splits it and pushes semantics into the detail branch.
                target = {"z": z, "z_q": aux["z_vq"],
                          "delta": z - aux["z_vq"]}[getattr(args, "repa_target", "z")]
                # .float() on both: this sums batch x 8 x 32 x 32 ~ 260k products, and bf16
                # has an 8-bit mantissa -- accumulating that many terms in it loses real
                # precision in the one quantity that carries the alignment signal.
                loss_vae = loss_vae + lam_p * (target.float()
                                               * grad_z_unit.detach().float()).sum()
                if align_head is not None:
                    loss_vae = loss_vae + lam_i * img_loss

                (loss_vae / accum).backward()

            if is_boundary:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
                opt_vae.step()
                opt_vae.zero_grad(set_to_none=True)

            # 3. Discriminator (its own optimizer, gradients averaged by hand -- see
            #    losses/gan.py). Stepped only on accumulation boundaries, so the generator and
            #    the critic keep the 1:1 update ratio they have when accum == 1.
            d_loss_val = discriminator_step(gan, epoch, images, recon) if is_boundary else 0.0

            # 4. Diffusion pass: latent DETACHED. This detach IS the stop-gradient that keeps
            #    the flow loss out of the tokenizer.
            sit_raw.train()
            with _maybe_no_sync(sit, is_boundary):
                with amp_context(device, use_amp):
                    out_sit = sit(x=z.detach(), y=labels, zs=[patch_tokens], align_only=False,
                                  time_input=out_align["time_input"], noises=out_align["noises"])
                flow_loss = out_sit["denoising_loss"].mean().float()
                repa_sit = out_sit["proj_loss"].float()
                loss_sit = flow_loss + args.repa_coeff_sit * repa_sit
                (loss_sit / accum).backward()

            if is_boundary:
                torch.nn.utils.clip_grad_norm_(sit_raw.parameters(), args.grad_clip)
                opt_sit.step()
                opt_sit.zero_grad(set_to_none=True)
                ema.update(sit_raw)

            # 5. Mode tracker: pure bookkeeping for Phase 2, no gradient, detached latent
            #    normalized with the CURRENT BatchNorm statistics (the space Phase 2 samples in).
            if modes is not None:
                with torch.no_grad():
                    if not bool(modes.initialized):
                        modes.observe_for_init(cls_token)
                    else:
                        z_norm = sit_raw.normalize_latent(z.detach().float())
                        mode_logs = modes.update(cls_token, z_norm, labels=labels,
                                                 batch_size_hint=args.batch_size * world_size())

            # dualvae_loss already divides by the batch, so these are per-sample already.
            running["loss_vae"] += float(loss_vae.detach())
            running["recon"] += float(recon_loss.detach())
            running["kl"] += float(kl_loss.detach())
            running["vq"] += float(vq_loss.detach())
            running["repa_vae"] += float(repa_vae.detach())
            running["flow"] += float(flow_loss.detach())
            running["repa_sit"] += float(repa_sit.detach())
            running["gan_g"] += g_loss_val
            running["gan_d"] += d_loss_val
            running["gan_w"] += d_weight_val
            running["align/lam_patch"] += float(lam_p)
            running["align/img_loss"] += float(img_loss.detach())
            # Top-1 retrieval: can this latent pick its own image's [CLS] out of the batch?
            # Chance is 1/(batch*world). If it sits there, the image term is inert whatever
            # the loss reads.
            running["align/img_top1"] += img_acc
            if align_head is not None:
                running["align/lam_img"] += float(lam_i)
            # Does the alignment gradient actually REACH the encoder? It arrives through FSQ's
            # straight-through estimator and the tanh bound; where tanh saturates its
            # derivative vanishes and the term is silently inert exactly where the encoder is
            # most confident. This is the mean of d(tanh)/dx over the pre-quantization values.
            running["align/tanh_grad_scale"] += float(
                (1.0 - torch.tanh(aux["z_e_vq"].detach().float()).pow(2)).mean())

            # global_step counts OPTIMIZER steps, not micro-batches, so that the step-keyed
            # intervals below mean the same thing at any accum_steps -- and so that our step
            # numbers line up with REPA-E's reported 100k / 200k / 400k marks.
            if is_boundary:
                state["global_step"] += 1
                gs = state["global_step"]
                every_sample = int(getattr(args, "sampling_steps", 0) or 0)
                if every_sample and (gs % every_sample == 0 or gs == 1):
                    log_sample_grid(state, gs)
                every_ckpt = int(getattr(args, "checkpointing_steps", 0) or 0)
                if every_ckpt and gs % every_ckpt == 0 and is_main_process():
                    save_checkpoint(state, epoch,
                                    os.path.join(args.checkpoints, f"step_{gs:07d}.pt"))
            pbar.update(images.shape[0] * world_size())
            pbar.set_postfix(flow=f"{running['flow'] / (step + 1):.4f}",
                             repa=f"{running['repa_sit'] / (step + 1):.3f}")

    metrics = {k: v / max(1, n_steps) for k, v in running.items()}
    metrics.update(mode_logs)
    return all_reduce_metrics(metrics, device)


@torch.no_grad()
def log_sample_grid(state, global_step):
    """Sample from the EMA SiT, decode, and log one grid to wandb. Rank 0 only.

    This is the only signal that GENERATION is progressing. The flow loss is a regression
    error against a per-sample velocity target and barely moves once training is underway; it
    can look healthy while samples are noise. REPA-E logs the same thing every 10k steps and
    computes no metrics in training at all.

    Deliberately the deterministic ODE sampler at a low step count: this is a picture, not a
    number. The SDE sampler at 250 steps that their tables use lives in
    tools/samplers_repae.py and is for the offline evaluation, where it is worth the cost.

    Only rank 0 works here. The other ranks simply arrive at the next backward's all-reduce
    and wait, which is safe because nothing in this function is a collective -- the EMA copy
    is a plain module and its SyncBatchNorm runs on running statistics in eval mode.
    """
    if not is_main_process() or not getattr(state["args"], "do_wandb", False):
        return
    args, device = state["args"], state["device"]
    sit_ema, vae = state["ema"].ema, state["vae"]
    n = int(getattr(args, "sample_grid_size", 8))

    was_training = vae.training
    vae.eval()
    sit_ema.eval()
    try:
        mean, std = sit_ema.latent_stats()
        shape = (n, sit_ema.in_channels, args.resize_img // 8, args.resize_img // 8)
        generator = torch.Generator(device=device).manual_seed(int(getattr(args, "seed", 42)))
        # A FIXED class list and a FIXED seed, so consecutive grids are comparable: what
        # changes between them is the model, not the prompt or the noise.
        y = torch.arange(n, device=device) % args.num_classes
        noise = torch.randn(shape, device=device, generator=generator)

        z_norm = euler_sampler(sit_ema, noise, y,
                               num_steps=int(getattr(args, "sample_steps", 50)),
                               cfg_scale=float(getattr(args, "sample_cfg_scale", 1.5)),
                               guidance_low=float(getattr(args, "guidance_low", 0.0)),
                               guidance_high=float(getattr(args, "guidance_high", 1.0)),
                               null_class=args.num_classes)
        # The sampler works in the normalized space the SiT was trained on; the decoder needs
        # the raw latent back, which is exactly what the BatchNorm statistics carry.
        z = z_norm * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        images = vae.decode_latent(z.to(next(vae.parameters()).dtype))
        images = ((images.float() + 1.0) / 2.0).clamp(0, 1)

        import wandb
        from torchvision.utils import make_grid
        grid = make_grid(images.cpu(), nrow=int(n ** 0.5) or 1)
        # Keyed by OPTIMIZER step, the same x-axis the epoch metrics use (and the one REPA-E
        # plots against). Keying one by epoch and the other by step makes wandb drop whichever
        # arrives with a smaller step.
        wandb.log({"samples": wandb.Image(grid)}, step=global_step)
    except Exception as exc:                      # never let a picture kill a multi-day run
        print(f"[sample] skipped at step {global_step}: {type(exc).__name__}: {exc}")
    finally:
        sit_ema.eval()
        if was_training:
            vae.train()


@torch.no_grad()
def validate(state, epoch):
    """Reconstruction quality and flow loss on the validation split.

    Deliberately NOT rFID/gFID: those have their own protocols in tools/rfid_imagenet.py and
    sample_latent_flow.py, and running them inline would make every epoch cost an evaluation.
    What this catches is the failure mode end-to-end training actually has -- the tokenizer
    silently degrading while the flow loss improves.
    """
    args, device = state["args"], state["device"]
    vae, sit_raw = state["vae"], state["sit_raw"]
    recon_criterion = state["recon_criterion"]
    vae.eval()
    sit_raw.eval()
    totals = {"val_recon": 0.0, "val_flow": 0.0, "val_latent_std": 0.0}
    n = 0
    max_steps = getattr(args, "limit_val_batches", 0) or len(state["valloader"])
    for step, batch in enumerate(state["valloader"]):
        if step >= max_steps:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        with amp_context(device, args.use_amp):
            patch_tokens, _ = state["repr_encoder"](images)
            z, aux = vae.encode_latent(images, sample=False)
            recon = vae.decode_latent(z)
            out = sit_raw(x=z, y=labels, zs=[patch_tokens], align_only=False)
        # ReconstructionCriterion returns (total, pixel, perceptual), all MEAN-reduced.
        totals["val_recon"] += float(recon_criterion(recon.float(), images.float())[0])
        totals["val_flow"] += float(out["denoising_loss"].mean())
        # The collapse signature from REPA-E's analysis: latent variance falling over time
        # means the tokenizer is simplifying the space to make denoising easy.
        totals["val_latent_std"] += float(z.float().std())
        n += 1
    vae.train()
    sit_raw.train()
    metrics = {k: v / max(1, n) for k, v in totals.items()}
    return all_reduce_metrics(metrics, device)


# --------------------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------------------- #

def save_checkpoint(state, epoch, path):
    """One file holds everything needed to resume or to export the tuned tokenizer.

    The mode tracker's buffers ride along: they ARE the fitted mixture, and re-deriving them
    on resume would silently redefine every mode.
    """
    payload = {
        "epoch": epoch,
        "global_step": state["global_step"],
        # The tokenizer is saved UNWRAPPED, so the file is a drop-in DualVAE checkpoint that
        # tools/latent_ae.py and the frozen-AE flow trainers can load unchanged.
        "tokenizer": state["vae"].state_dict(),
        "sit": state["sit_raw"].state_dict(),
        "sit_ema": state["ema"].state_dict(),
        # The alignment head is not needed for evaluation, but it IS needed to resume: a
        # freshly initialized head would restart the contrastive term from chance and jolt
        # the tokenizer.
        "align_head": (state["align_head"].state_dict()
                       if state.get("align_head") is not None else None),
        "opt_vae": state["opt_vae"].state_dict(),
        "opt_sit": state["opt_sit"].state_dict(),
        "args": vars(state["args"]),
    }
    if state["modes"] is not None:
        payload["modes"] = state["modes"].state_dict()
    if state["gan"] is not None:
        payload["gan_disc"] = state["gan"]["disc"].state_dict()
        payload["gan_opt"] = state["gan"]["opt"].state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic: a preempted job never leaves a half-written file


def load_checkpoint(state, path):
    ckpt = torch.load(path, map_location="cpu")
    state["vae"].load_state_dict(ckpt["tokenizer"])
    if state.get("align_head") is not None and ckpt.get("align_head") is not None:
        state["align_head"].load_state_dict(ckpt["align_head"])
    state["sit_raw"].load_state_dict(ckpt["sit"])
    state["ema"].load_state_dict(ckpt["sit_ema"])
    state["opt_vae"].load_state_dict(ckpt["opt_vae"])
    state["opt_sit"].load_state_dict(ckpt["opt_sit"])
    if state["modes"] is not None and "modes" in ckpt:
        state["modes"].load_state_dict(ckpt["modes"])
    if state["gan"] is not None and "gan_disc" in ckpt:
        state["gan"]["disc"].load_state_dict(ckpt["gan_disc"])
        state["gan"]["opt"].load_state_dict(ckpt["gan_opt"])
    state["global_step"] = ckpt.get("global_step", 0)
    return ckpt.get("epoch", -1) + 1


# --------------------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------------------- #

def train_repae(args):
    device, local_rank, rank, world = ddp_setup()
    args.device = str(device)
    torch.manual_seed(args.seed + rank)
    if getattr(args, "allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = getattr(args, "cudnn_benchmark", True)

    trainset, valset, _, trainloader, valloader, _ = prepare_data(args)
    num_classes = len(getattr(trainset, "classes", [])) or getattr(args, "num_classes", 1000)
    args.num_classes = num_classes

    vae, latent_channels, downsample = build_tokenizer(args, device)
    latent_size = args.resize_img // downsample
    repr_encoder = DinoV2Features(getattr(args, "repr_encoder", "dinov2_vitb14"),
                                  getattr(args, "repr_image_size", 224)).to(device)
    sit_raw = build_generator(args, latent_size, latent_channels, repr_encoder.embed_dim, device)
    # REPA compares the two token grids position by position, so they must have the same
    # shape. Checked against the built model rather than assuming patch size 2, and checked
    # BEFORE training rather than discovering it as a broadcast error mid-epoch.
    if repr_encoder.grid_size != sit_raw.x_embedder.grid_size:
        raise ValueError(
            f"token grids disagree: DINOv2 at {repr_encoder.image_size}px gives "
            f"{repr_encoder.grid_size}^2 tokens, the {latent_size}^2 latent at SiT patch "
            f"{sit_raw.patch_size} gives {sit_raw.x_embedder.grid_size}^2. Adjust "
            f"repr_image_size (a multiple of 14) or the SiT variant so they match.")

    # Warm-start the BN before the first alignment pass (which reads running statistics).
    stats_path = os.path.join(args.checkpoints, "latent_stats.pt")
    if os.path.exists(stats_path):
        st = torch.load(stats_path, map_location="cpu")
        mean, std = st["mean"], st["std"]
    else:
        mean, std = estimate_latent_stats(vae, trainloader, device,
                                          getattr(args, "latent_stats_batches", 40), args.use_amp)
        if is_main_process():
            create_directory(args.checkpoints)
            torch.save({"mean": mean, "std": std}, stats_path)
    sit_raw.init_bn(mean, std)
    if is_main_process():
        print(f"[bn] initialized from latent stats: mean {mean.tolist()}, std {std.tolist()}")

    modes = None
    if getattr(args, "track_modes", True):
        modes = ModeTracker(
            num_modes=getattr(args, "num_modes", 4096),
            descriptor_dim=repr_encoder.embed_dim,
            latent_shape=(latent_channels, latent_size, latent_size),
            decay=getattr(args, "mode_decay", 0.99),
            dead_threshold_frac=getattr(args, "mode_dead_threshold_frac", 0.1),
            stats_decay=getattr(args, "mode_stats_decay", None),
            var_shrinkage=getattr(args, "mode_var_shrinkage", 10.0),
            num_classes=num_classes,
        ).to(device)

    # The GAN bundle reaches into the decoder's last layer for the adaptive weight, so it is
    # built from the UNWRAPPED tokenizer, before any DDP wrapping.
    gan = build_gan(args, vae, device)

    # Image-level alignment head. Trained WITH the tokenizer (its job is to make the latent
    # legible, not to model anything), so its parameters join opt_vae below. Deliberately tiny
    # -- see models/align_heads.py on why capacity here defeats the purpose.
    align_head = None
    if getattr(args, "img_align", False):
        align_head = LatentCLSHead(latent_channels,
                                   pool=getattr(args, "img_align_pool", 4),
                                   out_dim=repr_encoder.embed_dim).to(device)
        if is_main_process():
            n_head = sum(p.numel() for p in align_head.parameters())
            print(f"[align] image-level head: pool {getattr(args, 'img_align_pool', 4)}^2 -> "
                  f"{repr_encoder.embed_dim} ({n_head / 1e6:.2f}M params), "
                  f"patch target = {getattr(args, 'repa_target', 'z')}")

    tokenizer = E2ETokenizer(vae).to(device)
    if is_dist():
        # SyncBatchNorm so the latent normalization uses global batch statistics; with 4 ranks
        # a per-rank BN would normalize with a quarter of the batch and the ranks would
        # disagree about the space the SiT sees.
        sit_raw = nn.SyncBatchNorm.convert_sync_batchnorm(sit_raw)
        tokenizer = DDP(tokenizer, device_ids=[local_rank])
        sit = DDP(sit_raw, device_ids=[local_rank])
        vae, sit_raw = tokenizer.module.vae, sit.module
    else:
        sit = sit_raw

    state = {
        "args": args, "device": device,
        "tokenizer": tokenizer, "vae": vae,
        "sit": sit, "sit_raw": sit_raw,
        "repr_encoder": repr_encoder, "modes": modes,
        "recon_criterion": build_recon_criterion(args),
        "gan": gan,
        "align_head": align_head,
        "opt_vae": torch.optim.AdamW(list(vae.parameters())
                                     + (list(align_head.parameters()) if align_head else []),
                                     lr=args.vae_lr,
                                     betas=(0.9, getattr(args, "adam_beta2", 0.999)),
                                     weight_decay=getattr(args, "weight_decay", 0.0)),
        "opt_sit": torch.optim.AdamW(sit_raw.parameters(), lr=args.lr,
                                     betas=(0.9, getattr(args, "adam_beta2", 0.999)),
                                     weight_decay=getattr(args, "weight_decay", 0.0)),
        "ema": ModelEMA(sit_raw, decay=getattr(args, "ema_decay_sit", 0.9999)),
        "trainloader": trainloader, "valloader": valloader,
        "global_step": 0,
    }

    start_epoch = 0
    resume_path = getattr(args, "resume", None)
    if resume_path == "auto":
        # What a requeued SLURM job passes: pick up this run's own last.pt if it exists,
        # otherwise start fresh. Anything else is treated as an explicit path.
        candidate = os.path.join(args.checkpoints, "last.pt")
        resume_path = candidate if os.path.exists(candidate) else None
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(state, resume_path)
        if is_main_process():
            print(f"[resume] from {resume_path} at epoch {start_epoch}")

    if is_main_process():
        create_directory(args.checkpoints)
        if getattr(args, "do_wandb", False):
            import wandb
            wandb.init(project=args.wandb_project, entity=getattr(args, "wandb_entity", None),
                       name=getattr(args, "run_prefix", "repae"), config=vars(args))

    for epoch in range(start_epoch, args.epochs):
        if is_dist() and hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)
        t0 = time.time()
        # Reset per epoch so the reported peak is this epoch's, not the run's high-water mark
        # from a one-off spike during setup.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_metrics = train_one_epoch(state, epoch)
        val_metrics = validate(state, epoch)

        # The [CLS] descriptor is frozen, so the partition converges within about one pass;
        # freezing it keeps Phase 2's mode definitions stable while the statistics keep up
        # with the still-moving tokenizer.
        if modes is not None and epoch + 1 >= getattr(args, "mode_freeze_epoch", 2):
            modes.freeze_codebook()

        # The per-mode latent statistics are cumulative by default, so early epochs are
        # averaged in forever. That is fine when the tokenizer starts pretrained and barely
        # moves; it is wrong from scratch, where the first epochs describe a latent space that
        # no longer exists. Resetting once the tokenizer has settled makes mu_m and sigma2_m
        # describe the CURRENT latent, which is the only one Phase 2 will sample in.
        reset_at = getattr(args, "mode_stats_reset_epoch", 0)
        if modes is not None and reset_at and epoch + 1 == reset_at:
            modes.reset_latent_statistics()
            if is_main_process():
                print(f"[modes] latent statistics reset at epoch {epoch + 1}; "
                      f"mu_m and sigma2_m now describe the current tokenizer only")

        if is_main_process():
            elapsed = time.time() - t0
            steps = min(len(trainloader), getattr(args, "limit_train_batches", 0) or len(trainloader))
            logs = {**train_metrics, **val_metrics,
                    # codebook_health_metrics already namespaces its keys with "Codebook/".
                    **codebook_health_metrics(state["vae"]),
                    "epoch": epoch, "epoch_time_s": elapsed,
                    # The two numbers the pre-flight probe exists to produce: seconds per
                    # optimizer step (which sets the epoch budget for the real run) and the
                    # peak memory (which decides whether the batch size is safe).
                    "s_per_step": elapsed / max(1, steps)}
            if torch.cuda.is_available():
                logs["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            bn_mean, bn_std = sit_raw.latent_stats()
            logs["bn_std_mean"] = float(bn_std.mean())
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                              for k, v in logs.items()}, indent=None))
            if getattr(args, "do_wandb", False):
                import wandb
                # Optimizer step, not epoch: it is the axis the sample grids use and the one
                # REPA-E's figures are plotted against, so curves line up with their marks.
                wandb.log(logs, step=state["global_step"])
            save_checkpoint(state, epoch, os.path.join(args.checkpoints, "last.pt"))
            every = getattr(args, "save_every_n_epochs", 5)
            if every and (epoch + 1) % every == 0:
                save_checkpoint(state, epoch, os.path.join(args.checkpoints, f"epoch_{epoch:03d}.pt"))

    if is_main_process():
        save_checkpoint(state, args.epochs - 1, os.path.join(args.checkpoints, "final_epoch.pt"))
    ddp_cleanup()
