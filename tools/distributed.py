"""Single-node multi-GPU (DDP) helpers.

The trainers were written single-process; this module is the only place that knows about
`torch.distributed`, so the single-GPU path stays bit-identical when the job is launched
without torchrun (every helper degrades to a no-op when `is_dist()` is False).

Launch contract: `torchrun --nproc_per_node=N`, which sets RANK / WORLD_SIZE / LOCAL_RANK.
Running `python main.py` directly leaves those unset -> single-process, unchanged behavior.

Why NCCL all-reduce shows up in three places (see the trainers):
  * gradients            -- handled automatically by DistributedDataParallel;
  * EMA codebook stats   -- NOT handled by DDP (they are buffers mutated in-place, not
                            gradients), so models/modules/embedding.py all-reduces them
                            explicitly. Without that, each rank would run online k-means on
                            1/N of the batch and the codebooks would silently diverge;
  * epoch metrics        -- each rank sees a different shard, so the logged numbers must be
                            averaged over ranks (all_reduce_metrics) or they only describe
                            rank 0's shard.
"""

import os
import datetime

import torch
import torch.distributed as dist


def is_dist():
    """True once the process group is up (i.e. launched under torchrun with N>1)."""
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main_process():
    """Rank 0. Gate ALL logging, checkpointing and wandb on this."""
    return get_rank() == 0


def ddp_setup(timeout_minutes=60):
    """Initialize the process group from torchrun's env vars and bind this rank to its GPU.

    Returns (device, local_rank, rank, world). When the env vars are absent (plain
    `python main.py`) nothing is initialized and this returns the single-GPU device, so the
    same code path serves both launch modes.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    has_cuda = torch.cuda.is_available()
    # NCCL for GPU collectives; gloo is the CPU fallback, which exists so the DDP wiring
    # (sampler sharding, EMA all-reduce, metric reduction, checkpoint keys) can be tested
    # without occupying GPUs.
    backend = "nccl" if has_cuda else "gloo"
    if has_cuda:
        torch.cuda.set_device(local_rank)
    # The generous timeout covers the one-off stalls that are normal at the start of a big run
    # (k-means init on rank 0, the first FID/KID sync) while the other ranks sit in a collective.
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
    device = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    return device, local_rank, dist.get_rank(), dist.get_world_size()


def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def barrier():
    if is_dist():
        dist.barrier()


def unwrap(model):
    """The underlying module, so trainer code can reach `.vq_layer`, `.encoder`, `.decoder`
    etc. regardless of whether the model is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


def all_reduce_metrics(metrics, device):
    """Average a {name: float} epoch-metric dict across ranks.

    Each rank only ever saw its own shard of the data, so without this the logged curves
    are rank 0's shard, not the run's. Packed into ONE tensor so this is a single collective
    per epoch rather than one per metric.
    """
    if not is_dist() or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    t = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world_size()
    return {k: t[i].item() for i, k in enumerate(keys)}


def all_reduce_sum_(*tensors):
    """In-place SUM all-reduce of the given tensors (no-op single-process).

    Used for the EMA codebook statistics, where summing the per-rank counts/vector-sums is
    exactly equivalent to having computed them on the full global batch.
    """
    if not is_dist():
        return
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def broadcast_(tensor, src=0):
    """In-place broadcast from `src` (no-op single-process).

    Needed anywhere a rank makes a RANDOM choice that must be identical everywhere --
    dead-code restarts sample from the local batch, so without this the codebooks drift apart.
    """
    if not is_dist():
        return
    dist.broadcast(tensor, src=src)


def broadcast_module_state_(module, src=0):
    """Broadcast every parameter AND buffer of `module` from `src`.

    Used after the k-means codebook init, which runs on one rank's batch: the resulting
    centroids and their seeded EMA statistics have to be copied to every rank before the
    first training step.
    """
    if not is_dist():
        return
    with torch.no_grad():
        for t in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(t.data, src=src)


def all_reduce_grads_(module):
    """Average gradients across ranks for a module that is NOT DDP-wrapped.

    The PatchGAN discriminator is deliberately left unwrapped: it is stepped by its own
    optimizer in a SECOND backward pass each iteration, and DDP's reducer expects exactly
    one backward per forward, so wrapping it triggers the 'gradient ready twice' error.
    It is small (~2.8M params), so one explicit all-reduce per step is cheap.
    """
    if not is_dist():
        return
    ws = world_size()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    # Flatten into one buffer -> a single NCCL call instead of one per tensor.
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= ws
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n
