"""Strip a REPA-E training checkpoint down to what EVALUATION needs, for transfer to another
cluster.

    python tools/export_for_eval.py --ckpt <checkpoint.pt> --out-dir <dir>

A training checkpoint is ~3.6 GB, but two thirds of it is optimizer state that no evaluation
reads: Adam's two moments for both models, the discriminator and its optimizer. Moving four of
them across clusters is 14 GB for 4.8 GB of useful weights.

This writes two things per checkpoint:

  eval_<name>.pt        tokenizer + EMA SiT + args + step/epoch (~0.9 GB).
                        What generate_repae.py and eval_cknna.py load. They read exactly these
                        keys, so they work unchanged.

  vae_<name>/           config_used.yaml + final_epoch.pt, a PLAIN DualVAE checkpoint directory
                        (~0.34 GB). What tools/rfid_imagenet.py expects -- it builds the model
                        from a config and loads a bare state_dict, so it cannot read the
                        training checkpoint's nested format.

The mode tracker's buffers (~285 MB: 4096 modes x 8192 dims, twice) are DROPPED by default.
They are the fitted mixture for the MM-FM phase and no evaluation touches them; pass
--keep-modes if you are moving the checkpoint for that work instead.
"""

import argparse
import os

import torch
import yaml

# Fields tools/rfid_imagenet.py (and tools/latent_ae.py) need to rebuild the tokenizer. Listed
# explicitly rather than dumping every training arg, so the exported config cannot accidentally
# carry a stale path or a training-only knob that changes how the model is built.
ARCH_KEYS = [
    "model", "quantizer", "fsq_levels", "latent_channels", "downsample_factor", "rq_depth",
    "residual_continuous", "component_prior", "sigma2_floor", "sigma2_ceil", "cont_dropout_p",
    "num_embeddings", "commitment_cost", "l2_normalize_codes", "use_ema_codebook",
    "ema_decay", "ema_eps", "ema_dead_threshold", "wavelet_detail", "wavelet_band_channels",
    "hierarchical_semantic", "coarse_factor", "coarse_num_embeddings",
    "resize_img", "dataset_name", "kl_beta",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, nargs="+", help="one or more training checkpoints")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keep-modes", action="store_true",
                   help="keep the mode-tracker buffers (only needed for the MM-FM phase)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.ckpt:
        name = os.path.basename(path).replace(".pt", "")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        src_gb = os.path.getsize(path) / 1024 ** 3

        slim = {
            "tokenizer": ckpt["tokenizer"],
            "sit_ema": ckpt["sit_ema"],
            "args": ckpt["args"],
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
        }
        if args.keep_modes and "modes" in ckpt:
            slim["modes"] = ckpt["modes"]
        slim_path = os.path.join(args.out_dir, f"eval_{name}.pt")
        torch.save(slim, slim_path)

        # The plain-DualVAE directory for rfid_imagenet.py.
        vae_dir = os.path.join(args.out_dir, f"vae_{name}")
        os.makedirs(vae_dir, exist_ok=True)
        torch.save(ckpt["tokenizer"], os.path.join(vae_dir, "final_epoch.pt"))
        cfg = {k: ckpt["args"][k] for k in ARCH_KEYS if k in ckpt["args"]}
        cfg["model"] = "dualvae"
        with open(os.path.join(vae_dir, "config_used.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        out_gb = (os.path.getsize(slim_path)
                  + os.path.getsize(os.path.join(vae_dir, "final_epoch.pt"))) / 1024 ** 3
        print(f"{name}: step {slim['global_step']} epoch {slim['epoch']} | "
              f"{src_gb:.2f} GB -> {out_gb:.2f} GB")

    print(f"\nwrote to {args.out_dir}")
    print("transfer that directory; generate_repae.py and eval_cknna.py take eval_*.pt, "
          "tools/rfid_imagenet.py takes --checkpoint-dir vae_*/")


if __name__ == "__main__":
    main()
