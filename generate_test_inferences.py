import os
import csv
import json
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _FID_AVAILABLE = True
except ImportError:
    _FID_AVAILABLE = False

from tools.arguments import parse_args, load_config_as_args
from tools.utils import scale_ratio
from data.datasets import PineappleDataset, get_benchmark_dataset

# Import your models
from models.vae import VAE
from models.vqvae import VQVAE
from models.dual_vae import DUALVAE
from models.sw_dualvae import SW_DUALVAE

# Branch-ablation modes supported by DUALVAE / SW_DUALVAE. -1 keeps both branches active,
# 0 zeroes the vanilla (continuous) branch, 1 zeroes the VQ branch (see forward() in those models).
ABLATIONS = {
    "full": -1,
    "vq_only": 0,
    "vanilla_only": 1,
}

DUAL_BRANCH_MODELS = ("dualvae", "swd_dualvae")

# Defaults filled in for --run_list runs that don't set every field a normal single-config
# invocation would get from tools/arguments.py's parser.
RUN_LIST_DEFAULTS = {
    'path_test_ids': './test_ids.txt',
    'inference_split': 'val',
    'ablation_mode': 'all',
    'skip_fid': False,
    'num_workers': 2,
}

CSV_COLUMNS = [
    "config", "model", "checkpoint", "dataset", "split", "ablation_mode",
    "mse", "psnr", "ssim", "fid",
    "scale_ratio", "codebook_perplexity", "codebook_loss", "actual_mean_variance",
]


def denormalize(tensor, dataset_name, device):
    """Reverses input normalization so metrics/saved images are computed in real pixel space."""
    name = dataset_name.lower()
    if name == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    if name == 'imagenette':
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean
    # Pineapple and MNIST are already in [0, 1] range via Min-Max scaling
    return tensor


def prepare_data(args, split):
    """Loads the requested split, mirroring the logic used by the train_*.py scripts."""
    dataset_name = getattr(args, 'dataset_name', 'pineapple').lower()

    if dataset_name == 'pineapple':
        dataset = PineappleDataset(
            path=args.dataset_path,
            split=split,
            test_txt=args.path_test_ids,
            augment=False,
            seed=args.seed
        )
    elif dataset_name == 'imagenette':
        # get_benchmark_dataset ignores 'split' for imagenette and always returns (train, val).
        train_dataset, val_dataset = get_benchmark_dataset(
            dataset_name, path=args.dataset_path, resize_img=args.resize_img, seed=args.seed
        )
        dataset = train_dataset if split == 'train' else val_dataset
    else:
        dataset = get_benchmark_dataset(
            dataset_name, path=args.dataset_path, split=split,
            val_ratio=getattr(args, 'val_ratio', 0.2), resize_img=args.resize_img, seed=args.seed
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 2)
    )
    return dataset, loader, dataset_name


def load_model(args, device):
    """Instantiates the correct model and loads the checkpoint."""
    downsample_factor = getattr(args, 'downsample_factor', 8)
    l2_normalize_codes = getattr(args, 'l2_normalize_codes', False)
    # Must match the training-time flags: EMA checkpoints carry extra buffers
    # (ema_cluster_size / ema_embed_sum / ema_res_sq) and the residual wiring changes
    # the vanilla bottleneck's shape, so load_state_dict fails if the flags disagree.
    use_ema_codebook = getattr(args, 'use_ema_codebook', False)
    rq_depth = getattr(args, 'rq_depth', 1)
    residual_continuous = getattr(args, 'residual_continuous', False)
    component_prior = getattr(args, 'component_prior', False)

    if args.model == "vae":
        model = VAE(downsample_factor=downsample_factor)
    elif args.model == "vqvae":
        model = VQVAE(
            commitment_cost=args.commitment_cost,
            latent_channels=args.latent_channels,
            num_embeddings=args.num_embeddings,
            downsample_factor=downsample_factor,
            l2_normalize_codes=l2_normalize_codes,
            use_ema_codebook=use_ema_codebook,
            rq_depth=rq_depth
        )
    elif args.model == "dualvae":
        model = DUALVAE(
            commitment_cost=args.commitment_cost,
            latent_channels=args.latent_channels,
            num_embeddings=args.num_embeddings,
            downsample_factor=downsample_factor,
            l2_normalize_codes=l2_normalize_codes,
            use_ema_codebook=use_ema_codebook,
            rq_depth=rq_depth,
            residual_continuous=residual_continuous,
            component_prior=component_prior
        )
    elif args.model == "swd_dualvae":
        model = SW_DUALVAE(
            commitment_cost=args.commitment_cost,
            latent_channels=args.latent_channels,
            num_embeddings=args.num_embeddings,
            downsample_factor=downsample_factor,
            combine_mode=getattr(args, 'combine_mode', 'residual_addition'),
            l2_normalize_codes=l2_normalize_codes,
            use_ema_codebook=use_ema_codebook,
            rq_depth=rq_depth,
            residual_continuous=residual_continuous,
            component_prior=component_prior,
            wavelet_detail=getattr(args, 'wavelet_detail', False),
            wavelet_band_channels=getattr(args, 'wavelet_band_channels', None),
            learned_band_variance=getattr(args, 'learned_band_variance', False),
            band_sigma0_prior=getattr(args, 'swd_sigma0_bands', None)
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)

    # Load the trained weights
    if not os.path.exists(args.checkpoint_path_test):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path_test}")

    model.load_state_dict(torch.load(args.checkpoint_path_test, map_location=device))
    model.eval()

    return model


def forward_model(model, model_name, images, ablation_mode):
    """Runs the forward pass appropriate for the model, applying the branch ablation when supported.

    For dualvae/swd_dualvae, also returns the vq/vanilla loss-term dicts so the caller can
    replicate the same codebook/variance diagnostics computed during training and validation.
    """
    if model_name in DUAL_BRANCH_MODELS:
        recon, vq_related_losses, vanilla_vae_related_losses = model(images, ablation_mode=ablation_mode)
        return recon, vq_related_losses, vanilla_vae_related_losses
    elif model_name == "vqvae":
        recon, _, _, codebook_loss = model(images)
        return recon, {"codebook_loss": codebook_loss}, None
    else:  # vae
        recon, _, _ = model(images)
        return recon, None, None


def get_filename(dataset, idx):
    """Recovers a stable filename for an image so reconstructions can be matched back to inputs."""
    if hasattr(dataset, 'images'):  # PineappleDataset
        return os.path.basename(dataset.images[idx])
    if hasattr(dataset, 'image_files'):  # FlatImageDataset (imagenette)
        return dataset.image_files[idx]
    return f"{idx:06d}.png"


def run_inference(model, loader, dataset, args, device, dataset_name, ablation_mode, output_dir, compute_fid):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(normalize=True).to(device) if compute_fid else None
    track_dual_diagnostics = args.model in DUAL_BRANCH_MODELS
    # vqvae also has a vq_layer (perplexity/codebook_loss) but no vanilla branch, so it gets
    # its own flag rather than reusing track_dual_diagnostics (no scale_ratio/mean_variance).
    track_codebook_diagnostics = track_dual_diagnostics or args.model == "vqvae"

    os.makedirs(output_dir, exist_ok=True)

    total_mse, total_psnr, total_ssim, num_batches = 0.0, 0.0, 0.0, 0
    total_scale_ratio, total_perplexity, total_codebook_loss, total_actual_mean_variance = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{ablation_mode}] Generating Inferences"):
            images = batch["image"].to(device)
            indices = batch["idx"]

            recon, vq_related_losses, vanilla_vae_related_losses = forward_model(
                model, args.model, images, ABLATIONS[ablation_mode]
            )

            # Denormalize before computing metrics and before saving, so both are in real [0, 1] pixel space.
            denorm_images = denormalize(images, dataset_name, device).clamp(0, 1)
            denorm_recon = denormalize(recon, dataset_name, device).clamp(0, 1)

            batch_mse = F.mse_loss(denorm_recon, denorm_images, reduction='mean')
            batch_psnr = psnr_metric(denorm_recon, denorm_images)
            batch_ssim = ssim_metric(denorm_recon, denorm_images)

            total_mse += batch_mse.item()
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            num_batches += 1

            if track_dual_diagnostics:
                # Same mechanism as train_dualvae.py / train_swd_dualvae.py's validate_one_epoch:
                # scale_ratio and z_vanilla_post/z_vq are the POST-ablation tensors, so in
                # vq_only/vanilla_only mode the zeroed-out branch will drive this ratio to the
                # extremes (as expected - it mirrors what happens during ablation training runs).
                raw_variance = torch.exp(vanilla_vae_related_losses["log_variance"]).mean()
                batch_scale_ratio = scale_ratio(vq_related_losses["z_vq"], vanilla_vae_related_losses["z_vanilla_post"])
                total_scale_ratio += batch_scale_ratio.item()
                total_actual_mean_variance += raw_variance.item()

            if track_codebook_diagnostics:
                total_perplexity += model.vq_layer.perplexity.item()
                total_codebook_loss += vq_related_losses["codebook_loss"].item()

            if fid_metric is not None:
                fid_metric.update(denorm_images, real=True)
                fid_metric.update(denorm_recon, real=False)

            for i in range(images.size(0)):
                dataset_idx = indices[i].item()
                filename = get_filename(dataset, dataset_idx)
                save_path = os.path.join(output_dir, filename)
                vutils.save_image(denorm_recon[i], save_path)

    metrics = {
        "mse": total_mse / num_batches,
        "psnr": total_psnr / num_batches,
        "ssim": total_ssim / num_batches,
    }
    if fid_metric is not None:
        metrics["fid"] = fid_metric.compute().item()
    if track_dual_diagnostics:
        metrics["scale_ratio"] = total_scale_ratio / num_batches
        metrics["actual_mean_variance"] = total_actual_mean_variance / num_batches
    if track_codebook_diagnostics:
        metrics["perplexity"] = total_perplexity / num_batches
        metrics["codebook_loss"] = total_codebook_loss / num_batches

    return metrics


def build_row(args, dataset_name, split, mode, metrics):
    return {
        "config": getattr(args, 'config', ''),
        "model": args.model,
        "checkpoint": args.checkpoint_path_test,
        "dataset": dataset_name,
        "split": split,
        "ablation_mode": mode,
        "mse": metrics.get("mse"),
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "fid": metrics.get("fid", ""),
        "scale_ratio": metrics.get("scale_ratio", ""),
        "codebook_perplexity": metrics.get("perplexity", ""),
        "codebook_loss": metrics.get("codebook_loss", ""),
        "actual_mean_variance": metrics.get("actual_mean_variance", ""),
    }


def write_csv(csv_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(args, split, all_metrics):
    has_dual_diagnostics = any('scale_ratio' in m for m in all_metrics.values())
    has_codebook_diagnostics = any('perplexity' in m for m in all_metrics.values())

    print("\n" + "=" * 66)
    print(f"FINAL METRICS SUMMARY ({args.model}, split={split})")
    print("-" * 66)
    header = f"{'Mode':<14}{'MSE':>10}{'PSNR':>10}{'SSIM':>10}{'FID':>10}"
    if has_dual_diagnostics:
        header += f"{'ScaleRatio':>12}"
    if has_codebook_diagnostics:
        header += f"{'Perplexity':>12}{'CBLoss':>10}"
    if has_dual_diagnostics:
        header += f"{'MeanVar':>10}"
    print(header)
    for mode, metrics in all_metrics.items():
        fid_str = f"{metrics['fid']:.4f}" if 'fid' in metrics else "n/a"
        row = f"{mode:<14}{metrics['mse']:>10.6f}{metrics['psnr']:>10.4f}{metrics['ssim']:>10.4f}{fid_str:>10}"
        if has_dual_diagnostics:
            # scale_ratio can blow up to ~1e7 or collapse to 0 in vq_only/vanilla_only mode
            # (the zeroed branch dominates the ratio) - use %g so the column stays aligned.
            row += f"{metrics['scale_ratio']:>12.4g}"
        if has_codebook_diagnostics:
            row += f"{metrics['perplexity']:>12.4f}{metrics['codebook_loss']:>10.4f}"
        if has_dual_diagnostics:
            row += f"{metrics['actual_mean_variance']:>10.4f}"
        print(row)
    print("=" * 66)


def generate_inferences(args):
    """Runs inference (+ metrics) for one model/config. Returns a list of CSV row dicts, one per ablation mode."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = getattr(args, 'inference_split', 'val')
    print(f"Loading '{split}' split for dataset '{getattr(args, 'dataset_name', 'pineapple')}'...")
    dataset, loader, dataset_name = prepare_data(args, split)

    print(f"Loading {args.model} model from {args.checkpoint_path_test}...")
    model = load_model(args, device)

    # Decide which branch-ablations to run. Only the dual-branch architectures support ablation_mode.
    if args.model in DUAL_BRANCH_MODELS:
        requested = getattr(args, 'ablation_mode', 'all')
        modes_to_run = list(ABLATIONS.keys()) if requested == 'all' else [requested]
    else:
        modes_to_run = ["full"]

    compute_fid = not getattr(args, 'skip_fid', False) and _FID_AVAILABLE
    if compute_fid:
        # FrechetInceptionDistance imports cleanly even without torch-fidelity; it only fails
        # once you actually try to build the InceptionV3 feature extractor, so probe it here.
        try:
            FrechetInceptionDistance(normalize=True)
        except ModuleNotFoundError:
            compute_fid = False
    if not compute_fid and not getattr(args, 'skip_fid', False):
        print("WARNING: FID unavailable (torchmetrics/torch-fidelity not installed) - skipping FID. "
              "Install with `pip install torchmetrics torch-fidelity` to enable it.")

    print(f"Inferences will be saved under: {args.output_dir_test}")

    all_metrics = {}
    for mode in modes_to_run:
        # Single-mode runs keep images directly in output_dir_test; multi-mode ("all") runs
        # split them into per-mode subfolders so the ablations don't overwrite each other.
        out_dir = args.output_dir_test if len(modes_to_run) == 1 else os.path.join(args.output_dir_test, mode)
        print(f"\n=== Mode: {mode} (ablation_mode={ABLATIONS[mode]}) -> {out_dir} ===")
        all_metrics[mode] = run_inference(model, loader, dataset, args, device, dataset_name, mode, out_dir, compute_fid)

    os.makedirs(args.output_dir_test, exist_ok=True)
    summary_path = os.path.join(args.output_dir_test, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print_summary(args, split, all_metrics)
    print(f"Saved metrics summary to {summary_path}")

    rows = [build_row(args, dataset_name, split, mode, metrics) for mode, metrics in all_metrics.items()]

    # Also write this run's own CSV (useful standalone); --run_list overwrites it with a combined one afterwards.
    csv_path = getattr(args, 'csv_path', None) or os.path.join(args.output_dir_test, "metrics_report.csv")
    write_csv(csv_path, rows)
    print(f"Saved metrics CSV to {csv_path}")
    print("Inference complete!")

    return rows


def load_run_args(run_entry):
    """Builds an args Namespace for one --run_list entry: its config YAML plus per-run overrides.

    checkpoint_path_test / output_dir_test can be set either in the config YAML itself (handy
    when a checkpoint is permanently tied to that architecture) or in the run_list entry (handy
    for sweeping different checkpoints/output dirs over the same config) - the run_entry value
    wins if both are present. Either way, the merged result must end up with both set.
    """
    if "config" not in run_entry:
        raise ValueError(f"Each entry in --run_list must set 'config'. Got: {run_entry}")

    cfg_args = load_config_as_args(run_entry["config"])
    cfg_args.config = run_entry["config"]

    for key, value in run_entry.items():
        if key == "config":
            continue
        setattr(cfg_args, key, value)

    for required in ("checkpoint_path_test", "output_dir_test"):
        if not getattr(cfg_args, required, None):
            raise ValueError(
                f"'{required}' must be set either in '{run_entry['config']}' or in its --run_list entry."
            )

    for key, value in RUN_LIST_DEFAULTS.items():
        if not hasattr(cfg_args, key):
            setattr(cfg_args, key, value)

    return cfg_args


def run_from_list(run_list_path, csv_path_override=None):
    """Evaluates every entry in a --run_list manifest (each can be a different model class/checkpoint)
    and writes all their per-ablation-mode rows into a single combined CSV."""
    with open(run_list_path, "r") as f:
        manifest = yaml.safe_load(f)

    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"No 'runs' entries found in manifest: {run_list_path}")

    csv_path = csv_path_override or manifest.get("csv_path", "./inferences/metrics_report.csv")

    all_rows = []
    for run_entry in runs:
        run_args = load_run_args(run_entry)
        print(f"\n{'#' * 70}\n# Run: {run_entry['config']}  (model={run_args.model})\n{'#' * 70}")
        all_rows.extend(generate_inferences(run_args))

    write_csv(csv_path, all_rows)
    print(f"\nWrote combined metrics CSV ({len(all_rows)} rows across {len(runs)} run(s)) to {csv_path}")


if __name__ == "__main__":
    # Detect --run_list up front: in that mode we skip tools.arguments.parse_args() entirely,
    # since its --config default doesn't exist on this machine and each run supplies its own config.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--run_list', default=None, type=str)
    pre_parser.add_argument('--csv_path', default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.run_list:
        run_from_list(pre_args.run_list, csv_path_override=pre_args.csv_path)
    else:
        args = parse_args()
        generate_inferences(args)
