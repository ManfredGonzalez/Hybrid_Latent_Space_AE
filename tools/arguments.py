import argparse
import yaml
from argparse import Namespace

def load_config_as_args(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def flatten_dict(d, parent_key='', sep='_'):
        """Flatten a nested dictionary preserving types."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v  # value type is preserved
                print(f"Key {k}", type(v))
        return items

    flat_cfg = flatten_dict(cfg)
    return Namespace(**flat_cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Pineapple Dataset")
    parser.add_argument('--config', default="/home/rtxmsi1/Documents/VAE/configs/vae_perceptual.yaml", type=str)
    parser.add_argument('--model', default="vae_perceptual", type=str)
    parser.add_argument('--path_test_ids', default="./test_ids.txt", type=str)
    parser.add_argument('--checkpoint_path_test', default="./checkpoints/VAE/betaKL@0.001/best.pt", type=str)
    parser.add_argument('--output_dir_test', default="./inferences/vae/test", type=str)
    parser.add_argument('--inference_split', default="val", type=str, choices=['train', 'val', 'test'],
                         help="Which split to run generate_test_inferences.py / evaluate_metrics.py on.")
    parser.add_argument('--ablation_mode', default="all", type=str, choices=['all', 'full', 'vq_only', 'vanilla_only'],
                         help="For dualvae/swd_dualvae: which branch-ablation to run in generate_test_inferences.py. "
                              "'all' runs full model plus both single-branch ablations.")
    parser.add_argument('--skip_fid', action='store_true',
                         help="Skip FID computation in generate_test_inferences.py (e.g. if torch-fidelity isn't installed).")
    parser.add_argument('--run_list', default=None, type=str,
                         help="Path to a YAML manifest listing multiple (config, checkpoint, output_dir) runs for "
                              "generate_test_inferences.py, so several models/classes can be evaluated in one go.")
    parser.add_argument('--csv_path', default=None, type=str,
                         help="Where generate_test_inferences.py writes the combined metrics CSV. Defaults to "
                              "<output_dir_test>/metrics_report.csv (or the manifest's csv_path in --run_list mode).")
    args, unknown = parser.parse_known_args()

    cfg_args = load_config_as_args(args.config)
    final_args = parser.parse_args(namespace=cfg_args)
    return final_args