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
    args, unknown = parser.parse_known_args()

    cfg_args = load_config_as_args(args.config)
    final_args = parser.parse_args(namespace=cfg_args)
    return final_args