#!/bin/bash
#SBATCH --job-name=flow_dualvae
#SBATCH --output=/data/mgonzalez/flow_dualvae.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=nukwa-08.cnca

# Class-conditional latent flow matching in the DUALVAE latent space
# (frozen AE: checkpoints/dualvae/dualvae_20260726-230704_866e33/best.pt).
# Pair job: train_flow_vae.slurm -- same generator, vanilla-VAE latents.

cd /work/mgonzalez/Hybrid_Latent_Space_AE

python main.py --model latent_flow --config ./configs/flow_dualvae.yaml
