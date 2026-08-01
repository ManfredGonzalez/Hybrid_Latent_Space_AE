#!/bin/bash
#SBATCH --job-name=flow_vae
#SBATCH --output=/data/mgonzalez/flow_vae.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=nukwa-09.cnca

# Class-conditional latent flow matching in the VANILLA VAE latent space -- the control for
# train_flow_dualvae.slurm (frozen AE: checkpoints/vae/vae_20260729-110353_138ede/best.pt).
# Run this on a DIFFERENT node than the dualvae job so the two train in parallel.

cd /work/mgonzalez/Hybrid_Latent_Space_AE

python main.py --model latent_flow --config ./configs/flow_vae.yaml
