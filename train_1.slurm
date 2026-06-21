#!/bin/bash
#SBATCH --job-name=hybrid
#SBATCH --output=/data/mgonzalez/hybrid.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=nukwa-08.cnca

cd /work/mgonzalez/Hybrid_Latent_Space_AE

python main.py --model dualvae --config ./configs/dualvae.yaml