#!/bin/bash
#SBATCH --job-name=exp_vae_01
#SBATCH --output=/data/ffallas/generative/VAE/output/exp_vae_01.txt
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=nukwa-00.cnca
##SBATCH --mail-user=fabian.fallasmoya@gmail.com
##SBATCH --mail-type=ALL	

python /data/ffallas/generative/VAE/main.py --config ./configs/vae.yaml --model vae