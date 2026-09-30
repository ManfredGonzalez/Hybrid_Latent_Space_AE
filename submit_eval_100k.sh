#!/bin/bash
# Submit rFID, CKNNA and gFID for one exported checkpoint, each pinned to its own nukwa node.
# Usage: bash submit_eval_100k.sh [step] [rfid_node] [cknna_node] [gfid_node]
# JOBS="rfid cknna" bash submit_eval_100k.sh ...   submits only the listed jobs.
# All three use the full 50k ImageNet val split.

JOBS=${JOBS:-"rfid cknna gfid"}
STEP=${1:-0100000}
N_RFID=${2:-nukwa-04.cnca}
N_CKNNA=${3:-nukwa-05.cnca}
N_GFID=${4:-nukwa-06.cnca}

E=/data/image-models-project/manfred/eval_export
D=/data/image-models-project/datasets/imagenet_full_256.h5
S=/work/mgonzalez/samples/step_$STEP
REPO=/work/mgonzalez/Hybrid_Latent_Space_AE
PRE="source /data/mgonzalez/miniconda3/etc/profile.d/conda.sh && conda activate ddpm && cd $REPO"
COMMON="--partition=nukwa-long --ntasks=1 --cpus-per-task=16 --exclusive"

[[ " $JOBS " == *" rfid "* ]] && sbatch $COMMON --job-name=rfid_$STEP --nodelist=$N_RFID --time=04:00:00 \
  --output=$REPO/logs/rfid_${STEP}_%j.txt \
  --wrap="$PRE && python -u tools/rfid_imagenet.py --checkpoint-dir $E/vae_step_$STEP \
    --subset imagenet-val-h5 --dataset-path $D --batch-size 64 --kid-subset 1000"

[[ " $JOBS " == *" cknna "* ]] && sbatch $COMMON --job-name=cknna_$STEP --nodelist=$N_CKNNA --time=04:00:00 \
  --output=$REPO/logs/cknna_${STEP}_%j.txt \
  --wrap="$PRE && python -u eval_cknna.py --ckpt $E/eval_step_$STEP.pt --dataset-path $D --batch-size 64"

# Generation and scoring run back to back in one job, so scoring only starts once the .npz exists.
[[ " $JOBS " == *" gfid "* ]] && sbatch $COMMON --job-name=gfid_$STEP --nodelist=$N_GFID --time=48:00:00 \
  --output=$REPO/logs/gfid_${STEP}_%j.txt \
  --wrap="$PRE && python -u generate_repae.py --ckpt $E/eval_step_$STEP.pt --num-samples 50000 \
    --mode sde --num-steps 250 --cfg-scale 1.0 --out-dir $S \
    && python -u tools/score_npz_fid.py --dataset-path $D --npz $S/*.npz"
