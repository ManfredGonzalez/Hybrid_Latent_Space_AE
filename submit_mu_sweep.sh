#!/bin/bash
# Residual-mean sweep: Delta = alpha * mu + sigma * eps, rFID + recon metrics + CKNNA on the full
# 50k ImageNet val split, for one exported checkpoint. alpha = 1 is the plain evaluation
# (rfid_imagenet.json / cknna_eval_step_*.json) and is not rerun.
# 14 runs (~20 min rFID, ~13 min CKNNA each) packed onto 3 nodes, ~80 min wall clock.
# Usage: bash submit_mu_sweep.sh [step] [node_a] [node_b] [node_c]
STEP=${1:-0200000}
NA=${2:-nukwa-04.cnca}
NB=${3:-nukwa-05.cnca}
NC=${4:-nukwa-07.cnca}

E=/data/image-models-project/manfred/eval_export
D=/data/image-models-project/datasets/imagenet_full_256.h5
O=$E/mu_sweep_$STEP
REPO=/work/mgonzalez/Hybrid_Latent_Space_AE
PRE="source /data/mgonzalez/miniconda3/etc/profile.d/conda.sh && conda activate ddpm && cd $REPO"
COMMON="--partition=nukwa-long --ntasks=1 --cpus-per-task=16 --exclusive --time=04:00:00"
mkdir -p $O

rfid()  { echo "python -u tools/rfid_imagenet.py --checkpoint-dir $E/vae_step_$STEP --subset imagenet-val-h5 --dataset-path $D --batch-size 64 --kid-subset 1000 --mu-scale $1 --out $O/rfid_mu$1.json"; }
cknna() { echo "python -u eval_cknna.py --ckpt $E/eval_step_$STEP.pt --dataset-path $D --batch-size 64 --mu-scale $1 --out $O/cknna_mu$1.json"; }

# ';' not '&&': one failed alpha must not cancel the rest of the node's queue.
submit() {  # name node cmd...
  local name=$1 node=$2; shift 2
  local cmd; cmd=$(IFS=';'; echo "$*")
  sbatch $COMMON --job-name=$name --nodelist=$node --output=$REPO/logs/${name}_%j.txt \
    --wrap="$PRE; $cmd"
}

submit mu_a_$STEP $NA "$(rfid 0.975)" "$(rfid 0.950)" "$(rfid 0.900)" "$(rfid 0.850)"
submit mu_b_$STEP $NB "$(rfid 0.800)" "$(rfid 0.750)" "$(rfid 0.700)" "$(cknna 0.975)"
submit mu_c_$STEP $NC "$(cknna 0.950)" "$(cknna 0.900)" "$(cknna 0.850)" "$(cknna 0.800)" "$(cknna 0.750)" "$(cknna 0.700)"
