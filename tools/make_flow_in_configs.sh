#!/bin/bash
# Regenerates the three sibling ImageNet flow configs from configs/flow_in_vq01.yaml.
#
# The four arms of this comparison must differ in EXACTLY three lines -- ae_checkpoint,
# checkpoints, run_prefix -- or a Gen/FID difference stops being attributable to the latent
# space. Generating them mechanically is how that stays true after someone retunes a
# hyperparameter: edit flow_in_vq01.yaml, re-run this, and the other three follow.
#
# Verify at any time with:  bash tools/make_flow_in_configs.sh --check
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=configs/flow_in_vq01.yaml
CHECK=${1:-}

# tag | ae_checkpoint | header line describing the arm
make_arm() {
  local tag="$1" ae="$2" desc="$3"
  local out="configs/flow_in_${tag}.yaml"
  local tmp; tmp=$(mktemp)
  sed -e "s|^ae_checkpoint: .*|ae_checkpoint: \"${ae}\"|" \
      -e "s|^run_prefix: .*|run_prefix: \"flow_in_${tag}\"|" \
      -e "s|^checkpoints: .*|checkpoints: \"/work/mgonzalez/Hybrid_Latent_Space_AE/checkpoints/flow_in_${tag}/\"|" \
      -e "s|^# ARM 1 of 4: .*|# ARM: ${desc}|" \
      -e "s|^# Run:  sbatch flow_in_vq01.slurm|# Run:  sbatch flow_in_${tag}.slurm|" \
      "$SRC" > "$tmp"
  if [[ "$CHECK" == "--check" ]]; then
    if diff -q "$tmp" "$out" >/dev/null 2>&1; then
      echo "OK    $out"
    else
      echo "DRIFT $out  (run without --check to regenerate)"; diff "$out" "$tmp" || true
    fi
    rm -f "$tmp"
  else
    mv "$tmp" "$out"; echo "wrote $out"
  fi
}

make_arm fsq01 "./checkpoints/dualvae/FSQ_0.1/dualvae_20260811-093454_af2305" \
  "the DualVAE FSQ 0.1 latent space (FSQ, kl_beta 0.1, rFID 0.463)."
make_arm fsqE  "./checkpoints/dualvae/fsq_E/dualvae_20260813-204926_f85dfe" \
  "the DualVAE FSQ-E latent space (FSQ, kl_beta 1.0, N(0,I) prior, rFID 0.586)."
make_arm vae   "./checkpoints/vae/vae_20260810-165000_c9374d" \
  "the plain VAE latent space (no z_vq branch, rFID 0.472)."

if [[ "$CHECK" != "--check" ]]; then
  echo
  echo "Differences from the source config (should be 3 settings + 2 comment lines each):"
  for t in fsq01 fsqE vae; do
    n=$(diff "$SRC" "configs/flow_in_${t}.yaml" | grep -c '^[<>]' || true)
    echo "  flow_in_${t}.yaml: $n changed lines"
  done
fi
