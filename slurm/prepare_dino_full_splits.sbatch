#!/usr/bin/env bash
# Build full train pools while preserving model-validation, gate and test reserves.
# The output directory must be new; the script refuses to overwrite it.
#SBATCH --job-name=ligia-dino-splits
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-dino-splits-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

root="${DWD_ROOT_PARENT:-$PWD/data/raw/dwd}"
processed="${PROCESSED_ROOT:-$root/processed}"
processed_v4="${PROCESSED_V4_ROOT:-$root/processed-v4}"
frozen="${FROZEN_MANIFEST:-$root/domain-experts-1000x100-modelval100-gate100-gateval100-test100-20260919/split_manifest.json}"
output="${DINO_SPLIT_ROOT:-$root/full-train-frozen-eval-20261005}"
test ! -e "$output"

sha256sum scripts/make_full_domain_splits.py "$frozen" "$processed"/*.coco.json
args=(--root "$root" --processed "$processed" --frozen-manifest "$frozen" --output "$output")
test ! -e "$processed_v4" || args+=(--processed-v4 "$processed_v4")
.venv-ligia/bin/python scripts/make_full_domain_splits.py "${args[@]}" --write
test -s "$output/split_manifest.json"
sha256sum "$output/split_manifest.json" "$output"/*/*.coco.json
