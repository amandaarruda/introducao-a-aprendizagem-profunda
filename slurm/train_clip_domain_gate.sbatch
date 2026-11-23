#!/usr/bin/env bash
# Re-train the frozen CLIP RN101 domain router on gate_train/gate_val only.
#SBATCH --job-name=ligia-clip-domain-gate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/clip-domain-gate-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
clip="${CLIP_RN101_CHECKPOINT:-$PWD/checkpoints/RN101.pt}"
output="${GATE_OUTPUT_DIR:-$PWD/outputs/domain-gate-clip-rn101-${SLURM_JOB_ID}}"
for path in "$root/split_manifest.json" "$clip"; do test -s "$path"; done
test ! -e "$output"
.venv-ligia/bin/python scripts/train_domain_gate.py \
  --annotations-root "$root" --encoder clip-rn101 \
  --clip-checkpoint "$clip" --output-dir "$output" \
  --batch-size 32 --workers 2 --max-epochs 200 --seed 1337
test -s "$output/domain_gate_linear.pt"
sha256sum "$output/domain_gate_linear.pt" "$root/split_manifest.json" "$clip"
