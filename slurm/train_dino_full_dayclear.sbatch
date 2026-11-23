#!/usr/bin/env bash
# Train the DINO DayClear baseline on every eligible official train image.
# The four frozen 100-image reserves remain outside the training pool.
#SBATCH --job-name=ligia-dino-dayclear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-dino-dayclear-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate

export DWD_ROOT="${DINO_SPLIT_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export TRAIN_DOMAIN=day_clear MAX_ITERS="${MAX_ITERS:-30000}" INIT_CHECKPOINT=""
export DINO_R50_CHECKPOINT="${DINO_R50_CHECKPOINT:?set DINO_R50_CHECKPOINT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config="${CONFIG_PATH:-$PWD/configs/dino_dwd_full.py}"
work_dir="$PWD/outputs/dino-dayclear-full-${SLURM_JOB_ID}"
train_ann="$DWD_ROOT/day_clear/day_clear_train.coco.json"
val_ann="$DWD_ROOT/day_clear/day_clear_val.coco.json"
for path in "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$DINO_R50_CHECKPOINT" "$config"; do test -s "$path"; done
test ! -e "$work_dir"
mkdir -p "$work_dir"

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("cuda_device=", torch.cuda.get_device_name(0), "capability=", torch.cuda.get_device_capability(0))
print("cuda_smoke=", (torch.arange(4, device="cuda") * 2).cpu().tolist())
CUDA_PREFLIGHT
echo "job=$SLURM_JOB_ID domain=day_clear images=full-eligible updates=$MAX_ITERS start=$(date -Is)"
sha256sum "$config" "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$DINO_R50_CHECKPOINT"
mim train mmdet "$config" --work-dir "$work_dir"
checkpoint="$work_dir/iter_${MAX_ITERS}.pth"
test -s "$checkpoint"
printf '%s\n' "$checkpoint" > "$work_dir/checkpoint_path.txt"
sha256sum "$checkpoint" "$work_dir/checkpoint_path.txt"
echo "job=$SLURM_JOB_ID finished=$(date -Is)"
