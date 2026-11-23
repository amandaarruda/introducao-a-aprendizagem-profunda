#!/usr/bin/env bash
# Fine-tune one DINO expert per target domain from a frozen DayClear baseline.
# Set BASELINE_CHECKPOINT to the completed DayClear iter_*.pth and use its job ID
# as CAMPAIGN_ID so all experts share one output root.
#SBATCH --job-name=ligia-dino-experts
#SBATCH --array=0-3%1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-dino-expert-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
domains=(day_foggy dusk_rainy night_clear night_rainy)
export TRAIN_DOMAIN="${domains[$SLURM_ARRAY_TASK_ID]}"
: "${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT}"
: "${CAMPAIGN_ID:?set CAMPAIGN_ID}"
source .venv-ligia/bin/activate

export DWD_ROOT="${DINO_SPLIT_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export MAX_ITERS="${MAX_ITERS:-12000}" INIT_CHECKPOINT="$BASELINE_CHECKPOINT"
export DINO_R50_CHECKPOINT="${DINO_R50_CHECKPOINT:?set DINO_R50_CHECKPOINT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config="${CONFIG_PATH:-$PWD/configs/dino_dwd_full.py}"
work_dir="$PWD/outputs/dino-moe-experts-${CAMPAIGN_ID}/${TRAIN_DOMAIN}"
train_ann="$DWD_ROOT/$TRAIN_DOMAIN/${TRAIN_DOMAIN}_train.coco.json"
for path in "$DWD_ROOT/split_manifest.json" "$train_ann" "$BASELINE_CHECKPOINT" "$DINO_R50_CHECKPOINT" "$config"; do test -s "$path"; done
test ! -e "$work_dir"
mkdir -p "$work_dir"

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("cuda_device=", torch.cuda.get_device_name(0), "capability=", torch.cuda.get_device_capability(0))
print("cuda_smoke=", (torch.arange(4, device="cuda") * 2).cpu().tolist())
CUDA_PREFLIGHT
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID domain=$TRAIN_DOMAIN updates=$MAX_ITERS start=$(date -Is)"
sha256sum "$config" "$DWD_ROOT/split_manifest.json" "$train_ann" "$BASELINE_CHECKPOINT"
mim train mmdet "$config" --work-dir "$work_dir"
checkpoint="$work_dir/iter_${MAX_ITERS}.pth"
test -s "$checkpoint"
sha256sum "$checkpoint"
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID domain=$TRAIN_DOMAIN finished=$(date -Is)"
