#!/usr/bin/env bash
# Full DayClear CLIP baseline using the existing detector config unchanged.
#SBATCH --job-name=ligia-clip-full-dayclear
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/clip-full-dayclear-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export CLIP_RN101_CHECKPOINT="${CLIP_RN101_CHECKPOINT:-$PWD/checkpoints/RN101.pt}"
export TRAIN_DOMAIN=day_clear MAX_ITERS=30000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

config="$PWD/configs/vltdet_r101_clip_dayclear_5000_640_30000.py"
work_dir="$PWD/outputs/vitdet-r101-dayclear-full-640-30000-${SLURM_JOB_ID}"
train_ann="$DWD_ROOT/day_clear/day_clear_train.coco.json"
val_ann="$DWD_ROOT/day_clear/day_clear_val.coco.json"
for path in "$config" "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$CLIP_RN101_CHECKPOINT"; do
  test -s "$path"
done
test ! -e "$work_dir"
mkdir -p "$work_dir"

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.arange(4, device="cuda")
print("cuda_device=", torch.cuda.get_device_name(0),
      "capability=", torch.cuda.get_device_capability(0),
      "smoke=", (x * 2).cpu().tolist())
CUDA_PREFLIGHT

echo "job=$SLURM_JOB_ID node=$(hostname) domain=day_clear train_images=19095 modelval_images=100 updates=$MAX_ITERS resolution=640 batch=1 seed=1337 start=$(date -Is)"
sha256sum "$config" "$PWD/configs/vltdet_r101_clip_vitdet_smoke.py" \
  "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$CLIP_RN101_CHECKPOINT"
mim train mmdet "$config" --work-dir "$work_dir"
checkpoint="$work_dir/iter_${MAX_ITERS}.pth"
test -s "$checkpoint"
echo "$checkpoint" > "$work_dir/final_checkpoint_path.txt"
sha256sum "$checkpoint" "$work_dir/final_checkpoint_path.txt"
echo "job=$SLURM_JOB_ID baseline_finished=$(date -Is)"
