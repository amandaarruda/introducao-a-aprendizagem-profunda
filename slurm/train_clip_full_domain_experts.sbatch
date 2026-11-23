#!/usr/bin/env bash
# Four CLIP experts initialized from the full DayClear baseline; serialized.
#SBATCH --job-name=ligia-clip-experts
#SBATCH --array=0-3%1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/clip-experts-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${BASELINE_JOB_ID:?submit with BASELINE_JOB_ID}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export CLIP_RN101_CHECKPOINT="${CLIP_RN101_CHECKPOINT:-$PWD/checkpoints/RN101.pt}"
export MAX_ITERS=12000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

domains=(day_foggy dusk_rainy night_clear night_rainy)
index="${SLURM_ARRAY_TASK_ID:?}"
test "$index" -ge 0 && test "$index" -lt "${#domains[@]}"
export TRAIN_DOMAIN="${domains[$index]}"
config="$PWD/configs/vltdet_r101_clip_dayclear_5000_640_30000.py"
baseline="$PWD/outputs/vitdet-r101-dayclear-full-640-30000-${BASELINE_JOB_ID}/iter_30000.pth"
expert_root="${EXPERTS_ROOT:-$PWD/outputs/vitdet-r101-clip-full-experts-${SLURM_ARRAY_JOB_ID}}"
work_dir="$expert_root/$TRAIN_DOMAIN"
recovery_source="$PWD/outputs/vitdet-r101-clip-full-experts-${BASELINE_JOB_ID}/$TRAIN_DOMAIN"
train_ann="$DWD_ROOT/$TRAIN_DOMAIN/${TRAIN_DOMAIN}_train.coco.json"
val_ann="$DWD_ROOT/$TRAIN_DOMAIN/${TRAIN_DOMAIN}_val.coco.json"
for path in "$config" "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$CLIP_RN101_CHECKPOINT" "$baseline"; do
  test -s "$path"
done
mkdir -p "$work_dir"

# Resume only from this expert's own work directory. The recovery source from older runs
# may lack optimizer state; all new checkpoints below include it.
if [[ -s "$work_dir/last_checkpoint" ]]; then
  resume_checkpoint="$(cat "$work_dir/last_checkpoint")"
  test -s "$resume_checkpoint"
  resume=True
  echo "resume_checkpoint=$resume_checkpoint"
else
  resume_checkpoint="$baseline"
  resume=False
fi

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import os, torch
assert torch.cuda.is_available(), "CUDA unavailable"
props = torch.cuda.get_device_properties(0)
min_gpu_gib = float(os.environ.get("MIN_GPU_GIB", "20"))
assert props.total_memory >= min_gpu_gib * 1024**3, f"Need >={min_gpu_gib:g} GiB GPU, got {props.total_memory / 1024**3:.1f} GiB"
x = torch.arange(4, device="cuda")
print("cuda_device=", torch.cuda.get_device_name(0),
      "memory_gib=", round(props.total_memory / 1024**3, 1),
      "capability=", torch.cuda.get_device_capability(0),
      "smoke=", (x * 2).cpu().tolist())
CUDA_PREFLIGHT

echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname) domain=$TRAIN_DOMAIN train=full-frozen-reserves updates=$MAX_ITERS resolution=640 batch=1 init=$resume_checkpoint resume=$resume start=$(date -Is)"
sha256sum "$config" "$PWD/configs/vltdet_r101_clip_vitdet_smoke.py" \
  "$DWD_ROOT/split_manifest.json" "$train_ann" "$val_ann" "$baseline" "$CLIP_RN101_CHECKPOINT"
mim train mmdet "$config" --work-dir "$work_dir" \
  --cfg-options "load_from=$resume_checkpoint" "resume=$resume" "default_hooks.checkpoint.save_optimizer=True"
final_checkpoint="$work_dir/iter_${MAX_ITERS}.pth"
test -s "$final_checkpoint"
shopt -s nullglob
best_checkpoints=("$work_dir"/best_coco_bbox_mAP_50_iter_*.pth)
test "${#best_checkpoints[@]}" -ge 1
test "${#best_checkpoints[@]}" -eq 1
printf '%s\n' "${best_checkpoints[0]}" > "$work_dir/best_checkpoint_path.txt"
sha256sum "$final_checkpoint" "${best_checkpoints[@]}" "$work_dir/best_checkpoint_path.txt"
echo "job=$SLURM_JOB_ID expert_finished=$(date -Is) domain=$TRAIN_DOMAIN"
