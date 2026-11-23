#!/usr/bin/env bash
#SBATCH --job-name=dino-r50-bs8-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/dino-r50-finetune-debug-bs8-%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT_80

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
domain="${TRAIN_DOMAIN:?Submit exactly one TRAIN_DOMAIN per debug allocation}"
case "$domain" in day_foggy|dusk_rainy|night_clear|night_rainy) ;; *) echo "Unknown domain: $domain" >&2; exit 2;; esac
campaign_id="${SLURM_JOB_ID:?}"
run_root="$PWD/outputs/dino-r50-finetune-debug-bs8-${campaign_id}"
config="$PWD/configs/dino_dwd_finetune_bs8_fullgpu.py"
export DWD_ROOT="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export DINO_R50_CHECKPOINT="${DINO_R50_CHECKPOINT:-$PWD/checkpoints/resnet50-11ad3fa6.pth}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source .venv-ligia/bin/activate

for path in "$config" "$PWD/configs/dino_dwd_finetune.py" "$PWD/configs/dino_dwd_full.py" "$DINO_R50_CHECKPOINT" "$DWD_ROOT/split_manifest.json"; do
  test -s "$path" || { echo "Missing required input: $path" >&2; exit 3; }
done
test ! -e "$run_root" || { echo "Refusing to overwrite $run_root" >&2; exit 4; }
mkdir -p "$run_root"

nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, f"Expected one allocated GPU, got {torch.cuda.device_count()}"
props = torch.cuda.get_device_properties(0)
mem_gib = props.total_memory / 1024**3
print("cuda_device_0=", props.name, "total_memory_gib=", round(mem_gib, 2),
      "capability=", torch.cuda.get_device_capability(0))
assert "A100" in props.name, f"Refusing non-A100 allocation: {props.name}"
assert mem_gib >= 70, f"Refusing partial/MIG allocation ({mem_gib:.1f} GiB); need full A100 profile"
print("cuda_smoke_0=", (torch.arange(4, device="cuda:0") * 2).cpu().tolist())
CUDA_PREFLIGHT

checkpoint="${EXPERT_ROOT:-$PWD/outputs/dino-moe-experts-16511}/$domain/iter_12000.pth"
train_ann="$DWD_ROOT/$domain/${domain}_train.coco.json"
val_ann="$DWD_ROOT/$domain/${domain}_val.coco.json"
work_dir="$run_root/$domain/lr_8e-5_bs8"
for path in "$checkpoint" "$train_ann" "$val_ann"; do
  test -s "$path" || { echo "Missing candidate input: $path" >&2; exit 5; }
done
test ! -e "$work_dir" || { echo "Refusing to overwrite $work_dir" >&2; exit 6; }
mkdir -p "$work_dir"
sha256sum "$config" "$checkpoint" "$train_ann" "$val_ann" > "$work_dir/inputs.sha256"
echo "candidate_begin job=$campaign_id domain=$domain lr=8e-5 batch_size=8 start=$(date -Is)"
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
     --export=ALL,TRAIN_DOMAIN="$domain",FINETUNE_LR=8e-5,INIT_CHECKPOINT="$checkpoint",CANDIDATE_WORK_DIR="$work_dir" \
     bash -c 'set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; source .venv-ligia/bin/activate; mim train mmdet "$PWD/configs/dino_dwd_finetune_bs8_fullgpu.py" --work-dir "$CANDIDATE_WORK_DIR"' \
     > "$work_dir/train.log" 2>&1
best=$(find "$work_dir" -maxdepth 1 -type f -name 'best_coco_bbox_mAP_50_iter_*.pth' -print | sort | tail -n 1)
test -n "$best" && test -s "$best"
sha256sum "$best"
echo "candidate_complete job=$campaign_id domain=$domain end=$(date -Is)"
