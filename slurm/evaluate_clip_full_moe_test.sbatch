#!/usr/bin/env bash
# Single final CLIP baseline vs frozen routed MoE evaluation on test splits.
# Submit only after reviewing the gate-val report and freezing the model/map.
#SBATCH --job-name=ligia-clip-moe-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/clip-moe-test-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${BASELINE_JOB_ID:?submit with BASELINE_JOB_ID}"
: "${GATEVAL_JOB_ID:?set the completed gate-val Slurm job ID}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export CLIP_RN101_CHECKPOINT="${CLIP_RN101_CHECKPOINT:-$PWD/checkpoints/RN101.pt}"
export TRAIN_DOMAIN=day_clear
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

baseline_config="$PWD/configs/vltdet_r101_clip_dayclear_5000_640_30000.py"
baseline_checkpoint="$PWD/outputs/vitdet-r101-dayclear-full-640-30000-${BASELINE_JOB_ID}/iter_30000.pth"
gate_checkpoint="${GATE_CHECKPOINT:-$PWD/outputs/domain-gate-clip-rn101-15672/domain_gate_linear.pt}"
spec_dir="$PWD/outputs/clip-full-dayclear-${BASELINE_JOB_ID}"
models_json="$spec_dir/moe_models.json"
route_map_json="$spec_dir/moe_route_map.json"
gateval_dir="$spec_dir/gateval-${GATEVAL_JOB_ID}"
output_dir="$spec_dir/test-final-${SLURM_JOB_ID}"
freeze_file="$spec_dir/test-freeze-${GATEVAL_JOB_ID}.sha256"

test -s "$gateval_dir/moe_evaluation.md"
test -s "$gateval_dir/moe_evaluation.json"
test ! -e "$output_dir"
for path in "$DWD_ROOT/split_manifest.json" "$baseline_config" "$baseline_checkpoint" \
            "$gate_checkpoint" "$CLIP_RN101_CHECKPOINT" "$models_json" "$route_map_json"; do
  test -s "$path"
done
for domain in day_clear day_foggy dusk_rainy night_clear night_rainy; do
  test -s "$DWD_ROOT/$domain/${domain}_test.coco.json"
done

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.arange(4, device="cuda")
print("cuda_device=", torch.cuda.get_device_name(0),
      "capability=", torch.cuda.get_device_capability(0),
      "smoke=", (x * 2).cpu().tolist())
CUDA_PREFLIGHT

# Persist or verify the exact model, route, split, gate and checkpoint hashes before test inference.
freeze_tmp=$(mktemp "$spec_dir/.test-freeze-${GATEVAL_JOB_ID}.XXXXXX")
sha256sum "$models_json" "$route_map_json" "$baseline_config" "$baseline_checkpoint" \
  "$gate_checkpoint" "$CLIP_RN101_CHECKPOINT" "$DWD_ROOT/split_manifest.json" \
  "$PWD/scripts/evaluate_moe_gate.py" \
  "$DWD_ROOT"/*/*_test.coco.json > "$freeze_tmp"
if test -e "$freeze_file"; then
  if ! cmp -s "$freeze_tmp" "$freeze_file"; then
    echo "Frozen test inputs differ from the previously recorded hashes" >&2
    rm -f "$freeze_tmp"
    exit 3
  fi
  rm -f "$freeze_tmp"
else
  mv "$freeze_tmp" "$freeze_file"
  chmod a-w "$freeze_file"
fi
echo "job=$SLURM_JOB_ID baseline_job=$BASELINE_JOB_ID gateval_job=$GATEVAL_JOB_ID scope=test start=$(date -Is)"
cat "$freeze_file"
.venv-ligia/bin/python scripts/evaluate_moe_gate.py \
  --annotations-root "$DWD_ROOT" --split test \
  --gate-checkpoint "$gate_checkpoint" --clip-checkpoint "$CLIP_RN101_CHECKPOINT" \
  --models-json "$models_json" --route-map-json "$route_map_json" \
  --output-dir "$output_dir"
test -s "$output_dir/moe_evaluation.md"
test -s "$output_dir/moe_evaluation.json"
sha256sum "$output_dir/moe_evaluation.md" "$output_dir/moe_evaluation.json"
echo "job=$SLURM_JOB_ID scope=test finished=$(date -Is)"
