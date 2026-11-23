#!/usr/bin/env bash
# Evaluate the frozen DINO baseline/experts with the frozen CLIP domain gate.
# Default is gate_val. Set DINO_MOE_SPLIT=test only after the model list and
# route map have been frozen. The test output directory is protected.
#SBATCH --job-name=ligia-dino-moe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-dino-moe-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DINO_SPLIT_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export DINO_R50_CHECKPOINT="${DINO_R50_CHECKPOINT:?set DINO_R50_CHECKPOINT}"
export CLIP_RN101_CHECKPOINT="${CLIP_RN101_CHECKPOINT:?set CLIP_RN101_CHECKPOINT}"
export BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT}"
export CAMPAIGN_ID="${CAMPAIGN_ID:?set CAMPAIGN_ID}"
export DINO_GATE_CHECKPOINT="${DINO_GATE_CHECKPOINT:?set DINO_GATE_CHECKPOINT}"
split="${DINO_MOE_SPLIT:-gate_val}"
root="$PWD/outputs/dino-moe-${CAMPAIGN_ID}"
expert_root="$PWD/outputs/dino-moe-experts-${CAMPAIGN_ID}"
spec_root="$root/specs"

case "$split" in
  gate_val) output="$root/gateval-${SLURM_JOB_ID}";;
  test) output="$root/test-final";;
  *) echo "DINO_MOE_SPLIT must be gate_val or test" >&2; exit 2;;
esac
for path in "$DWD_ROOT/split_manifest.json" "$BASELINE_CHECKPOINT" "$DINO_GATE_CHECKPOINT" "$CLIP_RN101_CHECKPOINT" "$DINO_R50_CHECKPOINT"; do test -s "$path"; done
for domain in day_foggy dusk_rainy night_clear night_rainy; do test -s "$expert_root/$domain/iter_12000.pth"; done
if [[ ! -e "$spec_root" ]]; then
  .venv-ligia/bin/python scripts/build_dino_moe_specs.py \
    --config "$PWD/configs/dino_dwd_full.py" --baseline "$BASELINE_CHECKPOINT" \
    --expert-root "$expert_root" --output-dir "$spec_root"
fi
models="$spec_root/models.json"
routes="$spec_root/route_map.json"
test -s "$models"; test -s "$routes"; test ! -e "$output"
mkdir -p "$root"

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("cuda_device=", torch.cuda.get_device_name(0), "capability=", torch.cuda.get_device_capability(0))
CUDA_PREFLIGHT
echo "job=$SLURM_JOB_ID scope=$split campaign=$CAMPAIGN_ID start=$(date -Is)"
sha256sum "$PWD/scripts/evaluate_moe_gate.py" "$PWD/configs/dino_dwd_full.py" \
  "$models" "$routes" "$DINO_GATE_CHECKPOINT" "$DWD_ROOT/split_manifest.json" \
  "$BASELINE_CHECKPOINT" "$expert_root"/*/iter_12000.pth
.venv-ligia/bin/python scripts/evaluate_moe_gate.py \
  --annotations-root "$DWD_ROOT" --split "$split" \
  --gate-checkpoint "$DINO_GATE_CHECKPOINT" --clip-checkpoint "$CLIP_RN101_CHECKPOINT" \
  --models-json "$models" --route-map-json "$routes" \
  --evaluate-all-models --output-dir "$output"
test -s "$output/moe_evaluation.md"
test -s "$output/moe_evaluation.json"
sha256sum "$output/moe_evaluation.md" "$output/moe_evaluation.json"
echo "job=$SLURM_JOB_ID scope=$split finished=$(date -Is)"
