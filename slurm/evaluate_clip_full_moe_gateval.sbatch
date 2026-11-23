#!/usr/bin/env bash
# Validate the frozen CLIP router + full-data experts on gate-val only.
#SBATCH --job-name=ligia-clip-moe-gateval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/clip-moe-gateval-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${BASELINE_JOB_ID:?submit with BASELINE_JOB_ID}"
: "${EXPERTS_JOB_ID:?submit with EXPERTS_JOB_ID for the completed expert array}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DWD_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export CLIP_RN101_CHECKPOINT="${CLIP_RN101_CHECKPOINT:-$PWD/checkpoints/RN101.pt}"
export TRAIN_DOMAIN=day_clear
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

baseline_config="$PWD/configs/vltdet_r101_clip_dayclear_5000_640_30000.py"
baseline_checkpoint="$PWD/outputs/vitdet-r101-dayclear-full-640-30000-${BASELINE_JOB_ID}/iter_30000.pth"
expert_root="$PWD/outputs/vitdet-r101-clip-full-experts-${EXPERTS_JOB_ID}"
gate_checkpoint="${GATE_CHECKPOINT:-$PWD/outputs/domain-gate-clip-rn101-15672/domain_gate_linear.pt}"
spec_dir="$PWD/outputs/clip-full-dayclear-${BASELINE_JOB_ID}"
models_json="$spec_dir/moe_models.json"
route_map_json="$spec_dir/moe_route_map.json"
output_dir="$spec_dir/gateval-${SLURM_JOB_ID}"
mkdir -p "$spec_dir"
test ! -e "$output_dir"
for path in "$DWD_ROOT/split_manifest.json" "$baseline_config" "$baseline_checkpoint" "$gate_checkpoint" "$CLIP_RN101_CHECKPOINT"; do
  test -s "$path"
done
for domain in day_clear day_foggy dusk_rainy night_clear night_rainy; do
  test -s "$DWD_ROOT/$domain/${domain}_gate_val.coco.json"
done
for domain in day_foggy dusk_rainy night_clear night_rainy; do
  test -s "$expert_root/$domain/best_checkpoint_path.txt"
done

.venv-ligia/bin/python - "$models_json" "$route_map_json" "$baseline_config" "$baseline_checkpoint" "$expert_root" <<'PYMODELS'
import json, sys
from pathlib import Path
models_path, route_path, config, baseline, expert_root = map(Path, sys.argv[1:])
models = {"baseline_day_clear": {"config": str(config), "checkpoint": str(baseline)}}
route = {"day_clear": "baseline_day_clear"}
for domain in ("day_foggy", "dusk_rainy", "night_clear", "night_rainy"):
    model_name = f"clip_expert_{domain}"
    selected = Path(expert_root / domain / "best_checkpoint_path.txt").read_text().strip()
    if not Path(selected).is_file():
        raise FileNotFoundError(selected)
    models[model_name] = {"config": str(config), "checkpoint": selected}
    route[domain] = model_name
models_path.write_text(json.dumps(models, indent=2) + "\n")
route_path.write_text(json.dumps(route, indent=2) + "\n")
PYMODELS

.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.arange(4, device="cuda")
print("cuda_device=", torch.cuda.get_device_name(0),
      "capability=", torch.cuda.get_device_capability(0),
      "smoke=", (x * 2).cpu().tolist())
CUDA_PREFLIGHT
echo "job=$SLURM_JOB_ID baseline_job=$BASELINE_JOB_ID experts_job=$EXPERTS_JOB_ID scope=gate_val start=$(date -Is)"
sha256sum "$PWD/scripts/evaluate_moe_gate.py" "$baseline_config" "$models_json" "$route_map_json" \
  "$gate_checkpoint" "$CLIP_RN101_CHECKPOINT" "$DWD_ROOT/split_manifest.json" \
  "$baseline_checkpoint" \
  "$expert_root"/*/best_checkpoint_path.txt \
  "$DWD_ROOT"/*/*_gate_val.coco.json
.venv-ligia/bin/python scripts/evaluate_moe_gate.py \
  --annotations-root "$DWD_ROOT" --split gate_val \
  --gate-checkpoint "$gate_checkpoint" --clip-checkpoint "$CLIP_RN101_CHECKPOINT" \
  --models-json "$models_json" --route-map-json "$route_map_json" \
  --output-dir "$output_dir"
test -s "$output_dir/moe_evaluation.md"
test -s "$output_dir/moe_evaluation.json"
sha256sum "$output_dir/moe_evaluation.md" "$output_dir/moe_evaluation.json"
echo "job=$SLURM_JOB_ID gateval_finished=$(date -Is) review_before_test=1"
