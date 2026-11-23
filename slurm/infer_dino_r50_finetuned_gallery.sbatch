#!/usr/bin/env bash
# Qualitative-only inference over the frozen shared gallery image IDs.
#SBATCH --job-name=ligia-gallery-dino-ft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/gallery-dino-ft-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv-ligia/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export DWD_ROOT="${DINO_SPLIT_ROOT:-$PWD/data/raw/dwd/full-train-frozen-eval-20261005}"
export DINO_R50_CHECKPOINT="${DINO_R50_CHECKPOINT:?set the external ResNet-50 initialization checkpoint}"
export TRAIN_DOMAIN=day_clear INIT_CHECKPOINT=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${DINO_FINETUNED_CHECKPOINT_MAP:?set a JSON mapping the four target domains to external fine-tuned checkpoints}"

samples="$PWD/experiments/dino-r50-moe/galerias"
output="$PWD/outputs/dino-r50-finetuned-gallery-${SLURM_JOB_ID}"
for path in "$DWD_ROOT/split_manifest.json" "$DINO_FINETUNED_CHECKPOINT_MAP" \
            "$PWD/configs/dino_dwd_full.py" \
            "$samples/por-dominio/samples.json" \
            "$samples/por-categoria/samples.json"; do
  test -s "$path" || { echo "Missing input: $path" >&2; exit 3; }
done
.venv-ligia/bin/python - <<'CUDA_PREFLIGHT'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, f"expected one GPU, got {torch.cuda.device_count()}"
print("cuda_device=", torch.cuda.get_device_name(0),
      "capability=", torch.cuda.get_device_capability(0))
CUDA_PREFLIGHT
.venv-ligia/bin/python scripts/infer_dino_r50_finetuned_gallery.py \
  --annotations-root "$DWD_ROOT" \
  --sample-manifests "$samples/por-dominio/samples.json" \
                    "$samples/por-categoria/samples.json" \
  --config "$PWD/configs/dino_dwd_full.py" \
  --checkpoint-map "$DINO_FINETUNED_CHECKPOINT_MAP" --output-dir "$output"
