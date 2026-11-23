#!/usr/bin/env python3
"""Infer only the frozen DINO-R50 gallery samples with fine-tuned experts.

This is qualitative-only: it performs no metric calculation and does not
change the previously selected checkpoints or route map.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmdet.apis import inference_detector, init_detector

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")
CLASSES = ("bus", "bike", "car", "motor", "person", "rider", "truck")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--sample-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    checkpoint_map = json.loads(args.checkpoint_map.read_text(encoding="utf-8"))
    required = set(DOMAINS) - {"day_clear"}
    if set(checkpoint_map) != required:
        raise ValueError(f"checkpoint map must contain exactly {sorted(required)}")

    selected: dict[str, set[int]] = defaultdict(set)
    manifests = []
    for path in args.sample_manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({"path": str(path.resolve()), "samples": manifest["samples"]})
        for sample in manifest["samples"]:
            selected[sample["domain"]].add(int(sample["image_id"]))

    results = {domain: [] for domain in required}
    for domain in DOMAINS:
        if domain == "day_clear":
            continue  # The frozen DINO route keeps the DayClear baseline.
        cfg = Config.fromfile(str(args.config))
        cfg.train_dataloader.dataset.ann_file = str(
            args.annotations_root / domain / f"{domain}_train.coco.json")
        cfg.val_dataloader.dataset.ann_file = str(
            args.annotations_root / domain / f"{domain}_val.coco.json")
        cfg.test_dataloader.dataset.ann_file = str(
            args.annotations_root / domain / f"{domain}_test.coco.json")
        model = init_detector(cfg, str(checkpoint_map[domain]), device="cuda:0")
        ann_path = args.annotations_root / domain / f"{domain}_test.coco.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = {int(row["id"]): row["name"] for row in data["categories"]}
        category_ids = {name: cid for cid, name in cats.items()}
        images = {int(row["id"]): row for row in data["images"]}
        missing = selected[domain] - images.keys()
        if missing:
            raise ValueError(f"selected images missing from {domain} test annotations: {sorted(missing)}")
        for image_id in sorted(selected[domain]):
            image_path = Path(images[image_id]["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            output = inference_detector(model, str(image_path))
            pred = output.pred_instances
            boxes = pred.bboxes.detach().cpu().tolist()
            scores = pred.scores.detach().cpu().tolist()
            labels = pred.labels.detach().cpu().tolist()
            for box, score, label in zip(boxes, scores, labels):
                score = float(score)
                if score < args.score_threshold:
                    continue
                x1, y1, x2, y2 = map(float, box)
                results[domain].append({
                    "image_id": image_id,
                    "category_id": category_ids[CLASSES[int(label)]],
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": score,
                })
            print(f"{domain}: inferred {image_id} ({image_path.name})", flush=True)
        del model
        torch.cuda.empty_cache()

    for domain, predictions in results.items():
        path = args.output_dir / f"dino_r50_finetuned_predictions_{domain}.json"
        path.write_text(json.dumps(predictions), encoding="utf-8")
    summary = {
        "qualitative_only": True,
        "split": "test (frozen qualitative sample IDs only)",
        "route": "DayClear original baseline; other domains their fine-tuned DINO-R50 expert",
        "score_threshold": args.score_threshold,
        "sample_manifests": manifests,
        "sample_counts_by_domain": {domain: len(ids) for domain, ids in selected.items()},
        "prediction_counts_by_domain": {domain: len(rows) for domain, rows in results.items()},
        "config": str(args.config.resolve()),
        "checkpoints": {domain: str(Path(path).resolve()) for domain, path in checkpoint_map.items()},
    }
    (args.output_dir / "inference_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
