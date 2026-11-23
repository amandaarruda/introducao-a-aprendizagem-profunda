#!/usr/bin/env python3
"""Run a frozen image-level domain gate and evaluate routed detectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")
CLASSES = ("bus", "bike", "car", "motor", "person", "rider", "truck")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gate_encoder(clip_checkpoint: Path, device):
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(f"CLIP RN101 checkpoint missing: {clip_checkpoint}")
    model, _, transform = open_clip.create_model_and_transforms(
        "RN101", pretrained=None, pretrained_hf=False)
    state = torch.load(clip_checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        state = state.state_dict()
    state = state.get("state_dict", state)
    current = model.state_dict()
    state = {key: value for key, value in state.items() if key in current}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"incomplete CLIP load: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, transform


def gate_embedding(encoder, images):
    return F.normalize(encoder.encode_image(images).float(), dim=1)


def evaluate_coco(ann_path: Path, predictions: list[dict], output_path: Path):
    gt = COCO(str(ann_path))
    output_path.write_text(json.dumps(predictions), encoding="utf-8")
    dt = gt.loadRes(str(output_path))
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(im["id"] for im in gt.dataset["images"])
    evaluator.params.catIds = sorted(cat["id"] for cat in gt.dataset["categories"])
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = evaluator.stats.tolist()
    precision = evaluator.eval["precision"]
    categories = gt.loadCats(evaluator.params.catIds)
    per_class = {}
    for index, category in enumerate(categories):
        arr = precision[:, :, index, 0, -1]
        valid = arr[arr > -1]
        at50 = precision[0, :, index, 0, -1]
        at50 = at50[at50 > -1]
        per_class[category["name"]] = {
            "AP_50_95": float(np.mean(valid)) if valid.size else None,
            "AP_50": float(np.mean(at50)) if at50.size else None,
            "ground_truth": int(sum(
                1 for ann in gt.dataset["annotations"]
                if ann["category_id"] == category["id"] and not ann.get("iscrowd", 0)
            )),
        }
    result = {
        "AP_50_95": float(values[0]), "AP_50": float(values[1]),
        "AP_75": float(values[2]), "AP_small": float(values[3]),
        "AP_medium": float(values[4]), "AP_large": float(values[5]),
        "per_class": per_class,
    }
    return result


def append_coco_predictions(target: list[dict], sample, image_id: int, category_ids: dict[str, int], score_threshold: float = 0.05):
    pred = sample.pred_instances
    boxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = map(float, box)
        class_name = CLASSES[int(label)]
        target.append({
            "image_id": image_id,
            "category_id": category_ids[class_name],
            "bbox": [x1, y1, max(x2-x1, 0.0), max(y2-y1, 0.0)],
            "score": float(score),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--split", choices=("gate_val", "test"), default="gate_val")
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--models-json", type=Path, required=True,
                        help="mapping model name to {config, checkpoint}")
    parser.add_argument("--route-map-json", type=Path, required=True,
                        help="mapping predicted domain to selected model name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-all-models", action="store_true",
                        help="also score every frozen detector in models-json on every image")
    args = parser.parse_args()
    for path in (args.gate_checkpoint, args.models_json, args.route_map_json,
                 args.annotations_root / "split_manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for detector evaluation")
    os.environ.setdefault("TRAIN_DOMAIN", "day_clear")
    model_specs = json.loads(args.models_json.read_text(encoding="utf-8"))
    route_map = json.loads(args.route_map_json.read_text(encoding="utf-8"))
    if set(route_map) != set(DOMAINS):
        raise ValueError(f"route map must cover exactly {DOMAINS}")
    if "baseline_day_clear" not in model_specs:
        raise ValueError("baseline_day_clear is required for paired comparison")
    for name in route_map.values():
        if name not in model_specs:
            raise ValueError(f"route map references missing model: {name}")
    for name, spec in model_specs.items():
        for key in ("config", "checkpoint"):
            if not Path(spec[key]).is_file():
                raise FileNotFoundError(f"{name}.{key}: {spec[key]}")

    payload = torch.load(args.gate_checkpoint, map_location="cpu")
    gate_domains = tuple(payload["domains"])
    encoder_kind = payload.get("encoder_kind", "clip-rn101")
    if encoder_kind != "clip-rn101" or gate_domains != DOMAINS or payload["feature_dim"] != 512:
        raise ValueError(f"unexpected CLIP gate schema: domains={gate_domains}, dim={payload['feature_dim']}, encoder={encoder_kind}")
    gate_head = nn.Linear(512, len(DOMAINS))
    gate_head.load_state_dict(payload["state_dict"])
    gate_head.eval().to(device)
    encoder, transform = load_gate_encoder(args.clip_checkpoint, device)
    if sha256(args.clip_checkpoint) != payload.get("encoder_checkpoint_sha256", payload.get("clip_checkpoint_sha256")):
        raise ValueError("gate encoder checkpoint hash differs from the weights used to train the router")

    torch.cuda.empty_cache()
    detectors = {}
    for name in sorted({"baseline_day_clear", *route_map.values()}):
        spec = model_specs[name]
        cfg = Config.fromfile(spec["config"])
        if hasattr(cfg.model.test_cfg, "rcnn"):
            cfg.model.test_cfg.rcnn.score_thr = 0.05
        elif isinstance(cfg.model.test_cfg, dict):
            cfg.model.test_cfg.setdefault("max_per_img", 100)
        detectors[name] = init_detector(cfg, spec["checkpoint"], device=str(device))
        detectors[name].eval()
    started = time.time()
    confusion = [[0 for _ in DOMAINS] for _ in DOMAINS]
    routed_counts = Counter()
    report = {
        "scope": args.split,
        "annotation_note": "gate_val used for gate validation/selection; test allowed only after model and route-map freeze",
        "score_threshold": 0.05,
        "route_map": route_map,
        "gate_checkpoint": str(args.gate_checkpoint.resolve()),
        "gate_checkpoint_sha256": sha256(args.gate_checkpoint),
        "gate_encoder": encoder_kind,
        "models": {},
        "domains": {},
    }
    for domain in DOMAINS:
        ann_path = args.annotations_root / domain / f"{domain}_{args.split}.coco.json"
        coco_data = json.loads(ann_path.read_text(encoding="utf-8"))
        baseline_results, moe_results = [], []
        all_model_results = {name: [] for name in model_specs} if args.evaluate_all_models else {}
        route_records = []
        category_ids = {cat["name"]: int(cat["id"]) for cat in coco_data["categories"]}
        for image in coco_data["images"]:
            image_path = Path(image["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as im:
                clip_image = transform(im.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                embedding = gate_embedding(encoder, clip_image)
                probabilities = gate_head(embedding).softmax(dim=1)[0]
                pred_index = int(probabilities.argmax().item())
            predicted_domain = DOMAINS[pred_index]
            selected_name = route_map[predicted_domain]
            confusion[DOMAINS.index(domain)][pred_index] += 1
            routed_counts[selected_name] += 1
            samples = {
                "baseline_day_clear": inference_detector(
                    detectors["baseline_day_clear"], str(image_path))
            }
            if args.evaluate_all_models:
                for model_name in model_specs:
                    if model_name != "baseline_day_clear":
                        samples[model_name] = inference_detector(
                            detectors[model_name], str(image_path))
            selected_sample = samples.get(selected_name)
            if selected_sample is None:
                selected_sample = inference_detector(detectors[selected_name], str(image_path))
                samples[selected_name] = selected_sample
            if args.evaluate_all_models:
                for model_name, sample in samples.items():
                    append_coco_predictions(
                        all_model_results[model_name], sample, int(image["id"]), category_ids)
            baseline_sample = samples["baseline_day_clear"]
            append_coco_predictions(baseline_results, baseline_sample, int(image["id"]), category_ids)
            append_coco_predictions(moe_results, selected_sample, int(image["id"]), category_ids)
            route_records.append({
                "image_id": int(image["id"]), "path": str(image_path),
                "true_domain": domain, "predicted_domain": predicted_domain,
                "selected_model": selected_name,
                "probabilities": {d: float(probabilities[i].item()) for i,d in enumerate(DOMAINS)},
            })
            if len(route_records) % 25 == 0:
                print(f"domain={domain} images={len(route_records)}/{len(coco_data['images'])} routed={selected_name}", flush=True)
        moe_path = args.output_dir / f"moe_predictions_{domain}.json"
        baseline_path = args.output_dir / f"baseline_predictions_{domain}.json"
        route_path = args.output_dir / f"routes_{domain}.json"
        baseline_metrics = evaluate_coco(ann_path, baseline_results, baseline_path)
        moe_metrics = evaluate_coco(ann_path, moe_results, moe_path)
        standalone_metrics = {}
        if args.evaluate_all_models:
            for name, predictions in all_model_results.items():
                model_path = args.output_dir / f"predictions_{name}_{domain}.json"
                standalone_metrics[name] = evaluate_coco(ann_path, predictions, model_path)
                standalone_metrics[name]["predictions"] = str(model_path.resolve())
        route_path.write_text(json.dumps(route_records, indent=2) + "\n", encoding="utf-8")
        report["domains"][domain] = {
            "images": len(coco_data["images"]),
            "baseline": baseline_metrics,
            "moe": moe_metrics,
            "standalone_models": standalone_metrics,
            "predicted_domain_counts": dict(Counter(r["predicted_domain"] for r in route_records)),
            "selected_model_counts": dict(Counter(r["selected_model"] for r in route_records)),
            "predictions": str(moe_path.resolve()),
            "routes": str(route_path.resolve()),
        }
        print(json.dumps({"domain":domain,"baseline_mAP50":baseline_metrics["AP_50"],
                          "moe_mAP50":moe_metrics["AP_50"],
                          "standalone_mAP50":{k:v["AP_50"] for k,v in standalone_metrics.items()},
                          "selected_model_counts":report["domains"][domain]["selected_model_counts"]}), flush=True)
    gate_total = sum(sum(row) for row in confusion)
    report["gate"] = {
        "domain_accuracy": sum(confusion[i][i] for i in range(len(DOMAINS))) / gate_total,
        "confusion_matrix_rows_true_cols_predicted": confusion,
        "domain_order": list(DOMAINS),
        "routed_image_counts_by_model": dict(routed_counts),
    }
    report["models"] = {
        name: {"config": str(Path(spec["config"]).resolve()),
              "checkpoint": str(Path(spec["checkpoint"]).resolve()),
              "checkpoint_sha256": sha256(Path(spec["checkpoint"]))}
        for name, spec in model_specs.items()
    }
    report["elapsed_seconds"] = time.time() - started
    result_path = args.output_dir / "moe_evaluation.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Routed detector evaluation — {args.split}", "",
        f"- Gate domain accuracy: {100*report['gate']['domain_accuracy']:.2f}%",
        f"- Score threshold: 0.05", f"- Route map: `{json.dumps(route_map, sort_keys=True)}`", "",
        "| Domain | Baseline AP50 | MoE AP50 | Baseline AP@[.5:.95] | MoE AP@[.5:.95] |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        x=report["domains"][domain]
        lines.append(f"| {domain} | {100*x['baseline']['AP_50']:.2f}% | {100*x['moe']['AP_50']:.2f}% | {100*x['baseline']['AP_50_95']:.2f}% | {100*x['moe']['AP_50_95']:.2f}% |")
    lines += ["", "## Domain gate confusion matrix (rows true, columns predicted)", "",
              "| True \\ Predicted | " + " | ".join(DOMAINS) + " |", "|---|"+"---:|"*len(DOMAINS)]
    for i,domain in enumerate(DOMAINS):
        lines.append("| "+domain+" | "+" | ".join(map(str,confusion[i]))+" |")
    lines += ["", "The `test` split is read only when `--split test` is explicitly passed. Freeze the gate, route map, and model list before using it."]
    if args.evaluate_all_models:
        model_names = list(model_specs)
        lines += ["", "## Standalone frozen detector matrix", "",
                  "| Domain | " + " | ".join(model_names) + " |", "|---|" + "---:|"*len(model_names)]
        for domain in DOMAINS:
            standalone = report["domains"][domain]["standalone_models"]
            lines.append("| " + domain + " | " + " | ".join(
                f"{100*standalone[name]['AP_50']:.2f}%" for name in model_names) + " |")
    report_md=args.output_dir/"moe_evaluation.md"
    report_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"MOE_REPORT={report_md.resolve()}",flush=True)
    print(f"MOE_RESULTS={result_path.resolve()}",flush=True)


if __name__ == "__main__":
    main()
