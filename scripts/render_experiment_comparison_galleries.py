#!/usr/bin/env python3
"""Render shared-sample, qualitative comparison galleries for CLIP and DINO-R50."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from PIL import Image

from render_dino_moe_best_categories import CLASSES, DOMAINS, draw_panel, load_predictions

EXPERIMENTS = {
    "clip-full-baseline": "CLIP Full DayClear baseline",
    "clip-full-moe": "CLIP Full DayClear routed MoE",
    "dino-r50-finetuned": "DINO+ResNet50 fine-tuned experts",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--sample-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--dino-original-dir", type=Path, required=True)
    parser.add_argument("--clip-test-dir", type=Path, required=True)
    parser.add_argument("--dino-finetuned-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=20)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    prediction_paths = {
        "dino-reference": lambda domain: args.dino_original_dir / f"moe_predictions_{domain}.json",
        "clip-full-baseline": lambda domain: args.clip_test_dir / f"baseline_predictions_{domain}.json",
        "clip-full-moe": lambda domain: args.clip_test_dir / f"moe_predictions_{domain}.json",
        "dino-r50-finetuned": lambda domain: args.dino_finetuned_dir / f"dino_r50_finetuned_predictions_{domain}.json",
    }
    pred_cache = {}
    pred_cache["dino-reference"] = {
        d: load_predictions(prediction_paths["dino-reference"](d), args.score_threshold)
        for d in DOMAINS
    }
    for key, path_fn in prediction_paths.items():
        if key == "dino-reference":
            continue
        pred_cache[key] = {}
        for domain in DOMAINS:
            if key == "dino-r50-finetuned" and domain == "day_clear":
                # DayClear remains the original baseline route in the fine-tuned MoE.
                pred_cache[key][domain] = pred_cache["dino-reference"][domain]
            else:
                pred_cache[key][domain] = load_predictions(path_fn(domain), args.score_threshold)

    annotation_cache = {}
    for manifest_path in args.sample_manifests:
        selection = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = selection["samples"]
        gallery_name = "balanced-50" if manifest_path.parent.name in ("balanced-50", "por-dominio") else "best-categories-50"
        if len(samples) != 50:
            raise ValueError(f"expected 50 samples in {manifest_path}, got {len(samples)}")
        actual_domains = Counter(row["domain"] for row in samples)
        if gallery_name == "balanced-50" and actual_domains != Counter({d: 10 for d in DOMAINS}):
            raise ValueError(f"balanced gallery must have 10/domain; got {actual_domains}")
        selected_keys = [(row["domain"], int(row["image_id"])) for row in samples]
        if len(set(selected_keys)) != 50:
            raise ValueError(f"gallery does not contain 50 unique images: {manifest_path}")

        for experiment, title in EXPERIMENTS.items():
            out_dir = args.output_root / experiment / gallery_name
            out_dir.mkdir(parents=True)
            rows_json, figures = [], []
            for rank, row in enumerate(samples, 1):
                domain, image_id, focus = row["domain"], int(row["image_id"]), row["focus"]
                ann_path = args.annotations_root / domain / f"{domain}_test.coco.json"
                if domain not in annotation_cache:
                    annotation_cache[domain] = json.loads(ann_path.read_text(encoding="utf-8"))
                data = annotation_cache[domain]
                categories = {int(x["id"]): x["name"] for x in data["categories"]}
                image_info = next((item for item in data["images"] if int(item["id"]) == image_id), None)
                if image_info is None:
                    raise ValueError(f"image {domain}/{image_id} missing from {ann_path}")
                image_path = Path(image_info["file_name"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                gt = [a for a in data["annotations"]
                      if int(a["image_id"]) == image_id and not a.get("iscrowd", 0)]
                reference = pred_cache["dino-reference"][domain].get(image_id, [])
                target = pred_cache[experiment][domain].get(image_id, [])
                image = Image.open(image_path).convert("RGB")
                panels = [
                    draw_panel(image, f"Ground truth · {domain} · id={image_id}", gt,
                               categories, True, focus, args.max_boxes),
                    draw_panel(image, "DINO original MoE · reference", reference,
                               categories, False, focus, args.max_boxes),
                    draw_panel(image, title, target, categories, False, focus, args.max_boxes),
                ]
                canvas = Image.new("RGB", (panels[0].width * 3, panels[0].height), "white")
                for index, panel in enumerate(panels):
                    canvas.paste(panel, (index * panel.width, 0))
                filename = f"{rank:02d}_{focus}_{domain}_id{image_id}.png"
                canvas.save(out_dir / filename, optimize=True)
                rows_json.append({
                    "rank": rank, "focus": focus, "domain": domain, "image_id": image_id,
                    "image_file": image_path.name, "figure": filename,
                    "reference": "DINO original routed MoE predictions",
                    "experiment": experiment,
                })
                figures.append(f'<figure><img src="{escape(filename)}"><figcaption>#{rank} '
                               f'{escape(focus)} · {escape(domain)} · image {image_id}</figcaption></figure>')

            summary = {
                "qualitative_only": True,
                "source_selection_manifest": str(manifest_path.resolve()),
                "split": "frozen test sample IDs; qualitative display only",
                "selection_reused_exactly": True,
                "experiment": experiment,
                "reference": "DINO original routed MoE",
                "score_threshold": args.score_threshold,
                "rendered_total": len(rows_json),
                "counts_by_domain": dict(Counter(x["domain"] for x in rows_json)),
                "counts_by_focus": dict(Counter(x["focus"] for x in rows_json)),
                "samples": rows_json,
            }
            (out_dir / "samples.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            page = (f'<!doctype html><meta charset="utf-8"><title>{escape(title)}</title>\n'
                    '<style>body{font-family:sans-serif;background:#eef2ef;margin:24px}h1{color:#17251f}'
                    'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:18px}'
                    'figure{margin:0;background:white;padding:8px;box-shadow:0 2px 8px #0002}'
                    'img{width:100%;display:block}figcaption{padding:8px 2px;color:#17251f}</style>\n'
                    f'<h1>{escape(title)} · {gallery_name}</h1><p>50 identical held-out samples; '
                    'qualitative only. Panels: ground truth, original DINO MoE reference, target experiment.</p><main>\n'
                    + "\n".join(figures) + "\n</main>\n")
            (out_dir / "index.html").write_text(page, encoding="utf-8")
            print(f"rendered={experiment}/{gallery_name} total={len(rows_json)} domains={dict(actual_domains)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
