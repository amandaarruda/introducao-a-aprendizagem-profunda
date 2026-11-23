#!/usr/bin/env python3
"""Render a class- and domain-balanced gallery from frozen DINO MoE outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from PIL import Image

from render_dino_moe_best_categories import (
    CLASSES, DOMAINS, draw_panel, load_predictions, matches, render_row,
)

# Exactly 10 scenes per domain and 50 scenes overall. The extra class quota is
# assigned to car because it is the dominant class and has the largest stable
# pool of high-quality matches.
QUOTAS = {
    "day_clear": {"bus": 1, "bike": 1, "car": 2, "motor": 2, "person": 1, "rider": 2, "truck": 1},
    "day_foggy": {"bus": 1, "bike": 1, "car": 1, "motor": 2, "person": 2, "rider": 2, "truck": 1},
    "dusk_rainy": {"bus": 1, "bike": 2, "car": 2, "motor": 1, "person": 2, "rider": 1, "truck": 1},
    "night_clear": {"bus": 2, "bike": 2, "car": 1, "motor": 1, "person": 1, "rider": 1, "truck": 2},
    "night_rainy": {"bus": 2, "bike": 1, "car": 2, "motor": 1, "person": 1, "rider": 1, "truck": 2},
}


def expert_row(image, domain, image_id, focus, gt, baseline, routed, selected_model, categories, max_boxes):
    title = f"focus={focus} · {domain} · image_id={image_id}"
    panels = [
        draw_panel(image, f"Ground truth · {title}", gt, categories, True, focus, max_boxes),
        draw_panel(image, "DINO baseline · DayClear", baseline, categories, False, focus, max_boxes),
        draw_panel(image, f"DINO MoE · routed={selected_model}", routed, categories, False, focus, max_boxes),
    ]
    from PIL import Image as PILImage
    row = PILImage.new("RGB", (panels[0].width * 3, panels[0].height), "white")
    for i, panel in enumerate(panels):
        row.paste(panel, (i * panel.width, 0))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    candidates = defaultdict(list)
    for domain in DOMAINS:
        ann_path = args.annotations_root / domain / f"{domain}_test.coco.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(x["id"]): x["name"] for x in data["categories"]}
        class_ids = {name: cid for cid, name in categories.items()}
        images = {int(x["id"]): x for x in data["images"]}
        gt_by_image = defaultdict(list)
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0):
                gt_by_image[int(ann["image_id"])].append(ann)
        routes = {int(x["image_id"]): x for x in json.loads(
            (args.predictions_dir / f"routes_{domain}.json").read_text(encoding="utf-8"))}
        baseline = load_predictions(args.predictions_dir / f"baseline_predictions_{domain}.json", args.score_threshold)
        model_names = {routes[i]["selected_model"] for i in routes}
        predictions = {
            name: load_predictions(args.predictions_dir / f"predictions_{name}_{domain}.json", args.score_threshold)
            for name in model_names
        }
        for image_id, gt in gt_by_image.items():
            route = routes[image_id]
            selected_model = route["selected_model"]
            routed = predictions[selected_model][image_id]
            image_path = Path(images[image_id]["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            for focus in CLASSES:
                cid = class_ids[focus]
                focus_gt = sum(a["category_id"] == cid for a in gt)
                if not focus_gt:
                    continue
                scores = matches(gt, routed, cid, args.iou_threshold)
                if not scores:
                    continue
                candidates[(domain, focus)].append({
                    "domain": domain, "focus": focus, "image_id": image_id,
                    "image_file": str(image_path), "selected_model": selected_model,
                    "gt": gt, "baseline": baseline[image_id], "routed": routed,
                    "categories": categories, "focus_gt": focus_gt,
                    "focus_matched": len(scores), "focus_recall": len(scores) / focus_gt,
                    "focus_mean_iou": sum(scores) / len(scores),
                    "focus_iou75": sum(x >= 0.75 for x in scores),
                })

    selected = []
    used = set()
    for domain in DOMAINS:
        for focus, quota in sorted(QUOTAS[domain].items(), key=lambda item: len(candidates[(domain, item[0])])):
            rows = sorted(candidates[(domain, focus)], key=lambda row: (
                -row["focus_recall"], -row["focus_mean_iou"], -row["focus_iou75"],
                -row["focus_matched"], row["image_id"]))
            for row in rows:
                key = (domain, row["image_id"])
                if key in used:
                    continue
                selected.append(row)
                used.add(key)
                if sum(x["domain"] == domain and x["focus"] == focus for x in selected) == quota:
                    break
            count = sum(x["domain"] == domain and x["focus"] == focus for x in selected)
            if count != quota:
                raise RuntimeError(f"could not fill quota {domain}/{focus}: {count}/{quota}")

    if len(selected) != 50 or len({(x["domain"], x["image_id"]) for x in selected}) != 50:
        raise RuntimeError(f"expected 50 unique scenes, got {len(selected)}")

    samples = []
    html = []
    for rank, row in enumerate(selected, 1):
        image = Image.open(row["image_file"]).convert("RGB")
        rendered = expert_row(image, row["domain"], row["image_id"], row["focus"],
                              row["gt"], row["baseline"], row["routed"],
                              row["selected_model"], row["categories"], args.max_boxes)
        filename = f"{rank:02d}_{row['focus']}_{row['domain']}_id{row['image_id']}.png"
        rendered.save(args.output_dir / filename, optimize=True)
        samples.append({
            "rank": rank, "focus": row["focus"], "domain": row["domain"],
            "image_id": row["image_id"], "image_file": Path(row["image_file"]).name,
            "selected_model": row["selected_model"], "focus_gt": row["focus_gt"],
            "focus_matched": row["focus_matched"], "focus_recall": row["focus_recall"],
            "focus_mean_iou": row["focus_mean_iou"], "focus_iou75": row["focus_iou75"],
            "figure": filename,
        })
        html.append(f'<figure><img src="{escape(filename)}"><figcaption>#{rank} '
                    f'{escape(row["focus"])} · {escape(row["domain"])} · '
                    f'routed={escape(row["selected_model"])} · image {row["image_id"]} · '
                    f'{row["focus_matched"]}/{row["focus_gt"]} matched</figcaption></figure>')

    summary = {
        "source": "frozen DINO MoE test predictions and frozen gate routes",
        "split": "test", "qualitative_only": True,
        "selection": "exactly 10 unique scenes per domain; class quotas are 7 each plus one extra car scene; within each quota, rank by class-aware recall, mean matched IoU and IoU>=0.75 matches",
        "score_threshold": args.score_threshold, "iou_threshold": args.iou_threshold,
        "rendered_total": len(samples),
        "counts_by_domain": dict(Counter(x["domain"] for x in samples)),
        "counts_by_focus": dict(Counter(x["focus"] for x in samples)),
        "counts_by_selected_model": dict(Counter(x["selected_model"] for x in samples)),
        "samples": samples,
    }
    (args.output_dir / "samples.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    page = """<!doctype html><meta charset=\"utf-8\"><title>DINO MoE balanced experts</title>
<style>body{font-family:sans-serif;background:#eef2ef;margin:24px}h1{color:#17251f}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:18px}figure{margin:0;background:white;padding:8px;box-shadow:0 2px 8px #0002}img{width:100%;display:block}figcaption{padding:8px 2px;color:#17251f}</style>
<h1>DINO MoE balanced expert outputs</h1><p>50 unique held-out test scenes: 10 per domain and approximately balanced across detection classes.</p><main>
""" + "\n".join(html) + "\n</main>\n"
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "counts_by_domain": summary["counts_by_domain"],
                      "counts_by_focus": summary["counts_by_focus"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
