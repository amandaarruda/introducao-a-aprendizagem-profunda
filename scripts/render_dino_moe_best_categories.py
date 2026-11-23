#!/usr/bin/env python3
"""Render outcome-selected DINO MoE detections balanced across classes.

The script consumes saved COCO predictions from a frozen evaluation. It selects
unique scenes by class-aware one-to-one matches at IoU >= 0.5, renders ground
truth/baseline/routed-MoE panels, and writes a compact HTML gallery plus JSON
selection metadata. The result is qualitative evidence, not a new metric.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")
CLASSES = ("bus", "bike", "car", "motor", "person", "rider", "truck")
COLORS = {
    "bus": (65, 145, 230), "bike": (214, 90, 196), "car": (28, 180, 105),
    "motor": (237, 149, 43), "person": (230, 81, 81), "rider": (50, 190, 195),
    "truck": (147, 104, 220),
}
PANEL_W, PANEL_H, HEADER_H = 640, 400, 52


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def matches(gt: list[dict], predictions: list[dict], category_id: int, threshold: float):
    pairs = []
    gt_items = [(i, xywh_to_xyxy(x["bbox"])) for i, x in enumerate(gt)
                if category_id == -1 or x["category_id"] == category_id]
    pred_items = [(i, xywh_to_xyxy(x["bbox"])) for i, x in enumerate(predictions)
                  if category_id == -1 or x["category_id"] == category_id]
    for gi, gbox in gt_items:
        for pi, pbox in pred_items:
            score = iou(gbox, pbox)
            if score >= threshold:
                pairs.append((score, gi, pi))
    used_gt, used_pred, scores = set(), set(), []
    for score, gi, pi in sorted(pairs, reverse=True):
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            scores.append(score)
    return scores


def load_predictions(path: Path, threshold: float):
    by_image = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        if float(row["score"]) >= threshold:
            by_image[int(row["image_id"])].append(row)
    return by_image


def draw_panel(image: Image.Image, title: str, boxes: list[dict], categories: dict[int, str],
               ground_truth: bool, focus: str | None, max_boxes: int):
    canvas = Image.new("RGB", (PANEL_W, PANEL_H + HEADER_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PANEL_W, HEADER_H), fill=(23, 37, 31))
    draw.text((14, 15), title, fill="white", font=font(20))
    scale = min(PANEL_W / image.width, PANEL_H / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)))
    xoff = (PANEL_W - resized.width) // 2
    yoff = HEADER_H + (PANEL_H - resized.height) // 2
    canvas.paste(resized, (xoff, yoff))
    draw = ImageDraw.Draw(canvas)
    for item in (boxes if ground_truth else boxes[:max_boxes]):
        category_id = int(item["category_id"])
        name = categories.get(category_id, str(category_id))
        color = COLORS.get(name, (255, 220, 50))
        if ground_truth:
            x, y, w, h = item["bbox"]
            x2, y2 = x + w, y + h
            label = name
        else:
            x, y, x2, y2 = xywh_to_xyxy(item["bbox"])
            label = f"{name} {float(item['score']):.2f}"
        width = 5 if name == focus else 3
        coords = (xoff + x * scale, yoff + y * scale,
                  xoff + x2 * scale, yoff + y2 * scale)
        draw.rectangle(coords, outline=color, width=width)
        tx, ty = coords[0], max(yoff, coords[1] - 20)
        text_box = draw.textbbox((tx, ty), label, font=font(15))
        draw.rectangle(text_box, fill=color)
        draw.text((tx, ty), label, fill=(10, 20, 16), font=font(15))
    return canvas


def render_row(image: Image.Image, domain: str, image_id: int, focus: str,
               gt, baseline, moe, categories, max_boxes: int):
    title = f"focus={focus} · {domain} · image_id={image_id}"
    panels = [
        draw_panel(image, f"Ground truth · {title}", gt, categories, True, focus, max_boxes),
        draw_panel(image, "DINO baseline · DayClear", baseline, categories, False, focus, max_boxes),
        draw_panel(image, "DINO routed MoE", moe, categories, False, focus, max_boxes),
    ]
    row = Image.new("RGB", (PANEL_W * 3, PANEL_H + HEADER_H), "white")
    for index, panel in enumerate(panels):
        row.paste(panel, (index * PANEL_W, 0))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=7)
    parser.add_argument("--max-total", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_total < len(CLASSES) * args.per_class:
        raise ValueError("max-total must fit per-class quota")
    args.output_dir.mkdir(parents=True)

    candidates = []
    lookup = {}
    for domain in DOMAINS:
        ann_path = args.annotations_root / domain / f"{domain}_test.coco.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(x["id"]): x["name"] for x in data["categories"]}
        class_ids = {name: category_id for category_id, name in categories.items()}
        images = {int(x["id"]): x for x in data["images"]}
        gt_by_image = defaultdict(list)
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0):
                gt_by_image[int(ann["image_id"])].append(ann)
        baseline_by_image = load_predictions(
            args.predictions_dir / f"baseline_predictions_{domain}.json", args.score_threshold)
        moe_by_image = load_predictions(
            args.predictions_dir / f"moe_predictions_{domain}.json", args.score_threshold)
        for image_id, gt in gt_by_image.items():
            info = images[image_id]
            image_path = Path(info["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            baseline = baseline_by_image[image_id]
            moe = moe_by_image[image_id]
            overall = matches(gt, moe, -1, args.iou_threshold)
            for focus in CLASSES:
                category_id = class_ids[focus]
                focus_gt = sum(ann["category_id"] == category_id for ann in gt)
                if not focus_gt:
                    continue
                focus_matches = matches(gt, moe, category_id, args.iou_threshold)
                baseline_matches = matches(gt, baseline, category_id, args.iou_threshold)
                candidates.append({
                    "domain": domain, "image_id": image_id, "image_file": str(image_path),
                    "focus": focus, "focus_gt": focus_gt,
                    "focus_matched": len(focus_matches),
                    "focus_recall": len(focus_matches) / focus_gt,
                    "focus_mean_iou": sum(focus_matches) / len(focus_matches) if focus_matches else 0.0,
                    "focus_iou75": sum(x >= 0.75 for x in focus_matches),
                    "baseline_focus_matched": len(baseline_matches),
                    "moe_overall_matched": len(overall),
                    "gt": gt, "baseline": baseline, "moe": moe, "categories": categories,
                })

    # Select unique scenes, preferring class recall, localization quality and then
    # the number of matched objects. The second pass fills any rare-class quota.
    selected = []
    used_images = set()
    for focus in CLASSES:
        rows = [row for row in candidates if row["focus"] == focus]
        rows.sort(key=lambda row: (-row["focus_recall"], -row["focus_mean_iou"],
                                   -row["focus_iou75"], -row["focus_matched"],
                                   row["domain"], row["image_id"]))
        chosen = [row for row in rows if (row["domain"], row["image_id"]) not in used_images][:args.per_class]
        if len(chosen) < args.per_class:
            chosen += [row for row in rows if row not in chosen][:args.per_class - len(chosen)]
        for row in chosen:
            row = dict(row)
            row["selection_rank"] = len(selected) + 1
            selected.append(row)
            used_images.add((row["domain"], row["image_id"]))

    remaining = [row for row in candidates if (row["domain"], row["image_id"]) not in used_images]
    remaining.sort(key=lambda row: (-row["moe_overall_matched"], -row["focus_mean_iou"],
                                    row["domain"], row["image_id"]))
    if len(selected) < args.max_total and remaining:
        row = dict(remaining[0])
        row["focus"] = "overall"
        row["selection_rank"] = len(selected) + 1
        selected.append(row)

    selected = selected[:args.max_total]
    summary = {
        "source": "frozen DINO MoE test predictions",
        "split": "test",
        "qualitative_only": True,
        "selection": "unique scenes ranked per class by MoE class-aware recall, mean matched IoU and IoU>=0.75 matches; one extra overall scene",
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "requested_total": args.max_total,
        "rendered_total": len(selected),
        "counts_by_focus": dict(Counter(row["focus"] for row in selected)),
        "samples": [],
    }
    html_rows = []
    for row in selected:
        image = Image.open(row["image_file"]).convert("RGB")
        rendered = render_row(image, row["domain"], row["image_id"], row["focus"],
                              row["gt"], row["baseline"], row["moe"], row["categories"], args.max_boxes)
        name = f"{row['selection_rank']:02d}_{row['focus']}_{row['domain']}_id{row['image_id']}.png"
        rendered.save(args.output_dir / name, optimize=True)
        summary["samples"].append({
            "rank": row["selection_rank"], "focus": row["focus"], "domain": row["domain"],
            "image_id": row["image_id"], "image_file": Path(row["image_file"]).name,
            "focus_gt": row["focus_gt"], "focus_matched": row["focus_matched"],
            "focus_recall": row["focus_recall"], "focus_mean_iou": row["focus_mean_iou"],
            "focus_iou75": row["focus_iou75"], "baseline_focus_matched": row["baseline_focus_matched"],
            "figure": name,
        })
        html_rows.append(
            f'<figure><img src="{escape(name)}"><figcaption>#{row["selection_rank"]} '
            f'{escape(row["focus"])} · {escape(row["domain"])} · image {row["image_id"]} '
            f'· MoE {row["focus_matched"]}/{row["focus_gt"]} matched</figcaption></figure>')

    (args.output_dir / "samples.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    html = """<!doctype html><meta charset="utf-8"><title>DINO MoE best category outputs</title>
<style>body{font-family:sans-serif;background:#eef2ef;margin:24px}h1{color:#17251f}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:18px}figure{margin:0;background:white;padding:8px;box-shadow:0 2px 8px #0002}img{width:100%;display:block}figcaption{padding:8px 2px;color:#17251f}</style>
<h1>DINO MoE qualitative outputs</h1><p>Outcome-selected, qualitative only; 50 unique held-out test scenes balanced across detection classes.</p><main>
""" + "\n".join(html_rows) + "\n</main>\n"
    (args.output_dir / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "rendered_total": len(selected),
                      "counts_by_focus": summary["counts_by_focus"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
