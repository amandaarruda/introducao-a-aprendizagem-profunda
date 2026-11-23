#!/usr/bin/env python3
"""Render deterministic model-validation examples for qualitative inspection.

Selects the lowest image_id with at least ``min_instances`` ground-truth boxes
in each COCO validation set, then compares the annotations with predictions
from the 5k baseline and its DayFoggy fine-tune. This produces figures only;
it does not change checkpoints or select models/routes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageDraw, ImageFont

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")
COLORS = {
    "bus": (65, 145, 230), "bike": (214, 90, 196), "car": (28, 180, 105),
    "motor": (237, 149, 43), "person": (230, 81, 81), "rider": (50, 190, 195),
    "truck": (147, 104, 220),
}
PANEL_W, PANEL_H, HEADER_H = 640, 400, 48


def load_font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def select_sample(annotation_path: Path, min_instances: int):
    data = json.loads(annotation_path.read_text())
    images = {item["id"]: item for item in data["images"]}
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            by_image[ann["image_id"]].append(ann)
    candidates = [image_id for image_id in sorted(images)
                  if len(by_image[image_id]) >= min_instances]
    if not candidates:
        raise RuntimeError(f"No image with {min_instances} GT boxes: {annotation_path}")
    image_id = candidates[0]
    categories = {c["id"]: c["name"] for c in data["categories"]}
    return data, images[image_id], by_image[image_id], categories


def locate_image(info: dict, image_root: Path | None) -> Path:
    path = Path(info["file_name"])
    if path.is_file():
        return path
    if image_root is not None:
        candidate = image_root / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found: {info['file_name']}; provide --image-root")


def extract_predictions(result, categories: dict[int, str], threshold: float):
    pred = result.pred_instances
    boxes = pred.bboxes.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    items = []
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < threshold:
            continue
        category_id = int(label) + 1
        items.append({"category_id": category_id,
                      "category": categories.get(category_id, str(category_id)),
                      "score": float(score), "bbox_xyxy": [float(x) for x in box]})
    return sorted(items, key=lambda item: item["score"], reverse=True)


def draw_panel(image: Image.Image, title: str, boxes: list[dict], categories: dict[int, str],
               ground_truth: bool, max_boxes: int):
    canvas = Image.new("RGB", (PANEL_W, PANEL_H + HEADER_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PANEL_W, HEADER_H), fill=(23, 37, 31))
    draw.text((16, 13), title, fill="white", font=load_font(22))
    image = image.convert("RGB")
    scale = min(PANEL_W / image.width, PANEL_H / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)))
    xoff = (PANEL_W - resized.width) // 2
    yoff = HEADER_H + (PANEL_H - resized.height) // 2
    canvas.paste(resized, (xoff, yoff))
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    items = boxes if ground_truth else boxes[:max_boxes]
    for item in items:
        if ground_truth:
            x, y, w, h = item["bbox"]
            x2, y2 = x + w, y + h
            name = categories.get(item["category_id"], str(item["category_id"]))
            text = name
        else:
            x, y, x2, y2 = item["bbox_xyxy"]
            name = item["category"]
            text = f"{name} {item['score']:.2f}"
        color = COLORS.get(name, (255, 220, 50))
        coords = (xoff + x * scale, yoff + y * scale,
                  xoff + x2 * scale, yoff + y2 * scale)
        draw.rectangle(coords, outline=color, width=3)
        tx, ty = coords[0], max(yoff, coords[1] - 20)
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), text, fill=(10, 20, 16), font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--finetune-config", type=Path, required=True)
    parser.add_argument("--finetune-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    models = {}
    for label, config, checkpoint in (
        ("baseline_5k", args.baseline_config, args.baseline_checkpoint),
        ("dayfoggy_finetune_5k", args.finetune_config, args.finetune_checkpoint),
    ):
        models[label] = init_detector(str(config), str(checkpoint), device="cuda:0")
        models[label].eval()

    metadata = {"selection_rule": f"lowest image_id with >= {args.min_instances} GT boxes",
                "split": "model-validation", "score_threshold": args.score_threshold,
                "domains": {}}
    domain_panels = {}
    for domain in DOMAINS:
        annotation_path = args.annotations_root / domain / f"{domain}_val.coco.json"
        data, image_info, gt, categories = select_sample(annotation_path, args.min_instances)
        image_path = locate_image(image_info, args.image_root)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not decode {image_path}")
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        preds = {name: extract_predictions(inference_detector(model, str(image_path)),
                                           categories, args.score_threshold)
                 for name, model in models.items()}
        gt_counts = Counter(categories.get(a["category_id"], str(a["category_id"])) for a in gt)
        panels = [draw_panel(image, "Ground truth", gt, categories, True, args.max_boxes),
                  draw_panel(image, "Baseline DayClear · 5k", preds["baseline_5k"], categories, False, args.max_boxes),
                  draw_panel(image, "Fine-tune DayFoggy · 5k", preds["dayfoggy_finetune_5k"], categories, False, args.max_boxes)]
        row = Image.new("RGB", (PANEL_W * 3, PANEL_H + HEADER_H), "white")
        for i, panel in enumerate(panels):
            row.paste(panel, (i * PANEL_W, 0))
        row.save(args.output_dir / f"{domain}.png", optimize=True)
        domain_panels[domain] = row
        metadata["domains"][domain] = {
            "image_id": image_info["id"], "image_file": image_path.name,
            "image_width": image_info["width"], "image_height": image_info["height"],
            "gt_count_by_class": dict(gt_counts),
            "ground_truth": [{"category_id": a["category_id"],
                              "category": categories.get(a["category_id"], str(a["category_id"])),
                              "bbox_xywh": a["bbox"]} for a in gt],
            "predictions": preds,
        }
        print(f"rendered {domain} image_id={image_info['id']} GT={len(gt)} "
              f"baseline={len(preds['baseline_5k'])} finetune={len(preds['dayfoggy_finetune_5k'])}")

    # Five-domain contact sheet; deterministic order matches results tables.
    sheet = Image.new("RGB", (PANEL_W * 3, (PANEL_H + HEADER_H) * len(DOMAINS)), "white")
    for i, domain in enumerate(DOMAINS):
        sheet.paste(domain_panels[domain], (0, i * (PANEL_H + HEADER_H)))
    sheet.save(args.output_dir / "five_domain_contact_sheet.jpg", quality=88, optimize=True)
    # Presentation figure highlights the specialization gain and its night-domain cost.
    highlights = Image.new("RGB", (PANEL_W * 3, (PANEL_H + HEADER_H) * 2), "white")
    for i, domain in enumerate(("day_foggy", "night_clear")):
        highlights.paste(domain_panels[domain], (0, i * (PANEL_H + HEADER_H)))
    highlights.save(args.output_dir / "presentation_dayfoggy_nightclear.jpg", quality=90, optimize=True)
    (args.output_dir / "samples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
