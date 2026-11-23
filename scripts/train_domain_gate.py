#!/usr/bin/env python3
"""Train a frozen CLIP RN101 linear domain router on gate-train only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import open_clip

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class GateImages(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.transform = transform
        self.items = []
        for label, domain in enumerate(DOMAINS):
            ann_path = root / domain / f"{domain}_{split}.coco.json"
            if not ann_path.is_file():
                raise FileNotFoundError(ann_path)
            coco = json.loads(ann_path.read_text(encoding="utf-8"))
            for image in coco["images"]:
                path = Path(image["file_name"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.items.append({
                    "path": path,
                    "label": label,
                    "image_id": int(image["id"]),
                    "domain": domain,
                })
        counts = {d: sum(x["domain"] == d for x in self.items) for d in DOMAINS}
        if len(self.items) != 500 or any(counts[d] != 100 for d in DOMAINS):
            raise ValueError(f"expected 100 {split} images per domain; got {counts}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, item["label"], item["image_id"], str(item["path"])


def load_clip(checkpoint: Path, device: torch.device):
    model, _, transform = open_clip.create_model_and_transforms(
        "RN101", pretrained=None, pretrained_hf=False)
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        state = state.state_dict()
    state = state.get("state_dict", state)
    model_state = model.state_dict()
    state = {key: value for key, value in state.items() if key in model_state}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"incomplete CLIP RN101 load: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, transform


def extract(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    features, labels, metadata = [], [], []
    for images, y, image_ids, paths in loader:
        images = images.to(device, non_blocking=True)
        embeddings = F.normalize(model.encode_image(images).float(), dim=1)
        features.append(embeddings.cpu())
        labels.append(y.long())
        metadata.extend({"image_id": int(i), "path": p} for i, p in zip(image_ids, paths))
    return torch.cat(features), torch.cat(labels), metadata



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gate feature extraction")
    if not args.clip_checkpoint.is_file():
        parser.error("CLIP RN101 requires an existing --clip-checkpoint")
    encoder_hash = sha256(args.clip_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    model, transform = load_clip(args.clip_checkpoint, device)
    train_set = GateImages(args.annotations_root, "gate_train", transform)
    val_set = GateImages(args.annotations_root, "gate_val", transform)
    x_train, y_train, train_meta = extract(model, train_set, device, args.batch_size, args.workers)
    x_val, y_val, val_meta = extract(model, val_set, device, args.batch_size, args.workers)
    del model
    torch.cuda.empty_cache()
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    head = nn.Linear(x_train.shape[1], len(DOMAINS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=0.01)
    best = {"accuracy": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = head(x_train)
        train_loss = F.cross_entropy(train_logits, y_train)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(x_val)
            val_loss = float(F.cross_entropy(val_logits, y_val).item())
            val_pred = val_logits.argmax(dim=1)
            val_accuracy = float((val_pred == y_val).float().mean().item())
        improved = (val_accuracy > best["accuracy"] or
                    (val_accuracy == best["accuracy"] and val_loss < best["loss"]))
        if improved:
            best = {"accuracy": val_accuracy, "loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"epoch={epoch} train_loss={train_loss.item():.6f} val_loss={val_loss:.6f} "
                  f"val_domain_accuracy={val_accuracy:.4f} best_epoch={best['epoch']}", flush=True)
        if stale_epochs >= 30:
            break
    if best["state"] is None:
        raise RuntimeError("gate training produced no checkpoint")
    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        val_logits = head(x_val)
        val_probs = val_logits.softmax(dim=1).cpu().tolist()
        val_pred = val_logits.argmax(dim=1).cpu().tolist()
    true_labels = y_val.cpu().tolist()
    confusion = [[0 for _ in DOMAINS] for _ in DOMAINS]
    for truth, prediction in zip(true_labels, val_pred):
        confusion[truth][prediction] += 1
    per_domain = {}
    for index, domain in enumerate(DOMAINS):
        total = sum(confusion[index])
        per_domain[domain] = {
            "images": total,
            "recall": confusion[index][index] / total if total else None,
        }
    manifest = args.annotations_root / "split_manifest.json"
    checkpoint_path = args.output_dir / "domain_gate_linear.pt"
    torch.save({
        "state_dict": best["state"],
        "domains": list(DOMAINS),
        "feature_dim": int(x_train.shape[1]),
        "encoder": "OpenCLIP RN101 frozen image embedding, L2 normalized",
        "best_epoch": best["epoch"],
        "gate_val_domain_accuracy": best["accuracy"],
        "gate_val_domain_loss": best["loss"],
        "seed": args.seed,
        "encoder_kind": "clip-rn101",
        "encoder_checkpoint_sha256": encoder_hash,
    }, checkpoint_path)
    prediction_rows = []
    for meta, truth, prediction, probs in zip(val_meta, true_labels, val_pred, val_probs):
        prediction_rows.append({
            **meta,
            "true_domain": DOMAINS[truth],
            "predicted_domain": DOMAINS[prediction],
            "probabilities": {domain: float(probs[i]) for i, domain in enumerate(DOMAINS)},
        })
    predictions_path = args.output_dir / "gate_val_predictions.json"
    predictions_path.write_text(json.dumps(prediction_rows, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "scope": "gate_train/gate_val only; no model-validation or test images loaded",
        "encoder": "OpenCLIP RN101 frozen image embedding, L2 normalized",
        "classifier": "single linear layer, cross entropy, AdamW lr=0.03 weight_decay=0.01",
        "best_epoch": best["epoch"],
        "train_images": len(train_set),
        "gate_val_images": len(val_set),
        "gate_val_domain_accuracy": best["accuracy"],
        "gate_val_domain_loss": best["loss"],
        "per_domain_recall": per_domain,
        "confusion_matrix_rows_true_cols_predicted": confusion,
        "domains": list(DOMAINS),
        "seed": args.seed,
        "manifest_sha256": sha256(manifest),
        "encoder_kind": "clip-rn101",
        "encoder_checkpoint_sha256": encoder_hash,
        "gate_checkpoint": str(checkpoint_path.resolve()),
        "gate_checkpoint_sha256": sha256(checkpoint_path),
        "predictions": str(predictions_path.resolve()),
    }
    metrics_path = args.output_dir / "gate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Frozen CLIP RN101 domain gate",
        "",
        "The CLIP RN101 encoder was frozen. A linear five-domain classifier was trained from the 500 `gate_train` images and checkpoint selection used domain accuracy/loss on the separate 500-image `gate_val` set.",
        "Only file names and domain labels from these two splits were loaded; detection boxes, model-validation, and test annotations were not used.",
        "",
        f"- Best epoch: {best['epoch']}",
        f"- Gate-val domain accuracy: {100 * best['accuracy']:.2f}%",
        f"- Gate-val cross-entropy: {best['loss']:.6f}",
        f"- Checkpoint SHA-256: `{sha256(checkpoint_path)}`",
        "",
        "## Confusion matrix (rows true, columns predicted)",
        "",
        "| True \ Predicted | " + " | ".join(DOMAINS) + " |",
        "|---|" + "---:|" * len(DOMAINS),
    ]
    for i, domain in enumerate(DOMAINS):
        lines.append("| " + domain + " | " + " | ".join(map(str, confusion[i])) + " |")
    lines += ["", "## Recall by domain", "", "| Domain | Images | Recall |", "|---|---:|---:|"]
    for domain in DOMAINS:
        value = per_domain[domain]["recall"]
        lines.append(f"| {domain} | {per_domain[domain]['images']} | {100 * value:.2f}% |")
    report_path = args.output_dir / "gate_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"GATE_REPORT={report_path.resolve()}", flush=True)
    print(f"GATE_METRICS={metrics_path.resolve()}", flush=True)
    print(f"GATE_CHECKPOINT_SHA256={sha256(checkpoint_path)}", flush=True)


if __name__ == "__main__":
    main()
