#!/usr/bin/env python3
"""Build the full DayClear training pool without consuming frozen reserves."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from zipfile import ZipFile


def official_ids(archive: Path, split: str) -> set[str]:
    suffix = f"/ImageSets/Main/{split}.txt"
    with ZipFile(archive) as zf:
        matches = [name for name in zf.namelist() if name.endswith(suffix)
                   and "/annotations_cache/" not in name]
        if len(matches) != 1:
            raise ValueError(f"expected one {split}.txt in {archive.name}, found {matches}")
        return {line.strip() for line in zf.read(matches[0]).decode().splitlines() if line.strip()}


def subset(data: dict, names: set[str]) -> dict:
    selected = [image for image in data["images"] if Path(image["file_name"]).stem in names]
    ids = {image["id"] for image in selected}
    return {"images": copy.deepcopy(selected),
            "annotations": [copy.deepcopy(ann) for ann in data["annotations"] if ann["image_id"] in ids],
            "categories": copy.deepcopy(data["categories"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root, source, base_path, output = (p.resolve() for p in
                                       (args.root, args.source, args.base_manifest, args.output))
    expected_root = Path.cwd().resolve() / "data/raw/dwd"
    if root != expected_root or not all(p.is_relative_to(root) for p in (source, base_path, output)):
        raise ValueError("run from project root and keep all DWD files under data/raw/dwd")
    if args.write and output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    row = next(item for item in base["domains"] if item["domain"] == "day_clear")
    data = json.loads(source.read_text(encoding="utf-8"))
    by_stem = {Path(image["file_name"]).stem: image for image in data["images"]}
    official_train = official_ids(root / "Daytime_Sunny.zip", "train")
    official_test = official_ids(root / "Daytime_Sunny.zip", "test")
    reserve_fields = ("model_validation_images", "gate_training_images",
                      "gate_validation_images", "test_images")
    reserves = set().union(*(set(row[field]) for field in reserve_fields[:-1]))
    test_ids = set(row["test_images"])
    if reserves & test_ids or not test_ids <= official_test:
        raise ValueError("frozen DayClear reserves are invalid or test IDs are not official test IDs")
    if not reserves <= official_train or not (official_train - reserves) <= by_stem:
        raise ValueError("source annotations do not cover the complete official DayClear train pool")
    train_ids = official_train - reserves
    result = subset(data, train_ids)
    output_row = copy.deepcopy(row)
    output_row["train_images"] = sorted(train_ids)
    output_row["train_images_added"] = sorted(train_ids - set(row.get("train_images", [])))
    print(json.dumps({"train_images": len(result["images"]),
                      "model_val_images": len(row["model_validation_images"]),
                      "gate_train_images": len(row["gate_training_images"]),
                      "gate_val_images": len(row["gate_validation_images"]),
                      "test_images": len(row["test_images"]),
                      "mode": "write" if args.write else "dry-run",
                      "output": str(output)}, sort_keys=True))
    if args.write:
        directory = output / "day_clear"
        directory.mkdir(parents=True)
        (directory / "day_clear_train.coco.json").write_text(json.dumps(result, separators=(",", ":")))
        val = subset(data, set(row["model_validation_images"]))
        (directory / "day_clear_val.coco.json").write_text(json.dumps(val, separators=(",", ":")))
        (output / "split_manifest.json").write_text(json.dumps({
            "seed": args.seed, "source_manifest": str(base_path),
            "source_annotations": str(source), "train_images_per_domain": len(result["images"]),
            "model_val_images_per_domain": len(val["images"]),
            "gate_train_images_per_domain": len(row["gate_training_images"]),
            "gate_val_images_per_domain": len(row["gate_validation_images"]),
            "test_images_per_domain": len(row["test_images"]), "domains": [output_row]
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
