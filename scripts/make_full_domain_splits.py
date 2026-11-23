#!/usr/bin/env python3
"""Build full official-training COCO splits without touching source data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

DOMAINS = (
    ("day_clear", "Daytime_Sunny.zip", "train", "test"),
    ("day_foggy", "Daytime-Foggy.zip", "trainval", "test"),
    ("dusk_rainy", "Dusk-rainy.zip", "train", None),
    ("night_clear", "Night-Sunny.zip", "trainval", "test"),
    ("night_rainy", "Night_rainy.zip", "train", None),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_ids(archive: Path, split: str) -> set[str]:
    suffix = f"/ImageSets/Main/{split}.txt"
    with ZipFile(archive) as zf:
        matches = [name for name in zf.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one {split}.txt in {archive.name}, found {matches}")
        return {
            line.strip() for line in zf.read(matches[0]).decode("utf-8").splitlines()
            if line.strip()
        }


def subset_coco(data: dict, names: set[str]) -> dict:
    images = [image for image in data["images"] if Path(image["file_name"]).stem in names]
    if len(images) != len(names):
        found = {Path(image["file_name"]).stem for image in images}
        raise ValueError(f"COCO subset mismatch: requested={len(names)} found={len(found)}")
    image_ids = {image["id"] for image in images}
    return {
        "images": copy.deepcopy(images),
        "annotations": [
            copy.deepcopy(annotation)
            for annotation in data["annotations"]
            if annotation["image_id"] in image_ids
        ],
        "categories": copy.deepcopy(data["categories"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--processed-v4", type=Path)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    processed = args.processed.resolve()
    processed_v4 = args.processed_v4.resolve() if args.processed_v4 else None
    frozen_path = args.frozen_manifest.resolve()
    output = args.output.resolve()
    if (not processed.is_relative_to(root) or
            (processed_v4 is not None and not processed_v4.is_relative_to(root)) or
            not frozen_path.is_relative_to(root)):
        raise ValueError("all inputs must stay under data/raw/dwd")
    if args.write and output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rows = {row["domain"]: row for row in frozen["domains"]}
    if set(rows) != {domain for domain, *_ in DOMAINS}:
        raise ValueError("frozen manifest does not contain exactly the five domains")

    prepared = []
    for domain, archive_name, train_split, eval_split in DOMAINS:
        annotation_path = processed / f"{domain}.coco.json"
        archive_path = root / archive_name
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        by_stem = {Path(image["file_name"]).stem: image for image in data["images"]}
        if len(by_stem) != len(data["images"]):
            if processed_v4 is None:
                raise ValueError(f"{domain}: duplicate image stems in {annotation_path}")
            annotation_path = processed_v4 / f"{domain}.coco.json"
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            by_stem = {Path(image["file_name"]).stem: image for image in data["images"]}
        if len(by_stem) != len(data["images"]):
            raise ValueError(f"{domain}: duplicate image stems in {annotation_path}")
        train_ids = official_ids(archive_path, train_split)
        missing = train_ids - by_stem.keys()
        if missing and processed_v4 is not None and annotation_path.parent != processed_v4:
            candidate = processed_v4 / f"{domain}.coco.json"
            candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
            candidate_by_stem = {Path(image["file_name"]).stem: image for image in candidate_data["images"]}
            if not (train_ids - candidate_by_stem.keys()):
                annotation_path, data, by_stem = candidate, candidate_data, candidate_by_stem
                missing = set()
        if missing:
            raise ValueError(f"{domain}: {len(missing)} official train IDs are absent")

        row = rows[domain]
        frozen_sets = {
            "model_validation": set(row["model_validation_images"]),
            "gate_train": set(row["gate_training_images"]),
            "gate_val": set(row["gate_validation_images"]),
            "test": set(row["test_images"]),
        }
        for label, names in frozen_sets.items():
            if not names <= by_stem.keys():
                raise ValueError(f"{domain}: {label} contains an unknown image")
        for left, right in (("model_validation", "gate_train"),
                            ("model_validation", "gate_val"),
                            ("gate_train", "gate_val")):
            if frozen_sets[left] & frozen_sets[right]:
                raise ValueError(f"{domain}: frozen {left}/{right} overlap")
        if eval_split:
            official_test = official_ids(archive_path, eval_split)
            if not frozen_sets["test"] <= official_test:
                raise ValueError(f"{domain}: frozen test is not from official {eval_split}")
        elif not frozen_sets["test"] <= train_ids:
            raise ValueError(f"{domain}: project-defined test is not from official train")

        excluded = set().union(*frozen_sets.values())
        full_train = train_ids - excluded
        if len(full_train) < 1000:
            raise ValueError(f"{domain}: unexpectedly small full train split: {len(full_train)}")
        datasets = {
            "train": subset_coco(data, full_train),
            "val": subset_coco(data, frozen_sets["model_validation"]),
            "gate_train": subset_coco(data, frozen_sets["gate_train"]),
            "gate_val": subset_coco(data, frozen_sets["gate_val"]),
            "test": subset_coco(data, frozen_sets["test"]),
        }
        metadata = {
            "domain": domain,
            "archive": archive_name,
            "official_train_split": train_split,
            "official_eval_split": eval_split,
            "source_annotations": str(annotation_path),
            "source_annotations_sha256": sha256_file(annotation_path),
            "frozen_manifest": str(frozen_path),
            "official_train_images": len(train_ids),
            "excluded_frozen_images": {name: len(names) for name, names in frozen_sets.items()},
            "full_train_images": len(full_train),
            "image_counts": {name: len(dataset["images"]) for name, dataset in datasets.items()},
        }
        prepared.append((domain, datasets, metadata))
        print(json.dumps(metadata, sort_keys=True))

    print("mode:", "write" if args.write else "dry-run (no files written)")
    print("output:", output)
    if not args.write:
        return 0

    output.mkdir(parents=True)
    manifest = {
        "purpose": "full official training minus frozen model/gate/test reserves",
        "frozen_manifest": str(frozen_path),
        "domains": [metadata for _, _, metadata in prepared],
    }
    for domain, datasets, _ in prepared:
        directory = output / domain
        directory.mkdir()
        for split, dataset in datasets.items():
            (directory / f"{domain}_{split}.coco.json").write_text(
                json.dumps(dataset, separators=(",", ":")), encoding="utf-8")
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
