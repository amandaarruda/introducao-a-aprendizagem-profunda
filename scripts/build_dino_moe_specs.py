#!/usr/bin/env python3
"""Write frozen DINO model and pre-specified domain route manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAINS = ("day_clear", "day_foggy", "dusk_rainy", "night_clear", "night_rainy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True,
                        help="DINO ResNet-50 expert checkpoint root with one directory per domain")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = args.config.resolve()
    baseline = args.baseline.resolve()
    expert_root = args.expert_root.resolve()
    if not config.is_file() or not baseline.is_file():
        raise FileNotFoundError("DINO config or baseline checkpoint is missing")
    models = {"baseline_day_clear": {"config": str(config), "checkpoint": str(baseline)}}
    for domain in DOMAINS[1:]:
        checkpoint = (expert_root / domain / "iter_12000.pth").resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        models[f"expert_{domain}"] = {"config": str(config), "checkpoint": str(checkpoint)}
    route_map = {
        "day_clear": "baseline_day_clear", "day_foggy": "expert_day_foggy",
        "dusk_rainy": "expert_dusk_rainy", "night_clear": "expert_night_clear",
        "night_rainy": "expert_night_rainy",
    }
    output.mkdir(parents=True)
    (output / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (output / "route_map.json").write_text(json.dumps(route_map, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
