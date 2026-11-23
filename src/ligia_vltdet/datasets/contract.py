from __future__ import annotations

from collections.abc import Iterable, Mapping

CLASSES = ("bus", "bike", "car", "motor", "person", "rider", "truck")
DOMAINS = ("day_clear", "night_clear", "dusk_rainy", "night_rainy", "day_foggy")


class DatasetContractError(ValueError):
    """Raised when converted DWD metadata violates the frozen contract."""


def validate_records(records: Iterable[Mapping[str, object]]) -> None:
    seen: dict[str, str] = {}
    for index, record in enumerate(records, 1):
        required = {"path", "split", "domain", "width", "height", "boxes", "classes"}
        missing = required - record.keys()
        if missing:
            raise DatasetContractError(f"record {index}: missing {sorted(missing)}")
        path, domain = str(record["path"]), str(record["domain"])
        if domain not in DOMAINS:
            raise DatasetContractError(f"record {index}: unknown domain {domain!r}")
        if path in seen and seen[path] != domain:
            raise DatasetContractError(f"image {path!r} occurs in {seen[path]} and {domain}")
        seen[path] = domain
        width, height = int(record["width"]), int(record["height"])
        if width <= 0 or height <= 0:
            raise DatasetContractError(f"record {index}: non-positive dimensions")
        boxes = record["boxes"]
        classes = record["classes"]
        if not isinstance(boxes, list) or not isinstance(classes, list) or len(boxes) != len(classes):
            raise DatasetContractError(f"record {index}: boxes/classes length mismatch")
        for box, class_name in zip(boxes, classes):
            if class_name not in CLASSES:
                raise DatasetContractError(f"record {index}: unknown class {class_name!r}")
            if not isinstance(box, list) or len(box) != 4:
                raise DatasetContractError(f"record {index}: invalid box {box!r}")
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise DatasetContractError(f"record {index}: out-of-bounds box {box!r}")
