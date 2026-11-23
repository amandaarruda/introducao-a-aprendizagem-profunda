from __future__ import annotations

from collections.abc import Mapping

from ligia_vltdet.datasets.contract import DOMAINS


def summarize_domains(scores: Mapping[str, float]) -> dict[str, object]:
    missing = set(DOMAINS) - scores.keys()
    extra = scores.keys() - set(DOMAINS)
    if missing or extra:
        raise ValueError(f"domain mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    values = [float(scores[name]) for name in DOMAINS]
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError("mAP50 values must be percentages in [0, 100]")
    return {
        "metric": "mAP50",
        "domains": dict(zip(DOMAINS, values)),
        "mean_five_domains": sum(values) / 5,
        "dg_target_mean": sum(values[1:]) / 4,
    }
