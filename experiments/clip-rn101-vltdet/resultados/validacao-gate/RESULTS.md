# Routed detector evaluation — gate_val

- Gate domain accuracy: 97.60%
- Score threshold: 0.05
- Route map: `{"day_clear": "baseline_day_clear", "day_foggy": "clip_expert_day_foggy", "dusk_rainy": "clip_expert_dusk_rainy", "night_clear": "clip_expert_night_clear", "night_rainy": "clip_expert_night_rainy"}`

| Domain | Baseline AP50 | MoE AP50 | Baseline AP@[.5:.95] | MoE AP@[.5:.95] |
|---|---:|---:|---:|---:|
| day_clear | 34.35% | 34.34% | 20.11% | 20.11% |
| day_foggy | 20.67% | 26.77% | 12.04% | 16.44% |
| dusk_rainy | 17.47% | 26.92% | 8.33% | 13.39% |
| night_clear | 17.95% | 22.04% | 10.61% | 12.20% |
| night_rainy | 16.25% | 15.72% | 9.02% | 7.87% |

## Domain gate confusion matrix (rows true, columns predicted)

| True \ Predicted | day_clear | day_foggy | dusk_rainy | night_clear | night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 98 | 0 | 2 | 0 | 0 |
| day_foggy | 0 | 98 | 2 | 0 | 0 |
| dusk_rainy | 2 | 0 | 93 | 0 | 5 |
| night_clear | 0 | 0 | 0 | 100 | 0 |
| night_rainy | 0 | 0 | 1 | 0 | 99 |

The `test` split is read only when `--split test` is explicitly passed. Freeze the gate, route map, and model list before using it.
