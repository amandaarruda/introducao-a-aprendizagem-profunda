# Routed detector evaluation — test

- Gate domain accuracy: 98.00%
- Score threshold: 0.05
- Route map: `{"day_clear": "baseline_day_clear", "day_foggy": "expert_day_foggy", "dusk_rainy": "expert_dusk_rainy", "night_clear": "expert_night_clear", "night_rainy": "expert_night_rainy"}`

| Domain | Baseline AP50 | MoE AP50 | Baseline AP@[.5:.95] | MoE AP@[.5:.95] |
|---|---:|---:|---:|---:|
| day_clear | 45.11% | 44.87% | 24.69% | 24.48% |
| day_foggy | 24.41% | 35.53% | 13.80% | 20.83% |
| dusk_rainy | 33.57% | 51.09% | 18.59% | 24.40% |
| night_clear | 44.85% | 51.03% | 24.55% | 30.40% |
| night_rainy | 9.06% | 40.30% | 4.48% | 24.23% |

## Domain gate confusion matrix (rows true, columns predicted)

| True \ Predicted | day_clear | day_foggy | dusk_rainy | night_clear | night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 99 | 0 | 0 | 1 | 0 |
| day_foggy | 0 | 100 | 0 | 0 | 0 |
| dusk_rainy | 0 | 1 | 96 | 0 | 3 |
| night_clear | 1 | 0 | 0 | 99 | 0 |
| night_rainy | 0 | 0 | 4 | 0 | 96 |

The `test` split is read only when `--split test` is explicitly passed. Freeze the gate, route map, and model list before using it.

## Standalone frozen detector matrix

| Domain | baseline_day_clear | expert_day_foggy | expert_dusk_rainy | expert_night_clear | expert_night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 45.11% | 31.00% | 38.20% | 40.35% | 36.07% |
| day_foggy | 24.41% | 35.53% | 24.02% | 26.57% | 23.54% |
| dusk_rainy | 33.57% | 24.42% | 51.31% | 32.08% | 42.92% |
| night_clear | 44.85% | 24.61% | 38.60% | 51.04% | 41.75% |
| night_rainy | 9.06% | 6.89% | 35.29% | 19.30% | 40.31% |
