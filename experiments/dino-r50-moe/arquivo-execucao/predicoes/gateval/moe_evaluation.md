# Routed detector evaluation — gate_val

- Gate domain accuracy: 97.60%
- Score threshold: 0.05
- Route map: `{"day_clear": "baseline_day_clear", "day_foggy": "expert_day_foggy", "dusk_rainy": "expert_dusk_rainy", "night_clear": "expert_night_clear", "night_rainy": "expert_night_rainy"}`

| Domain | Baseline AP50 | MoE AP50 | Baseline AP@[.5:.95] | MoE AP@[.5:.95] |
|---|---:|---:|---:|---:|
| day_clear | 44.63% | 44.65% | 29.49% | 29.50% |
| day_foggy | 26.31% | 36.88% | 14.30% | 21.08% |
| dusk_rainy | 32.84% | 44.87% | 16.28% | 21.21% |
| night_clear | 29.14% | 32.82% | 19.14% | 17.91% |
| night_rainy | 27.88% | 28.35% | 12.82% | 14.43% |

## Domain gate confusion matrix (rows true, columns predicted)

| True \ Predicted | day_clear | day_foggy | dusk_rainy | night_clear | night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 98 | 0 | 2 | 0 | 0 |
| day_foggy | 0 | 98 | 2 | 0 | 0 |
| dusk_rainy | 2 | 0 | 93 | 0 | 5 |
| night_clear | 0 | 0 | 0 | 100 | 0 |
| night_rainy | 0 | 0 | 1 | 0 | 99 |

The `test` split is read only when `--split test` is explicitly passed. Freeze the gate, route map, and model list before using it.

## Standalone frozen detector matrix

| Domain | baseline_day_clear | expert_day_foggy | expert_dusk_rainy | expert_night_clear | expert_night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 44.63% | 32.19% | 42.18% | 41.24% | 41.09% |
| day_foggy | 26.31% | 36.28% | 25.77% | 27.52% | 24.88% |
| dusk_rainy | 32.84% | 18.48% | 45.09% | 27.98% | 32.98% |
| night_clear | 29.14% | 20.95% | 24.90% | 32.82% | 29.53% |
| night_rainy | 27.88% | 20.91% | 42.93% | 38.21% | 27.85% |
