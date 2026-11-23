# Routed detector evaluation — test

- Gate domain accuracy: 98.00%
- Score threshold: 0.05
- Route map: `{"day_clear": "baseline_day_clear", "day_foggy": "clip_expert_day_foggy", "dusk_rainy": "clip_expert_dusk_rainy", "night_clear": "clip_expert_night_clear", "night_rainy": "clip_expert_night_rainy"}`

| Domain | Baseline AP50 | MoE AP50 | Baseline AP@[.5:.95] | MoE AP@[.5:.95] |
|---|---:|---:|---:|---:|
| day_clear | 25.22% | 25.41% | 15.08% | 15.23% |
| day_foggy | 20.41% | 25.15% | 11.78% | 14.66% |
| dusk_rainy | 18.97% | 23.13% | 9.21% | 14.26% |
| night_clear | 28.27% | 27.55% | 13.73% | 15.29% |
| night_rainy | 8.35% | 15.53% | 4.35% | 9.21% |

## Domain gate confusion matrix (rows true, columns predicted)

| True \ Predicted | day_clear | day_foggy | dusk_rainy | night_clear | night_rainy |
|---|---:|---:|---:|---:|---:|
| day_clear | 99 | 0 | 0 | 1 | 0 |
| day_foggy | 0 | 100 | 0 | 0 | 0 |
| dusk_rainy | 0 | 1 | 96 | 0 | 3 |
| night_clear | 1 | 0 | 0 | 99 | 0 |
| night_rainy | 0 | 0 | 4 | 0 | 96 |

The `test` split is read only when `--split test` is explicitly passed. Freeze the gate, route map, and model list before using it.
