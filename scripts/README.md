# Scripts

Os scripts são executados diretamente (`python scripts/<nome>.py --help`). Eles se importam pelo nome do módulo, então ficam no mesmo diretório.

## Dados

| Script | Função |
|---|---|
| `make_full_domain_splits.py` | Gera os COCO JSONs de treino completos por domínio, preservando as reservas congeladas. |
| `make_dayclear_full_split.py` | Gera o pool completo de treino de `day_clear` sem consumir as reservas. |

## Gate e MoE

| Script | Função |
|---|---|
| `train_domain_gate.py` | Treina o gate linear sobre embeddings CLIP RN101 congelados, usando só `gate_train`. |
| `evaluate_moe_gate.py` | Avalia o MoE: roteia cada imagem pelo gate, aplica o detector da rota e calcula AP50 e COCO AP. |
| `build_dino_moe_specs.py` | Gera `models.json` e `route_map.json` para o MoE DINO. |

## Galerias e figuras

| Script | Função |
|---|---|
| `render_dino_moe_balanced_gallery.py` | Galeria de 50 cenas, 10 por domínio. |
| `render_dino_moe_best_categories.py` | Galeria de 50 cenas, 7 por classe e 1 geral. |
| `render_experiment_comparison_galleries.py` | Galerias comparativas CLIP e DINO nos mesmos IDs. |
| `infer_dino_r50_finetuned_gallery.py` | Inferência do fine-tuning nos IDs da galeria DINO. |
| `render_saved_predictions.py` | Desenha predições salvas, usado pelas galerias. |
| `render_qualitative_predictions.py` | Figuras qualitativas a partir de checkpoints. Os rótulos de configuração no texto usam nomes antigos. |
