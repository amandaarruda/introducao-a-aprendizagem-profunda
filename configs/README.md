# Configurações

Configurações MMDetection usadas pelos experimentos. Os caminhos de dados e checkpoints entram por variáveis de ambiente (`DWD_ROOT`, `CLIP_RN101_CHECKPOINT`, `DINO_R50_CHECKPOINT`), então os arquivos não contêm caminhos pessoais.

| Arquivo | Família | Papel |
|---|---|---|
| `vltdet_r101_clip_vitdet_smoke.py` | CLIP | Base da arquitetura: backbone CLIP RN101, neck ViTDet, Cascade R-CNN, pipeline e otimizador. Define 500 iterações por padrão, que os launchers sobrescrevem. |
| `vltdet_r101_clip_dayclear_1000_640_6000.py` | CLIP | Elo intermediário da herança: validação a cada 1000 iterações e checkpoint do melhor modelo. |
| `vltdet_r101_clip_dayclear_5000_640_30000.py` | CLIP | **Configuração efetiva.** Baseline `day_clear` (30 000 iterações) e especialistas (12 000). |
| `dino_dwd_full.py` | DINO | **Configuração efetiva.** Baseline, especialistas e avaliação MoE. |
| `dino_dwd_finetune.py` | DINO | Base do fine-tuning: herda `dino_dwd_full.py` e troca a taxa, a agenda e o hook de parada. |
| `dino_dwd_finetune_bs8_fullgpu.py` | DINO | **Usada no fine-tuning**: batch 8, 1000 atualizações, validação a cada 125 e parada antecipada. |

O `EarlyStoppingHook` é fornecido pelo MMEngine. Os configs resolvidos de cada execução estão em `experiments/*/arquivo-execucao/configs/`.
