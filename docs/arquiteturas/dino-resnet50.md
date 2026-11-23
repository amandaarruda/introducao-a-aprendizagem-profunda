# DINO com ResNet-50

## O que é o DINO neste projeto

**DINO** aqui é o detector de objetos ponta a ponta de Zhang et al., *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection* ([arXiv:2203.03605](https://arxiv.org/abs/2203.03605), 2022; publicado no ICLR 2023). É um detector baseado em Transformer no estilo DETR:

- prediz diretamente o conjunto de caixas e classes, sem âncoras fixas e sem NMS;
- usa **consultas (queries)** aprendidas, que são associadas às caixas de verdade por um casamento húngaro durante o treino;
- introduz três melhorias sobre o DETR original: **denoising contrastivo** (consultas de ruído que ensinam o modelo a corrigir caixas perturbadas), **seleção mista de consultas** (consultas iniciais tiradas de propostas da primeira etapa) e **refinamento de caixas iterativo** (o "look forward twice").

## O que o DINO não é

Os nomes abaixo identificam métodos distintos:

| Nome | O que é | Relação com este projeto |
|---|---|---|
| **DINO (detector)** | Detector DETR de Zhang et al., 2022 | **É o usado.** |
| DINO (auto-supervisionado) | Método de pré-treino auto-supervisionado para Vision Transformers, de Caron et al., 2021 | Não usado. Não há pré-treino auto-supervisionado aqui. |
| DINOv3 | Modelo fundacional de visão da Meta (2025, [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)), com backbone ViT pré-treinado em grande escala | Não usado. |

Os pesos do backbone são ResNet-50 pré-treinados em ImageNet (arquivo `resnet50-11ad3fa6.pth`). Ou seja, a única parte pré-treinada é a ResNet; o restante do detector é treinado do zero sobre a DWD.

## Arquitetura

Configuração em [`configs/dino_dwd_full.py`](../../configs/dino_dwd_full.py), implementada pelo MMDetection (`type="DINO"`).

```text
imagem 640 px
   │
   ▼
ResNet-50 (ImageNet)                    ← stem e stage 1 congelados, BatchNorm congelado
   │  C3, C4, C5 (canais 512, 1024, 2048)
   ▼
ChannelMapper                           ← projeções 1×1 e nível extra 3×3, 256 canais
   │
   ▼
Encoder deformável (6 camadas)          ← atenção deformável multi-escala
   │
   ▼
Decoder (6 camadas, 300 consultas)      ← auto-atenção + atenção deformável cruzada
   │  + consultas de denoising dinâmicas durante o treino
   ▼
Cabeça: classificação (Focal Loss) + caixa (L1 e GIoU)
```

| Parâmetro | Valor |
|---|---|
| Consultas | 300 de detecção; denoising dinâmico configurado com `num_dn_queries=100` |
| Duas etapas e refinamento de caixas | sim (`as_two_stage`, `with_box_refine`) |
| Perdas | Focal Loss (α 0,25, γ 2; peso 1), L1 (peso 5), GIoU (peso 2) |
| Casamento | húngaro; custos Focal, L1 e GIoU com pesos 2, 5 e 2 |

## Treino do baseline

| Item | Valor |
|---|---|
| Entrada | 640 px, batch 1 |
| Otimizador | AdamW, lr 1e-4, weight decay 1e-4, multiplicador 0,1 no backbone |
| Corte de gradiente | norma 0,1 |
| Agenda | aquecimento linear de 1000 iterações, depois cosseno até 1e-6 |
| Duração | 30 000 iterações em `day_clear` |
| Semente | 1337 |

## Especialistas

Os quatro especialistas partem do checkpoint do baseline e são treinados por 12 000 iterações em seus domínios, com a mesma configuração de modelo.

## Fine-tuning dos especialistas

Esta etapa retreina **apenas** os quatro especialistas de domínio, a partir do checkpoint de 12 000 iterações de cada um. O baseline `day_clear` e a rota de teste original não foram alterados.

| Item | Valor |
|---|---|
| Inicialização | especialista do domínio, iteração 12 000 |
| Arquitetura, perdas, pipeline, resolução, semente | iguais ao treino original |
| Batch | 8 |
| Taxa de aprendizado | 8e-5 |
| Aquecimento / cosseno | 100 iterações / cosseno até 1e-6 |
| Duração máxima | 1000 atualizações |
| Validação | a cada 125 atualizações, na reserva de *model-validation* |
| Seleção | melhor `coco/bbox_mAP_50` por domínio |
| Parada antecipada | após 3 validações sem melhora de pelo menos 0,001 |

Checkpoint escolhido por domínio: DayFoggy na atualização 125, DuskRainy 125, NightClear 250 e NightRainy 250.

O AP50 aumenta em DuskRainy, NightClear e NightRainy e diminui levemente em DayFoggy. O COCO AP diminui em DayFoggy e DuskRainy. Os números são de *model-validation*, usados para selecionar checkpoints e parar o treino. **Não houve avaliação no conjunto de teste para esta etapa**, e por isso o MoE com especialistas fine-tuned não tem uma métrica de teste neste repositório. Ver [RESULTADOS.md](../RESULTADOS.md#fine-tuning-dos-especialistas-dino).

Configurações: [`configs/dino_dwd_finetune.py`](../../configs/dino_dwd_finetune.py) (base) e [`configs/dino_dwd_finetune_bs8_fullgpu.py`](../../configs/dino_dwd_finetune_bs8_fullgpu.py) (a usada, com batch 8). A parada antecipada usa o `EarlyStoppingHook` do MMEngine.

## Resultados

Ver [RESULTADOS.md](../RESULTADOS.md) e [experiments/dino-r50-moe](../../experiments/dino-r50-moe/README.md) e [experiments/dino-r50-finetune](../../experiments/dino-r50-finetune/README.md).
