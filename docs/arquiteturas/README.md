# Arquiteturas

O projeto compara duas famílias de detectores de objetos, com variantes de baseline e especialistas por domínio:

| Família | Detector | Backbone | Variantes |
|---|---|---|---|
| CLIP RN101 (VLTDet) | Cascade R-CNN com neck ViTDet | ResNet-101 do CLIP (OpenCLIP) | baseline, MoE |
| DINO ResNet-50 | DINO (detector DETR) | ResNet-50 (ImageNet) | baseline, MoE, MoE com especialistas fine-tuned |

Os detectores são treinados sobre a base DWD, com sete classes (`bus`, `bike`, `car`, `motor`, `person`, `rider`, `truck`) e cinco domínios de condição:

| Domínio | Condição |
|---|---|
| `day_clear` | dia, céu limpo (domínio de treino do baseline) |
| `day_foggy` | dia, neblina |
| `dusk_rainy` | entardecer, chuva |
| `night_clear` | noite, céu limpo |
| `night_rainy` | noite, chuva |

## Baseline

Um único detector treinado apenas com `day_clear` é avaliado nos cinco domínios. É a referência contra a qual as demais variantes são comparadas.

## Mixture of Experts (MoE)

Um **gate** (roteador) classifica o domínio de cada imagem e a envia para **um** detector especializado. O domínio previsto `day_clear` aponta para o baseline; os demais apontam para os especialistas correspondentes. Ver [mixture-of-experts.md](mixture-of-experts.md).

```text
                     imagem
                        │
                        ▼
        gate: CLIP RN101 congelado + camada linear
                        │  domínio previsto (argmax)
    ┌──────────┬────────┴─────────┬─────────────┬──────────────┐
    ▼          ▼                  ▼             ▼              ▼
baseline   especialista      especialista  especialista  especialista
day_clear  day_foggy         dusk_rainy    night_clear   night_rainy
    │          │                  │             │              │
    └──────────┴──────────────────┴──────┬──────┴──────────────┘
                                         ▼
                              detecções (7 classes)
```

## Fine-tuning

Na família DINO, os quatro especialistas de domínio (não o baseline) foram treinados novamente com um cronograma curto e taxa de aprendizado menor. O detalhe está em [dino-resnet50.md](dino-resnet50.md#fine-tuning-dos-especialistas).

## Documentos

- [clip-rn101-vltdet.md](clip-rn101-vltdet.md): backbone CLIP, neck ViTDet, Cascade R-CNN e treino.
- [dino-resnet50.md](dino-resnet50.md): o que é o DINO aqui (e o que não é), arquitetura, treino e fine-tuning.
- [mixture-of-experts.md](mixture-of-experts.md): gate, rotas, avaliação e limitações.
