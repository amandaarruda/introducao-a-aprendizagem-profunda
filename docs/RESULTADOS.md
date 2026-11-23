# Resultados

Este documento consolida as métricas de todos os experimentos. Os relatórios do baseline e do MoE, por domínio e com matrizes, estão em `resultados/` dos respectivos experimentos. O fine-tuning apresenta as métricas de validação no README do experimento.

## Teste congelado

| Detector | mAP50 baseline | mAP50 MoE | Δ mAP50 | COCO AP baseline | COCO AP MoE | Δ COCO AP |
|---|---:|---:|---:|---:|---:|---:|
| CLIP RN101 VLTDet | 20,24% | 23,35% | +3,11 p.p. | 10,83% | 13,73% | +2,90 p.p. |
| DINO ResNet-50 | 31,40% | **44,56%** | **+13,16 p.p.** | 17,22% | **24,87%** | **+7,65 p.p.** |

Cada valor é a média macro (sem ponderação) de cinco domínios, com 100 imagens de teste por domínio. A variante MoE usa o gate CLIP congelado com acurácia de 98,0% no teste.

Relatórios: [CLIP](../experiments/clip-rn101-vltdet/resultados/teste/RESULTS.md) e [DINO](../experiments/dino-r50-moe/resultados/teste/RESULTS.md).

## Validação (gate-val)

Esta reserva foi usada para validar o gate e o mapa de rotas. Não é estimativa independente.

| Detector | mAP50 baseline | mAP50 MoE | COCO AP baseline | COCO AP MoE | Acurácia do gate |
|---|---:|---:|---:|---:|---:|
| CLIP RN101 VLTDet | 21,34% | 25,16% | 12,02% | 14,00% | 97,6% |
| DINO ResNet-50 | 32,16% | 37,51% | 18,41% | 20,83% | 97,6% |

## Por domínio (teste)

**mAP50**

| Domínio | CLIP baseline | CLIP MoE | DINO baseline | DINO MoE |
|---|---:|---:|---:|---:|
| day_clear | 25,22% | 25,41% | 45,11% | 44,87% |
| day_foggy | 20,41% | 25,15% | 24,41% | 35,53% |
| dusk_rainy | 18,97% | 23,13% | 33,57% | 51,09% |
| night_clear | 28,27% | 27,55% | 44,85% | 51,03% |
| night_rainy | 8,35% | 15,53% | 9,06% | 40,30% |

**COCO AP (AP@[.5:.95])**

| Domínio | CLIP baseline | CLIP MoE | DINO baseline | DINO MoE |
|---|---:|---:|---:|---:|
| day_clear | 15,08% | 15,23% | 24,69% | 24,48% |
| day_foggy | 11,78% | 14,66% | 13,80% | 20,83% |
| dusk_rainy | 9,21% | 14,26% | 18,59% | 24,40% |
| night_clear | 13,73% | 15,29% | 24,55% | 30,40% |
| night_rainy | 4,35% | 9,21% | 4,48% | 24,23% |

## Fine-tuning dos especialistas DINO

Métrica de *model-validation*, por domínio fine-tuned. Não há avaliação de teste para esta etapa.

| Domínio | AP original | AP fine-tuned | Δ AP | AP50 original | AP50 fine-tuned | Δ AP50 |
|---|---:|---:|---:|---:|---:|---:|
| DayFoggy | 0,249 | 0,242 | −0,007 | 0,406 | 0,405 | −0,001 |
| DuskRainy | 0,258 | 0,252 | −0,006 | 0,457 | 0,465 | +0,008 |
| NightClear | 0,197 | 0,259 | +0,062 | 0,329 | 0,434 | +0,105 |
| NightRainy | 0,191 | 0,205 | +0,014 | 0,416 | 0,419 | +0,003 |
| **Média não ponderada** | **0,224** | **0,240** | **+0,016** | **0,402** | **0,431** | **+0,029** |

O ganho médio vem de NightClear. AP cai levemente em DayFoggy e DuskRainy. Os checkpoints foram escolhidos por domínio com base nessa mesma reserva, então os números não são avaliação independente. Ver [dino-resnet50.md](arquiteturas/dino-resnet50.md#fine-tuning-dos-especialistas).

## Definições

- **mAP50**: média, entre as classes, da precisão média com IoU ≥ 0,5 (protocolo COCO, 101 pontos de interpolação). Os valores são percentuais.
- **COCO AP (AP@[.5:.95])**: média do AP em dez limiares de IoU, de 0,50 a 0,95 com passo de 0,05.
- **Macro**: média simples entre os cinco domínios, sem ponderar pelo número de objetos.
- **p.p.**: pontos percentuais.
- **Limiar de pontuação**: 0,05 nas avaliações MoE. As métricas de model-validation vêm do avaliador COCO do MMDetection.

## Limitações da comparação

- As famílias não são uma ablação controlada. Arquitetura, inicialização e agenda de treino mudam ao mesmo tempo.
- O teste tem 100 imagens por domínio. Diferenças pequenas (poucos pontos percentuais) não devem ser interpretadas sem intervalo de incerteza.
- Para DuskRainy e NightRainy, as reservas são do projeto, não listas oficiais da DWD.
- As duas seleções de galerias usam critérios baseados no resultado do DINO. São ilustrações qualitativas, não amostras representativas, e não alteram nenhuma métrica.
