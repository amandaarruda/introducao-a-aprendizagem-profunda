# CLIP RN101 VLTDet: baseline e MoE

Detector Cascade R-CNN com backbone ResNet-101 do CLIP e neck ViTDet. Arquitetura e treino em [docs/arquiteturas/clip-rn101-vltdet.md](../../docs/arquiteturas/clip-rn101-vltdet.md).

- **Baseline:** treinado por 30 000 iterações em `day_clear` (19 095 imagens elegíveis), 640 px, batch 1, semente 1337.
- **MoE:** o baseline mais quatro especialistas (`day_foggy`, `dusk_rainy`, `night_clear`, `night_rainy`), cada um inicializado pelo baseline e treinado por 12 000 iterações no pool do seu domínio. A rota é feita pelo gate CLIP RN101 congelado.

## Resultados (teste congelado)

| Domínio | AP50 baseline | AP50 MoE | COCO AP baseline | COCO AP MoE |
|---|---:|---:|---:|---:|
| day_clear | 25,22% | 25,41% | 15,08% | 15,23% |
| day_foggy | 20,41% | 25,15% | 11,78% | 14,66% |
| dusk_rainy | 18,97% | 23,13% | 9,21% | 14,26% |
| night_clear | 28,27% | 27,55% | 13,73% | 15,29% |
| night_rainy | 8,35% | 15,53% | 4,35% | 9,21% |
| **Média macro** | **20,24%** | **23,35%** | **10,83%** | **13,73%** |

Acurácia do gate: 97,6% em gate-val e 98,0% no teste. O MoE melhora a média de AP50 em 3,11 pontos percentuais. O ganho é grande em `day_foggy`, `dusk_rainy` e `night_rainy`. Em `night_clear` há queda de 0,7 p.p., e em `day_clear` a diferença é de 0,2 p.p.

Relatórios: [teste](resultados/teste/RESULTS.md) · [validação do gate](resultados/validacao-gate/RESULTS.md) · [mapa de rota](mapas-congelados/route_map.json)

## Galerias

As figuras usam os mesmos IDs da galeria DINO. Cada cena mostra o ground truth, o detector de referência e a previsão CLIP.

- Baseline: [50 por domínio](galerias/baseline/por-dominio/index.html) · [50 por categoria](galerias/baseline/por-categoria/index.html)
- MoE: [50 por domínio](galerias/moe/por-dominio/index.html) · [50 por categoria](galerias/moe/por-categoria/index.html)

As imagens das duas galerias foram selecionadas com base nos resultados do DINO. São ilustrações qualitativas, não uma amostra representativa.

## Estrutura

```text
resultados/teste/            relatórios e JSON do teste
resultados/validacao-gate/   relatórios e JSON do gate-val
galerias/                    figuras (baseline e MoE)
mapas-congelados/            rota por domínio
arquivo-execucao/configs/    configs resolvidos
arquivo-execucao/logs/       logs de treino, avaliação e Slurm
arquivo-execucao/predicoes/  predições por imagem e rotas do gate
```

Pesos e dados de entrada não estão incluídos. Ver [docs/EXECUCAO.md](../../docs/EXECUCAO.md) e [docs/DADOS_E_CHECKPOINTS.md](../../docs/DADOS_E_CHECKPOINTS.md).
