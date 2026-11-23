# DINO ResNet-50: baseline e MoE

Detector DINO (DETR com denoising e consultas aprendidas) com backbone ResNet-50, 640 px, batch 1, semente 1337. Arquitetura, treino e especialistas em [docs/arquiteturas/dino-resnet50.md](../../docs/arquiteturas/dino-resnet50.md).

- **Baseline:** 30 000 iterações em `day_clear`.
- **MoE:** o baseline mais quatro especialistas, cada um treinado por 12 000 iterações a partir do baseline no pool do seu domínio. O gate CLIP RN101 congelado escolhe a rota.

## Resultados (teste congelado)

| Domínio | AP50 baseline | AP50 MoE | Δ | COCO AP baseline | COCO AP MoE |
|---|---:|---:|---:|---:|---:|
| day_clear | 45,11% | 44,87% | −0,24 p.p. | 24,69% | 24,48% |
| day_foggy | 24,41% | 35,53% | +11,12 p.p. | 13,80% | 20,83% |
| dusk_rainy | 33,57% | 51,09% | +17,52 p.p. | 18,59% | 24,40% |
| night_clear | 44,85% | 51,03% | +6,18 p.p. | 24,55% | 30,40% |
| night_rainy | 9,06% | 40,30% | +31,24 p.p. | 4,48% | 24,23% |
| **Média macro** | **31,40%** | **44,56%** | **+13,16 p.p.** | **17,22%** | **24,87%** |

Gate: 490/500 no teste (98,0%). Em gate-val, a média macro vai de 32,16% para 37,51% de AP50 e de 18,41% para 20,83% de COCO AP. Essa reserva serviu para validar o gate e não é estimativa independente.

Na matriz de especialistas do teste, o detector associado a cada domínio tem o maior AP50 quando aplicado a todas as imagens desse domínio. Ver [mixture-of-experts.md](../../docs/arquiteturas/mixture-of-experts.md).

Relatórios: [teste](resultados/teste/RESULTS.md) · [validação do gate](resultados/validacao-gate/RESULTS.md) · [modelos e rotas](mapas-congelados/)

Relatório consolidado da execução: [docs/RESULTADOS.md](../../docs/RESULTADOS.md).

## Galerias

Cada figura mostra o ground truth, o baseline DINO e o especialista da rota.

- [50 por domínio](galerias/por-dominio/index.html) (10 cenas de teste por domínio)
- [50 por categoria](galerias/por-categoria/index.html) (7 cenas por classe e 1 geral)

As imagens das duas galerias foram selecionadas pelo resultado. São ilustrações qualitativas.

## Rastreabilidade

- Campanha `16511` (baseline e especialistas), preparação dos splits `16510`, gate-val `16535`, teste final `16514`.
- SHA-256 do baseline: `30c56261f37d408a7d68d0d357e9d91d3892ffdf76a70a3c017459db2264dca3`. Os dos especialistas estão em [DADOS_E_CHECKPOINTS.md](../../docs/DADOS_E_CHECKPOINTS.md).
- Esta é uma adaptação, não uma reprodução direta do artigo: arquitetura, inicialização e agenda de treino mudam em conjunto, e as reservas de DuskRainy e NightRainy são do projeto.

## Estrutura

```text
resultados/teste/            relatórios e JSON do teste
resultados/validacao-gate/   relatórios e JSON do gate-val
galerias/                    figuras (por domínio e por categoria)
mapas-congelados/            route_map.json e models.json
arquivo-execucao/configs/    configs resolvidos
arquivo-execucao/logs/       logs de treino, avaliação e Slurm
arquivo-execucao/predicoes/  predições por imagem e rotas do gate
```
