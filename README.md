# Detecção de objetos em condições climáticas e de iluminação adversas

Detectores de objetos treinados em imagens de dia com céu limpo costumam perder desempenho em neblina, chuva e à noite. Este repositório compara, nessas condições, dois detectores — **CLIP RN101 com neck ViTDet (VLTDet)** e **DINO com ResNet-50** — em três variantes: **baseline**, **Mixture of Experts (MoE)** por domínio e, no caso do DINO, **fine-tuning dos especialistas**.

- **Base:** DWD (Diverse Weather Dataset), 7 classes (`bus`, `bike`, `car`, `motor`, `person`, `rider`, `truck`) e 5 domínios.
- **Referência:** VLTDet, do artigo [arXiv:2312.02021](https://arxiv.org/abs/2312.02021), reproduzido com CLIP RN101.
- **Proposta:** substituir o detector por DINO com ResNet-50 e combinar especialistas por domínio por meio de um gate.

## Resultados (teste congelado)

Cada domínio tem 100 imagens de teste. Os valores são médias macro, sem ponderação, entre os cinco domínios.

| Detector | Variante | mAP50 macro | COCO AP macro |
|---|---|---:|---:|
| CLIP RN101 VLTDet | baseline | 20,24% | 10,83% |
| CLIP RN101 VLTDet | MoE | 23,35% | 13,73% |
| DINO ResNet-50 | baseline | 31,40% | 17,22% |
| **DINO ResNet-50** | **MoE** | **44,56%** | **24,87%** |

Por domínio (mAP50, teste):

| Domínio | CLIP baseline | CLIP MoE | DINO baseline | DINO MoE |
|---|---:|---:|---:|---:|
| day_clear | 25,22% | 25,41% | 45,11% | 44,87% |
| day_foggy | 20,41% | 25,15% | 24,41% | 35,53% |
| dusk_rainy | 18,97% | 23,13% | 33,57% | 51,09% |
| night_clear | 28,27% | 27,55% | 44,85% | 51,03% |
| night_rainy | 8,35% | 15,53% | 9,06% | 40,30% |

- O gate que escolhe o especialista acerta **98,0%** das imagens de teste (490/500) nas duas famílias. Em validação, a acurácia é 97,6%.
- O maior ganho do MoE está em `night_rainy` (de 9,06% para 40,30% no DINO), o domínio mais difícil do baseline.
- O **fine-tuning** dos especialistas DINO eleva a média de AP50 de validação dos quatro domínios de 40,2% para 43,1%, com ganho concentrado em `night_clear`. Essa etapa **não tem avaliação de teste**, por isso não aparece na tabela acima.

Detalhes, definições de métrica e limitações: [docs/RESULTADOS.md](docs/RESULTADOS.md).

## Artigo e abordagem

O artigo de Hümmer et al., [*Strong but simple: A Baseline for Domain Generalized Dense Perception by CLIP-based Transfer Learning*](https://arxiv.org/abs/2312.02021), estuda a generalização para domínios diferentes dos usados no treino. Sua abordagem ajusta integralmente modelos visuais pré-treinados com linguagem e imagem. Para detecção, emprega um decoder baseado em ViTDet.

A referência experimental deste projeto usa o encoder visual CLIP RN101, uma pirâmide de características no estilo ViTDet e um detector Cascade R-CNN. Ela estabelece o baseline treinado em `day_clear` e avaliado nos cinco domínios. A variante MoE acrescenta quatro detectores especializados e um gate que escolhe a rota de cada imagem.

A proposta substitui o detector por DINO com ResNet-50 e aplica o mesmo esquema de especialistas. Na comparação do teste, o DINO MoE alcança 44,56% de mAP50 macro, frente a 31,40% do seu baseline e 23,35% do CLIP MoE. Arquitetura, pré-treino e treino mudam entre as famílias, portanto a comparação mostra o resultado das configurações completas. O fine-tuning dos especialistas DINO é analisado separadamente na validação.

O conjunto `model-validation` seleciona checkpoints dos detectores. O `gate_train` treina o roteador, o `gate_val` mede e seleciona o gate, e o `test` é reservado à avaliação das variantes congeladas. Veja o [protocolo](docs/PROTOCOLO.md) e os [detalhes das arquiteturas](docs/arquiteturas/README.md).

## Arquiteturas

| Família | Detector | Backbone | Como funciona |
|---|---|---|---|
| CLIP RN101 VLTDet | Cascade R-CNN | ResNet-101 do CLIP | Neck ViTDet cria uma pirâmide de 4 níveis a partir do último estágio do backbone |
| DINO ResNet-50 | DINO (detector DETR) | ResNet-50 (ImageNet) | Transformer ponta a ponta com consultas de denoising, sem âncoras nem NMS |

**Atenção:** o DINO deste projeto é o detector de Zhang et al. (2022), não o método auto-supervisionado DINO nem o DINOv3.

As explicações completas estão em [docs/arquiteturas](docs/arquiteturas/README.md):

- [CLIP RN101 com neck ViTDet](docs/arquiteturas/clip-rn101-vltdet.md)
- [DINO com ResNet-50, o que é e o que não é, e fine-tuning](docs/arquiteturas/dino-resnet50.md)
- [Mixture of Experts: gate, rotas e avaliação](docs/arquiteturas/mixture-of-experts.md)

## Experimentos

| Experimento | Conteúdo | Leia primeiro |
|---|---|---|
| [clip-rn101-vltdet](experiments/clip-rn101-vltdet/) | baseline e MoE CLIP | [README](experiments/clip-rn101-vltdet/README.md) |
| [dino-r50-moe](experiments/dino-r50-moe/) | baseline e MoE DINO | [README](experiments/dino-r50-moe/README.md) |
| [dino-r50-finetune](experiments/dino-r50-finetune/) | fine-tuning dos especialistas DINO | [README](experiments/dino-r50-finetune/README.md) |

Os experimentos de baseline e MoE contêm `resultados/`, `galerias/`, `mapas-congelados/` e `arquivo-execucao/`. O experimento de fine-tuning apresenta as métricas em seu README e reúne as figuras e os registros em `galerias/` e `arquivo-execucao/`.

## Estrutura do repositório

```text
configs/          configurações MMDetection (CLIP e DINO) e a cadeia de herança
docs/             arquiteturas, protocolo, resultados, execução, dados e atribuições
experiments/      um diretório por experimento (ver acima)
scripts/          divisão dos dados, treino e avaliação do gate, avaliação MoE e galerias
slurm/            launchers usados no cluster Slurm
src/              pacote ligia_vltdet: backbone CLIP e neck ViTDet para MMDetection
data/             dados locais (não versionados)
checkpoints/      pesos externos (não versionados)
```

## Reprodução

Este repositório não inclui a base DWD, as anotações COCO, os checkpoints nem os ambientes virtuais. As novas execuções usam um diretório local `outputs/`, criado antes de submeter os jobs. Os relatórios e as figuras são resultados já produzidos. Instalar o projeto não recria esses números.

Para executar, veja [docs/EXECUCAO.md](docs/EXECUCAO.md) (ambiente, variáveis, ordem dos jobs) e [docs/DADOS_E_CHECKPOINTS.md](docs/DADOS_E_CHECKPOINTS.md) (dados e pesos esperados, com hashes).

## Protocolo

- Cada domínio tem reservas de 100 imagens para *model-validation*, `gate_train`, `gate_val` e `test`. O restante do pool elegível é usado no treino.
- Modelos e rotas do MoE foram fixados antes da leitura do teste. O teste foi lido uma vez por variante.
- DuskRainy e NightRainy usam reservas definidas pelo projeto, porque a DWD local não traz listas oficiais de teste para esses domínios.
- As duas famílias não são uma ablação de uma única variável: arquitetura, inicialização e agenda de treino mudam juntas. Ver [docs/PROTOCOLO.md](docs/PROTOCOLO.md).

## Licença e atribuição

Código próprio sob [MIT](LICENSE). A base DWD, os pesos CLIP e ResNet-50, o MMDetection e as imagens das galerias seguem termos próprios. Ver [docs/ATRIBUICAO.md](docs/ATRIBUICAO.md).
