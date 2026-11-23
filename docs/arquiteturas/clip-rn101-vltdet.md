# CLIP RN101 com neck ViTDet (VLTDet)

Esta arquitetura segue a configuração **VLTDet** do artigo *Strong but simple: A Baseline for Domain Generalized Dense Perception by CLIP-based Transfer Learning* (Hümmer et al., ACCV 2024; [arXiv:2312.02021](https://arxiv.org/abs/2312.02021)). No artigo, VLTDet é o nome das configurações baseadas em CLIP; para detecção, usa o decoder ViTDet sem a cabeça de máscara. O objetivo do artigo é generalização de domínio, e é isso que motiva usar um encoder pré-treinado em linguagem e imagem.

Aqui, a arquitetura é tratada como a reprodução de referência. Algumas escolhas de implementação não estão descritas no artigo e estão explicitadas nas configurações (ver [Pontos de interpretação](#pontos-de-interpretação)).

## Componentes

```text
imagem 640 px
   │
   ▼
ResNet-101 do CLIP (stem + layer1..layer4)     ← pesos CLIP RN101 (OpenCLIP)
   │  apenas C5: stride 32, 2048 canais
   ▼
ViTDet Simple Feature Pyramid (SFP)            ← 256 canais em 4 níveis
   │  strides 8, 16, 32, 64
   ▼
RPN (âncoras, 3 razões de aspecto)
   │  propostas
   ▼
Cascade R-CNN: 3 estágios (RoIAlign 7×7 + 2 FC)
   │  IoU 0.5 → 0.6 → 0.7, pesos de perda 1.0 / 0.5 / 0.25
   ▼
7 classes + caixas
```

### Backbone: ResNet-101 do CLIP

O encoder é a parte visual do CLIP com arquitetura ResNet-101 (`RN101`), carregada via OpenCLIP a partir do checkpoint `RN101.pt`. A implementação está em [`src/ligia_vltdet/models/openclip_resnet.py`](../../src/ligia_vltdet/models/openclip_resnet.py):

- o stem (três convoluções com `avgpool`) e os quatro estágios residuais são usados como extrator de características;
- apenas a saída do último estágio (C5, stride 32, 2048 canais) é passada ao neck (`out_indices=(3,)`);
- a cabeça de atenção do CLIP (`attnpool`) **não** é usada pelo detector. Ela só aparece no gate, que usa o embedding de imagem do CLIP (ver [mixture-of-experts.md](mixture-of-experts.md)).

### Neck: ViTDet Simple Feature Pyramid

Como o backbone CLIP entrega um único mapa de características, o neck cria uma pirâmide a partir dele, no estilo ViTDet ([`src/ligia_vltdet/models/vitdet_sfp.py`](../../src/ligia_vltdet/models/vitdet_sfp.py)):

| Nível | Operação sobre C5 | Stride resultante |
|---|---|---|
| P2 | 2 × deconvolução 2×2 (com GroupNorm e GELU) + 1×1 | 8 |
| P3 | deconvolução 2×2 (com GroupNorm e GELU) + 1×1 | 16 |
| P4 | 1×1 | 32 |
| P5 | max-pool 2×2 + 1×1 | 64 |

Todos os níveis têm 256 canais.

### Detector: Cascade R-CNN

O detector é o `CascadeRCNN` do MMDetection:

- **RPN** com âncoras de escala 4 e razões 0,5, 1 e 2, nos strides 8 a 64;
- **RoI head em cascata** com três estágios, cada um com RoIAlign 7×7 e duas camadas totalmente conectadas (1024 unidades), regressão de caixa independente de classe;
- limiares de IoU crescentes entre estágios (0,5, 0,6 e 0,7), que refinam as caixas em sequência;
- saída: 7 classes, limiar de pontuação 0,05 na avaliação, no máximo 100 detecções por imagem.

## Treino

Configuração em [`configs/vltdet_r101_clip_vitdet_smoke.py`](../../configs/vltdet_r101_clip_vitdet_smoke.py) e herdeiros (ver [configs/README.md](../../configs/README.md)):

| Item | Valor |
|---|---|
| Entrada | redimensionamento para 640 px, mantendo proporção |
| Batch | 1 |
| Otimizador | AdamW, lr 1e-4, weight decay 0,05, corte de gradiente 10 |
| Agenda | aquecimento linear de 3000 iterações, depois cosseno até 1e-6 |
| Duração (baseline) | 30 000 iterações em `day_clear` |
| Aumentos | Mosaic, RandomAffine, MixUp, RandAugment (cor, contraste, brilho, nitidez), flip horizontal |
| Semente | 1337 |

### Pontos de interpretação

- **Decaimento de taxa por camada.** O artigo descreve um fator por camada, mas não a correspondência com os estágios de uma ResNet. A configuração aplica o fator 0,7 por estágio do CLIP: `layer4` com multiplicador 1,0, `layer3` com 0,7, `layer2` com 0,7², `layer1` com 0,7³ e o stem com 0,7⁴.

## Experts (MoE)

Os quatro especialistas (`day_foggy`, `dusk_rainy`, `night_clear`, `night_rainy`) são inicializados a partir do baseline e treinados por 12 000 iterações em seus pools de domínio. O gate e o mapa de rota estão em [mixture-of-experts.md](mixture-of-experts.md).

## Resultados

Ver [RESULTADOS.md](../RESULTADOS.md) e [experiments/clip-rn101-vltdet](../../experiments/clip-rn101-vltdet/README.md).
