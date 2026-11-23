# Mixture of Experts por domínio

A variante MoE troca um único detector por um conjunto de detectores, um por domínio, e usa um **gate** para escolher qual deles processa cada imagem. Isso é roteamento top-1 (*hard routing*): cada imagem passa por exatamente um detector.

## Gate

O gate é uma regressão logística linear treinada sobre embeddings de imagem do **CLIP RN101 congelado**:

1. a imagem é codificada pelo `encode_image` do OpenCLIP RN101, produzindo um vetor de 512 dimensões;
2. o vetor é normalizado em L2;
3. uma camada linear `512 → 5` produz logits; o softmax os converte em probabilidades, e a classe de maior probabilidade define a rota.

Código: [`scripts/train_domain_gate.py`](../../scripts/train_domain_gate.py) (treino) e [`scripts/evaluate_moe_gate.py`](../../scripts/evaluate_moe_gate.py) (avaliação MoE).

O gate é treinado só com `gate_train` e validado em `gate_val`. O conjunto `test` não é lido até o gate, o mapa de rotas e os modelos estarem congelados.

## Rotas

| Domínio previsto | Detector usado (CLIP) | Detector usado (DINO) |
|---|---|---|
| `day_clear` | baseline `day_clear` | baseline `day_clear` |
| `day_foggy` | especialista `day_foggy` | especialista `day_foggy` |
| `dusk_rainy` | especialista `dusk_rainy` | especialista `dusk_rainy` |
| `night_clear` | especialista `night_clear` | especialista `night_clear` |
| `night_rainy` | especialista `night_rainy` | especialista `night_rainy` |

Os mapas de rota estão em `mapas-congelados/route_map.json` de cada experimento. Na variante DINO, o `models.json` lista os caminhos lógicos dos checkpoints.

As duas famílias usam o **mesmo gate CLIP**, com o mesmo checkpoint (SHA-256 `fd56183c…7e7`, ver [DADOS_E_CHECKPOINTS.md](../DADOS_E_CHECKPOINTS.md)). Assim, a diferença entre MoE CLIP e MoE DINO vem do detector, não do roteador.

## Desempenho do gate

| Conjunto | CLIP | DINO |
|---|---:|---:|
| gate-val | 97,6% | 97,6% |
| teste | 98,0% (490/500) | 98,0% (490/500) |

A matriz de confusão do teste está em cada `resultados/teste/RESULTS.md`. Os erros são poucos e ficam entre domínios vizinhos: `dusk_rainy` e `night_rainy`, principalmente.

## Por que o MoE ajuda

A matriz de especialistas, gerada na avaliação de teste, mostra o desempenho de **cada** detector em **cada** domínio. Na família DINO, o detector associado a cada domínio apresenta o maior AP50 da respectiva linha. Essa comparação aplica cada detector a todas as imagens do domínio. O gate escolhe uma rota por imagem, e seus erros também podem afetar o resultado.

## Limitações

- Roteamento *hard*: se o gate erra, a imagem vai para o detector errado, sem combinação de saídas.
- O gate é treinado só com a classificação de domínio, não com o desempenho de detecção.
