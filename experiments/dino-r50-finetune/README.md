# DINO ResNet-50: fine-tuning dos especialistas

Retreino curto dos quatro especialistas DINO do [dino-r50-moe](../dino-r50-moe/README.md). O baseline `day_clear` e a rota de teste originais não foram alterados.

Configuração (batch 8, taxa 8e-5, até 1000 atualizações, validação a cada 125, parada antecipada com paciência 3) em [docs/arquiteturas/dino-resnet50.md](../../docs/arquiteturas/dino-resnet50.md#fine-tuning-dos-especialistas).

## Resultados (model-validation)

| Domínio | AP original | AP fine-tuned | Δ AP | AP50 original | AP50 fine-tuned | Δ AP50 | Job / atualização |
|---|---:|---:|---:|---:|---:|---:|---|
| DayFoggy | 0,249 | 0,242 | −0,007 | 0,406 | 0,405 | −0,001 | 32556 / 125 |
| DuskRainy | 0,258 | 0,252 | −0,006 | 0,457 | 0,465 | +0,008 | 32558 / 125 |
| NightClear | 0,197 | 0,259 | +0,062 | 0,329 | 0,434 | +0,105 | 32560 / 250 |
| NightRainy | 0,191 | 0,205 | +0,014 | 0,416 | 0,419 | +0,003 | 32562 / 250 |
| **Média não ponderada** | **0,224** | **0,240** | **+0,016** | **0,402** | **0,431** | **+0,029** | — |

**Não há avaliação de teste para o fine-tuning.** Os números são da reserva de model-validation, usada para escolher o checkpoint e parar o treino, então não são estimativa independente. O ganho médio é concentrado em NightClear. O AP cai levemente em DayFoggy e DuskRainy, enquanto o AP50 aumenta em DuskRainy.

## Galerias

Cada figura mostra o ground truth, o MoE DINO original e a rota com os especialistas fine-tuned, nos mesmos IDs da galeria do [dino-r50-moe](../dino-r50-moe/galerias/).

- [50 por domínio](galerias/por-dominio/index.html)
- [50 por categoria](galerias/por-categoria/index.html)

As figuras são qualitativas.

## Estrutura

```text
galerias/                         figuras (por domínio e por categoria)
arquivo-execucao/configs/         configs resolvidos de cada domínio
arquivo-execucao/logs/            logs de treino e validação, incluindo as paradas antecipadas
arquivo-execucao/hashes/          SHA-256 das entradas e dos checkpoints selecionados
```

Os pesos dos checkpoints não estão incluídos. Os hashes estão em `arquivo-execucao/hashes/`.
