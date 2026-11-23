# Dados e checkpoints externos

Este repositório não contém a base DWD completa, as anotações COCO, os pesos treinados nem os dumps completos de execução (que ficam em `outputs/` e fora do Git por tamanho). Para os experimentos selecionados, `experiments/*/arquivo-execucao/` guarda configs resolvidos, logs finais, predições por imagem e hashes. Os relatórios e as rotas permitem conferir os valores publicados. O recálculo de AP requer as anotações, e a matriz completa de especialistas também requer os respectivos arquivos de predições, que não estão incluídos. Use apenas dados e pesos obtidos de fontes autorizadas.

## Estrutura local esperada

```text
data/raw/dwd/full-train-frozen-eval-20261005/
  split_manifest.json
  day_clear/     day_clear_train.coco.json, day_clear_val.coco.json, day_clear_gate_train.coco.json, day_clear_gate_val.coco.json, day_clear_test.coco.json
  day_foggy/     ...
  dusk_rainy/    ...
  night_clear/   ...
  night_rainy/   ...
checkpoints/
  RN101.pt                 # OpenCLIP RN101: backbone do CLIP e gate congelado
  resnet50-11ad3fa6.pth    # ResNet-50 pré-treinada em ImageNet: inicialização do DINO
outputs/                   # pesos de detectores, gate, logs e predições
```

Os COCO JSONs usados nas execuções de referência tinham `file_name` absolutos. Ajustar esses caminhos altera o hash do arquivo. Para conferir a equivalência, compare os IDs, anotações, divisões e imagens com os dados originais. O hash original só é reproduzido com os mesmos bytes.

## Caminhos nos artefatos arquivados

Os caminhos de entrada nos artefatos arquivados usam o marcador `<PROJECT_ROOT>`. Algumas referências em JSON usam `<experiment-root>`. Substitua os marcadores pelos caminhos apropriados ao usar esses arquivos. Os hashes de arquivos externos identificam os bytes originais.

## Identificadores e hashes

**DINO (baseline e especialistas)**: campanha `16511`; preparação dos splits `16510`; gate-val `16535`; teste `16514`.

- Baseline DINO (SHA-256): `30c56261f37d408a7d68d0d357e9d91d3892ffdf76a70a3c017459db2264dca3`
- Especialistas DINO originais (SHA-256):
  - DayFoggy: `413762df7f93ba31d50e6352d71ca4151c94463ead72b5013c38c0f5b4597e54`
  - DuskRainy: `08f7cab81eaaf487774365127c49335429b39d46454ef6c27237c59aa71f1561`
  - NightClear: `32f505a4aaff92751a3f87a2a371bdfccb375c94eb70b5c30e214337cd5c2a97`
  - NightRainy: `e8347ab73321e4d7227b944b7bcdc7d49f50016cee21739ca218960da9e3fa31`

**CLIP**: baseline `32364`; especialistas `32418`; gate-val `32453`; teste `32459`. Hashes adicionais estão nos relatórios de resultado.

- Checkpoint inicial CLIP RN101 (`RN101.pt`, SHA-256): `8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599`
- Gate CLIP congelado (SHA-256): `fd56183ca92380d2be99254e318399e72c28fcae035d6876d112595a645777e7`

**Fine-tuning DINO**: jobs `32556` (DayFoggy), `32558` (DuskRainy), `32560` (NightClear), `32562` (NightRainy). Os hashes dos checkpoints selecionados estão em `experiments/dino-r50-finetune/arquivo-execucao/hashes/`.

Checkpoints DINO fine-tuned selecionados (SHA-256): DayFoggy `271137f335261174e99cd74b428a8fa90f4865bf1ba36fb712acbc03e1bcee49`; DuskRainy `67c9ebebef31f43d518e1b28c07d2282a98ef3bb0a1281ddf6e774741bedaf79`; NightClear `ca03498c378a4000eba010e0edc9abda6b36a0e99d75d884937eb10e32f1f08a`; NightRainy `b3f97f166d335888aa3ac33ef07e5c135cf32e595ca0a4436e1dab6ed1873692`.

Os arquivos `hashes/*-selected-checkpoint.sha256` registram os pesos escolhidos sem distribuí-los. Os hashes identificam artefatos externos, mas não dão permissão para redistribuí-los. Os mapas de rota versionados usam nomes lógicos. O `models.json` usa `<experiment-root>` no lugar dos caminhos locais.

## Código das campanhas

Os hashes registrados nas avaliações históricas identificam versões de `scripts/evaluate_moe_gate.py` diferentes da versão distribuída. O hash histórico de `configs/vltdet_r101_clip_vitdet_smoke.py` também difere do arquivo disponível. Os configs resolvidos e os registros permanecem para consulta. Uma reprodução exata requer as versões originais correspondentes a esses hashes.

O log da avaliação DINO de teste registra uma falha de verificação após a gravação dos relatórios. As métricas e rotas arquivadas permitem conferir os resultados publicados.

## Jobs repetidos

Quando um job foi reenviado, cada tentativa tem log próprio, com o ID do job no nome do arquivo.
