# Mapas congelados: DINO ResNet-50 MoE

- `route_map.json`: rota por domínio previsto pelo gate. `day_clear` vai para o baseline; os outros quatro domínios vão para o especialista do domínio.
- `models.json`: lista dos cinco detectores congelados, com config e checkpoint. Os caminhos usam `<experiment-root>` no lugar do diretório local.

Os checkpoints não estão incluídos. Os hashes estão em [DADOS_E_CHECKPOINTS.md](../../../docs/DADOS_E_CHECKPOINTS.md).
