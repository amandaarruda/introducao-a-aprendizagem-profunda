# Arquivo de execução: DINO ResNet-50 MoE

Artefatos da campanha final: baseline `day_clear`, quatro especialistas e avaliação MoE.

- `configs/`: configs resolvidos do baseline e dos especialistas. Caminhos do cluster aparecem como `<PROJECT_ROOT>`.
- `logs/`: logs de treino e de Slurm do baseline, dos especialistas, do gate-val e do teste.
- `predicoes/gateval/` e `predicoes/test/`: predições por imagem do baseline e do MoE, avaliação agregada e rotas do gate.

Pesos e dados de entrada não estão incluídos. Os hashes dos pesos estão em [DADOS_E_CHECKPOINTS.md](../../../docs/DADOS_E_CHECKPOINTS.md).
