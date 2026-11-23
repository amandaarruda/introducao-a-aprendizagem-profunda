# Arquivo de execução: CLIP RN101 VLTDet

Artefatos das execuções finais do baseline (`day_clear`) e do MoE, para auditar os números publicados.

- `configs/`: configs resolvidos pelo MMDetection em cada execução. Caminhos do cluster aparecem como `<PROJECT_ROOT>`.
- `logs/`: saída do Slurm e logs de treino do baseline e dos quatro especialistas, além das avaliações de gate-val e teste.
- `predicoes/gateval/` e `predicoes/test/`: predições por imagem do baseline e do MoE, avaliação agregada e rota escolhida pelo gate em cada imagem.

Pesos e dados de entrada não estão incluídos. Os hashes ficam em [DADOS_E_CHECKPOINTS.md](../../../docs/DADOS_E_CHECKPOINTS.md).
