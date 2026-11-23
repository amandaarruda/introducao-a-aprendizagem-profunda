# Arquivo de execução: fine-tuning DINO ResNet-50

Artefatos das quatro execuções escolhidas pela validação (uma por domínio).

- `configs/`: configs resolvidos de cada domínio.
- `logs/`: logs completos de treino e de Slurm, com as validações e a parada antecipada.
- `hashes/`: SHA-256 dos dados de entrada e dos quatro checkpoints selecionados. Os pesos não estão incluídos.

Não há avaliação de teste para esta etapa. Métricas e limitações estão no [README do experimento](../README.md).
