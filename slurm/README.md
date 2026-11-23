# Launchers Slurm

Os `.sbatch` organizam as etapas executadas em Slurm. Cada um registra parâmetros, verificações de entrada, hashes e variáveis necessárias. A ordem de execução está em [docs/EXECUCAO.md](../docs/EXECUCAO.md#ordem-dos-jobs).

| Grupo | Launchers |
|---|---|
| Dados | `prepare_dino_full_splits.sbatch` |
| CLIP | `train_clip_domain_gate.sbatch`, `train_clip_full_dayclear.sbatch`, `train_clip_full_domain_experts.sbatch`, `evaluate_clip_full_moe_gateval.sbatch`, `evaluate_clip_full_moe_test.sbatch` |
| DINO | `train_dino_full_dayclear.sbatch`, `train_dino_expert_array.sbatch`, `evaluate_dino_moe.sbatch` |
| Fine-tuning e galerias | `finetune_dino_r50_bs8.sbatch`, `infer_dino_r50_finetuned_gallery.sbatch` |

Informe a partição e os recursos adequados ao ambiente no momento da submissão. Os caminhos e variáveis de cada etapa estão descritos em [docs/EXECUCAO.md](../docs/EXECUCAO.md).

`finetune_dino_r50_bs8.sbatch` aceita um domínio por alocação (`TRAIN_DOMAIN`). A variável `EXPERT_ROOT` indica o diretório com os checkpoints dos especialistas.
