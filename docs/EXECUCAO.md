# Ambiente e execução

Os experimentos usam Python, MMDetection, CUDA e Slurm. Os launchers em `slurm/` aceitam os parâmetros de recursos do ambiente de destino, incluindo a partição indicada com `sbatch --partition=<particao>`. Execute os comandos na raiz do projeto.

## Ambiente

O ambiente de referência usa Python 3.10 ou 3.11, PyTorch 2.1.2, torchvision 0.16.2, MMEngine 0.10.5, MMCV 2.1.0, MMDetection 3.3.0, OpenCLIP 2.24.0 e pycocotools 2.0.7. Instale uma versão do PyTorch compatível com o CUDA disponível e a wheel correspondente do MMCV.

```bash
python -m venv .venv-ligia
source .venv-ligia/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2
python -m pip install -U openmim
mim install mmcv==2.1.0
python -m pip install -e '.[cluster]'
```

O fine-tuning usa `EarlyStoppingHook` do MMEngine, com monitoramento de `coco/bbox_mAP_50`. Os launchers solicitam uma GPU e pressupõem uma instalação do Slurm com suporte a `--gres=gpu:1`.

## Dados e pesos

As imagens, anotações e checkpoints devem ser obtidos separadamente. A preparação dos splits requer os arquivos ZIP da DWD, as anotações COCO processadas por domínio e o manifesto das reservas congeladas. A estrutura e os hashes disponíveis estão em [DADOS_E_CHECKPOINTS.md](DADOS_E_CHECKPOINTS.md). O manifesto e as anotações usados na campanha de referência não estão incluídos, portanto reproduzir exatamente os splits requer obtê-los separadamente.

```bash
export DWD_ROOT="$PWD/data/raw/dwd/full-train-frozen-eval-20261005"
export DINO_SPLIT_ROOT="$DWD_ROOT"
export CLIP_RN101_CHECKPOINT="$PWD/checkpoints/RN101.pt"
export DINO_R50_CHECKPOINT="$PWD/checkpoints/resnet50-11ad3fa6.pth"
```

Quando os insumos de preparação estiverem disponíveis, `slurm/prepare_dino_full_splits.sbatch` usa `DWD_ROOT_PARENT`, `PROCESSED_ROOT`, `PROCESSED_V4_ROOT` e `FROZEN_MANIFEST` para localizar os dados. Os caminhos dos insumos precisam ficar sob `DWD_ROOT_PARENT`. A saída `DINO_SPLIT_ROOT` deve ser um diretório novo.

## Ordem dos jobs

Antes de submeter qualquer job, crie o diretório local para os logs e as saídas. O Slurm precisa que ele exista antes da execução.

```bash
mkdir -p outputs
```

Os exemplos abaixo usam os valores padrão de diretório. Acrescente a partição adequada a cada comando `sbatch`, quando necessário. O tempo e os demais recursos podem ser ajustados na submissão. Os IDs são retornados pelo Slurm.

1. Prepare os splits ou disponibilize os COCO JSONs e `split_manifest.json` em `DWD_ROOT`. Se os dados de entrada da preparação estiverem disponíveis, execute `sbatch slurm/prepare_dino_full_splits.sbatch`.
2. Obtenha o gate de referência ou execute `sbatch slurm/train_clip_domain_gate.sbatch`. A saída padrão fica em `outputs/domain-gate-clip-rn101-<job>/domain_gate_linear.pt`. Configure o caminho em `GATE_CHECKPOINT` e `DINO_GATE_CHECKPOINT`. Um gate treinado novamente pode produzir resultados diferentes.
3. Execute `sbatch slurm/train_clip_full_dayclear.sbatch`. Configure `BASELINE_JOB_ID` com o ID retornado e submeta `sbatch slurm/train_clip_full_domain_experts.sbatch`. Configure `EXPERTS_JOB_ID` com o ID do array.
4. Submeta `sbatch slurm/evaluate_clip_full_moe_gateval.sbatch`. Após revisar o relatório, configure `GATEVAL_JOB_ID` com seu ID e submeta `sbatch slurm/evaluate_clip_full_moe_test.sbatch`. Ambas as etapas aceitam `GATE_CHECKPOINT`.
5. Execute `sbatch slurm/train_dino_full_dayclear.sbatch`. Configure `BASELINE_CHECKPOINT` com o `iter_30000.pth` produzido e `CAMPAIGN_ID` com um identificador para a campanha. Execute `sbatch slurm/train_dino_expert_array.sbatch`.
6. Configure `DINO_GATE_CHECKPOINT` com o gate desejado e execute `sbatch slurm/evaluate_dino_moe.sbatch` com `DINO_MOE_SPLIT=gate_val`. Revise o relatório e submeta novamente com `DINO_MOE_SPLIT=test` para avaliar os modelos congelados.
7. Para cada um dos quatro domínios especialistas, configure `TRAIN_DOMAIN` e `EXPERT_ROOT="$PWD/outputs/dino-moe-experts-$CAMPAIGN_ID"`, depois execute `sbatch slurm/finetune_dino_r50_bs8.sbatch`. A etapa usa uma GPU A100 com pelo menos 70 GiB e grava o melhor checkpoint por domínio.

Os jobs de avaliação produzem `moe_evaluation.md`, `moe_evaluation.json`, predições e rotas no diretório da execução em `outputs/`. As tabelas publicadas em `experiments/*/resultados/` são cópias dos relatórios da campanha de referência. Consulte [RESULTADOS.md](RESULTADOS.md) para os valores e as limitações.

## Galerias

Os manifestos de amostras das galerias DINO ficam em `experiments/dino-r50-moe/galerias/por-dominio/samples.json` e `experiments/dino-r50-moe/galerias/por-categoria/samples.json`. Os dados de imagem e os checkpoints são externos.

Para executar a inferência qualitativa dos especialistas fine-tuned, crie um JSON que associe cada domínio ao caminho de seu checkpoint, defina `DINO_FINETUNED_CHECKPOINT_MAP` com esse arquivo e submeta `sbatch slurm/infer_dino_r50_finetuned_gallery.sbatch`. O job grava as predições em `outputs/dino-r50-finetuned-gallery-<job>/`.

```json
{
  "day_foggy": "/caminho/day_foggy.pth",
  "dusk_rainy": "/caminho/dusk_rainy.pth",
  "night_clear": "/caminho/night_clear.pth",
  "night_rainy": "/caminho/night_rainy.pth"
}
```

Com as anotações, as predições arquivadas do teste e as predições fine-tuned, renderize as galerias comparativas em um diretório novo:

```bash
python scripts/render_experiment_comparison_galleries.py \
  --annotations-root "$DWD_ROOT" \
  --sample-manifests experiments/dino-r50-moe/galerias/por-dominio/samples.json \
                     experiments/dino-r50-moe/galerias/por-categoria/samples.json \
  --dino-original-dir experiments/dino-r50-moe/arquivo-execucao/predicoes/test \
  --clip-test-dir experiments/clip-rn101-vltdet/arquivo-execucao/predicoes/test \
  --dino-finetuned-dir "$PWD/outputs/dino-r50-finetuned-gallery-<job>" \
  --output-root "$PWD/outputs/galerias-comparativas"
```

As duas galerias selecionam imagens com base no desempenho do DINO e servem para análise qualitativa. A seleção e a renderização não calculam métricas de teste do fine-tuning.

## Rastreabilidade

Os arquivos em `arquivo-execucao/` incluem configs resolvidos, logs, rotas e predições selecionadas. Os caminhos usam marcadores que precisam ser substituídos ao reutilizar esses artefatos. As versões de alguns arquivos usadas nas campanhas históricas não estão disponíveis com os hashes registrados. Consulte [DADOS_E_CHECKPOINTS.md](DADOS_E_CHECKPOINTS.md) antes de tentar reproduzir os resultados exatos.
