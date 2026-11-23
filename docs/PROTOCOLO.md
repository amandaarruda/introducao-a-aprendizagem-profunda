# Protocolo experimental

## Domínios e reservas

Os cinco domínios são `day_clear`, `day_foggy`, `dusk_rainy`, `night_clear` e `night_rainy`. Os splits de dados completos separam quatro conjuntos: `model_validation`, `gate_train`, `gate_val` e `test`. Cada reserva tem 100 imagens por domínio. O treino usa o restante do pool oficial elegível. A verificação dos splits encontrou zero sobreposição de IDs entre conjuntos em cada domínio.

Em DuskRainy e NightRainy, as reservas de validação e teste são definidas pelo projeto, porque os arquivos locais da DWD não trazem listas oficiais de teste para esses domínios.

## Seleção e teste

Modelos e rotas do MoE foram fixados antes da leitura do teste. O gate-val serve para validar o gate e para análise intermediária. O teste foi executado uma vez, com modelo e rota congelados, nos jobs registrados em `arquivo-execucao/`. AP50 e COCO AP são médias macro não ponderadas entre domínios.

No DINO, a matriz de especialistas compara o AP50 de cada detector em cada domínio. O resultado do sistema também depende das escolhas do gate para imagens individuais.

## Fine-tuning DINO

O fine-tuning parte dos quatro especialistas DINO (iteração 12 000). Preserva arquitetura, resolução, semente e pipeline. Usa batch 8, taxa de aprendizado 8e-5, no máximo 1000 atualizações, validação a cada 125 atualizações e parada antecipada após três validações sem melhora de pelo menos 0,001 em AP50. O checkpoint escolhido é o de melhor *model-validation* por domínio. O conjunto de teste não foi usado no treino nem na seleção dos checkpoints. Uma seleção de imagens do teste foi usada posteriormente em galerias qualitativas; não há métrica de teste para o fine-tuning.

## Galerias

As galerias comparativas usam os mesmos 50 IDs de teste nas duas famílias. A divisão *por domínio* tem 10 cenas por domínio. A divisão *por categoria* segue a regra da galeria original do DINO: 7 cenas por classe e 1 cena geral. Essa seleção depende do resultado, por isso as imagens são apenas qualitativas. Não mudam métricas, limiares nem escolhas de modelo.
