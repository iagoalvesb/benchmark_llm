# Rodando Evals (v1.1.0)


## Índice
- [Como Rodar](#como-rodar)
- [Configuração](#configuração)
- [Parâmetros](#parâmetros)
- [Rodando por Docker](#rodando-por-docker)
- [Problemas a serem resolvidos](#problemas-a-serem-resolvidos)
- [Proximo Release](#proximo-release)


### Como Rodar

Para executar o pipeline completo sem docker, use o comando:

```bash
src/run.sh yaml/config.yaml
```


### Configuração

1. Crie um arquivo YAML de configuração baseado no exemplo abaixo:

#### yaml/config.example.yaml
```yaml
# ID único para este run de evaluação
run_id: "meu_teste"

# Qual backend usar
backend: "vllm"

# Modelos a serem avaliados
model_paths:
  - path: "meta-llama/Llama-3.2-3B-Instruct"
    custom: false
    tokenizer_path: "meta-llama/Llama-3.2-3B-Instruct"

# Configuração multi-GPU (opcional)
multi_gpu:
  enabled: false
  num_gpus: 1

# Parâmetros do evaluation
run_local: true        # Se é para salvar os resultados localmente
num_shots: 5           # Quantidade de exemplos no few-shot
num_experiments: 1     # Número de variações no fewshot
update_leaderboard: false  # Atualizar leaderboard ao final

# Benchmarks a executar
benchmark_names:
    - assin2rte
    - assin2sts
    - bluex
    - enem
    - hatebr
    - portuguese_hate_speech
    - toxsyn_pt
    - faquad
    - tweetsentbr
    - oab
    - poscomp
    - energy_regulacao
    - aime24
    - aime25
    - mmlu
    - mmlu_redux
    - mmlu_pro
    - supergpqa

use_outlines: false   # Se deveria usar a lib outlines para deixar o modelo gerar respostas mais fieis ao que seriam geradas em situações reais. (Apresenta alguns problemas)
max_new_tokens: 4096  # Quantidade de tokens que podem ser gerados (Relevante para o use_outlines, sem outlines forçamos apenas 3)
batch_size: 128
use_percentage_dataset: 100 # Porcentagem total dos dados para usar.
```


### Parâmetros

- **run_id**: Identificador único para o experimento *(obrigatório)*
- **backend**: Qual backend vai ser usado. Opções são "vllm" e "hf" *(obrigatório)*
- **model_paths**: Lista de modelos do HuggingFace a avaliar *(obrigatório)*
  - **path**: ID do modelo no HuggingFace *(obrigatório)*
  - **custom**: Se é modelo é finetunado por nós (opcional, default: `false`)
  - **tokenizer_path**: Tokenizer a usar caso o mesmo do modelo der problema (opcional, default: mesmo que `path`)
- **multi_gpu**: Configuração para usar múltiplas GPUs (opcional, default: `{"enabled": false, "num_gpus": 1}`)
- **run_local**: Se é para rodar localmente ou não (opcional, default: `false`)
- **flash_attention**: Se é para usar FA2 ou não. Esse param não funciona usando vLLM pois ele ativa no default (opcional, default: `false`)
- **num_shots**: Quantidade de exemplos no contexto few-shot (opcional, default: `5`)
- **num_experiments**: Repetições por sample com diferentes few-shots (opcional, default: `3`)
- **update_leaderboard**: Se deve atualizar o leaderboard automaticamente (opcional, default: `false`)
- **benchmark_names**: Lista de benchmarks disponíveis em `src/UTILS_BENCHMARKS.py` (opcional)


## Rodando por Docker
### 1. Escolha o dockerfile com base no Cuda [11.8 ou 12.9]
**Build**: `docker build -f .devcontainer/Dockerfile_cuda118 -t energygpt-eval .`

**Run**: 
```bash

# 1. Inserir HUGGINGFACE_TOKEN
# 2. Inserir arquivo config.yaml manter path interno: /src/config.yaml
# 3. Desejavel especificar cache dir inteiro para /cache
docker run --rm --gpus=all \                                                            
  -e HUGGINGFACE_TOKEN=hf_ \
  -v "$PWD/yaml/config.yaml":/workspace/src/config.yaml \
  -v "$PWD/.cache:/cache/hf" \
  energygpt-eval

```

Obs: No dockerfile, a imagem base do pytorch que está sendo usada usa o CUDA 12.1 por causa da 4090. Se tiver algum problema de build, utilize uma imagem do pytorch differente

## Rodando versão com API
Se quiser rodar com API, é necessário passar a chave ao rodar o docker:

```bash
docker run --rm --gpus=all \
  -e HUGGINGFACE_TOKEN=hf_ \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e GOOGLE_API_KEY=your-google-key-here \
  -v "$PWD/yaml/config.yaml":/workspace/src/config.yaml \
  -v "$PWD/.cache:/cache/hf" \
  energygpt-eval
```
