# Rodando Evals (v1.0.3)


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

1. Crie um .env com um token do huggingface (HUGGINGFACE_TOKEN)
2. Crie um arquivo YAML de configuração baseado no exemplo abaixo:

#### yaml/config.example.yaml
```yaml
# ID único para este run de evaluação
run_id: "meu_teste"

# Modelos a serem avaliados
model_paths:
  - path: "Qwen/Qwen2.5-0.5B-Instruct"
    custom: false
    tokenizer_path: "Qwen/Qwen2.5-0.5B-Instruct"
  - path: "meta-llama/Llama-3.2-3B-Instruct"
    custom: false
    tokenizer_path: "meta-llama/Llama-3.2-3B-Instruct"

# Configuração multi-GPU (opcional)
multi_gpu:
  enabled: false
  num_gpus: 1

# Parâmetros do evaluation
flash_attention: false # Se é para usar FA2 ou não
num_shots: 5           # Quantidade de exemplos no few-shot
num_experiments: 3     # Número de experimentos por sample
update_leaderboard: false  # Atualizar leaderboard ao final

# Benchmarks a executar
benchmark_names:
  - "assin2rte"
  - "assin2sts"
  - "bluex"
  - "enem"
  - "hatebr"
  - "portuguese_hate_speech"
  - "faquad"
  - "tweetsentbr"
  - "oab"
```


### Parâmetros

- **run_id**: Identificador único para o experimento
- **model_paths**: Lista de modelos do HuggingFace a avaliar
  - **path**: ID do modelo no HuggingFace
  - **custom**: Se é modelo é finetunado por nós (opcional)
  - **tokenizer_path**: Tokenizer a usar caso o mesmo do modelo der problema (opcional)
- **multi_gpu**: Configuração para usar múltiplas GPUs (opcional)
- **flash_attention**: Se é para usar FA2 ou não (opcional)
- **num_shots**: Quantidade de exemplos no contexto few-shot (opcional)
- **num_experiments**: Repetições por sample com diferentes few-shots
- **update_leaderboard**: Se deve atualizar o leaderboard automaticamente (opcional)
- **benchmark_names**: Lista de benchmarks disponíveis em `src/UTILS_BENCHMARKS.py` (opcional)


### Rodando por Docker

**Build**: `docker build -f .devcontainer/Dockerfile -t energygpt-eval ..`
**Run**: `docker run --rm --gpus=all --env-file .env energygpt-eval`


### Problemas a serem resolvidos

Os problemas da aplicação para serem resolvidos depois são:
- O Gemma3 tem problemas com o KV Cache dele. Uma solução temporaria é aumentar `torch._dynamo.config.cache_size_limit`, mas isso pode fazer o eval demorar uns 10x mais para terminar e degradar performance.


### Proximo Release

Para a próxima versão:
- Criar uma flag para ter a opção de rodar tudo localmente
- Melhorar performance
