# Rodando Evals (v1.0.0)


## Índice
- [Como Rodar](#como-rodar)
- [Configuração](#configuração)
- [Parâmetros](#parâmetros)
- [Rodando por Docker](#rodando-por-docker)
- [Modelos Testados](#modelos-testados)
- [Problemas a serem resolvidos](#problemas-a-serem-resolvidos)


### Como Rodar

Para executar o pipeline completo sem docker, use o comando:

```bash
src/run.sh yaml/config.yaml
```


### Configuração

Crie um arquivo YAML de configuração baseado no exemplo abaixo:

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
  - **custom**: Se é modelo customizado (sempre false para modelos públicos)
  - **tokenizer_path**: Tokenizer a usar (geralmente igual ao modelo)
- **multi_gpu**: Configuração para usar múltiplas GPUs
- **num_shots**: Quantidade de exemplos no contexto few-shot
- **num_experiments**: Repetições por sample com diferentes few-shots
- **update_leaderboard**: Se deve atualizar o leaderboard automaticamente
- **benchmark_names**: Lista de benchmarks disponíveis em `src/UTILS_BENCHMARKS.py`


### Rodando por Docker

**Build**: `docker build -f .devcontainer/Dockerfile -t energygpt-eval ..`  
**Run**: `docker run --rm --gpus=all --env-file .env energygpt-eval`


### Modelos Testados

Um número de modelos foram testados para assegurar que funcionam com a pipeline atual. Pode acontecer de certos modelos apresentarem problemas. Com cada fix mais modelos serão testados.

Os modelos testados atê então são:
- Qwen2.5
- Qwen3
- Llama 3.2
- Llama 3.1


### Problemas a Serem Resolvidos

Para a v1.0.1
- Arrumar problema onde sem nenhum fewshot o código dá erro (fix estimado para dia 30 de junho)
- Param para habilitar ou desabilitar flash attention se for usar na B200
- Logging mais robusto e organizado
- Entender o porque do Gemma não funcionar e testar outros modelos abaixo de 14B