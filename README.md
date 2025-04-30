**Dockerfile não foi validado ainda, não usar** -> Estamos ativamente trabalhando nisso

# Como usar:

## Rodando Evals
### Rodando Localmente

Apenas rodando o `run.sh` dentro do diretório `src`, o script inteiro é executado. É necessário apenas mudar algumas configurações/parâmetros dentro do .sh

<br>

#### 1. **Configuration Parameters**
- NUM_SHOTS: Quantidade de shots.
- NUM_EXPERIMENTS: Número de experimentos a serem realizados, onde experimento é a quantidade de vezes um único sample do prompt vai rodar no eval. Cada experimento possui few shots diferentes, então mesmo se a pergunta a qual o modelo precisa responder é o mesmo, ele sempre terá um contexto differente no prompt pois terá fewshots differentes.
- TOKENIZER_PATH: Caminho do tokenizer dos modelos. (Como só passamos 1 tokenizer, os modelos que serão passados depois precisam ser da mesma familia)
- MODEL_ID: ID gerado do modelo no nosso eval. (IDs não podem repetir entre runs)

Exemplo:
```
NUM_SHOTS=5
NUM_EXPERIMENTS=3
TOKENIZER_PATH="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID="qwen2.5"
```

<br>

#### 2. **Path Definitions**
As paths `PROMPTS_PATH`, `ANSWERS_PATH` e `EVALUATION_PATH` não precisam ser alteradas pois o script já gera elas automaticamente. O path que é necessário mudar é a seguinte:


- MODEL_PATHS: O ID do modelo que for usado no HuggingFace. (Obs: Como passamos só um tokenizer, todos os modelos precisam usar o tokenizer definido anteriormente)

Exemplo:
```
MODEL_PATHS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)
```

<br>

#### 3. **Benchmarks to run**
Passe os benchmarks a serem rodados. Por default, o `run.sh` já roda todos. Se quiser rodar todos, não precisa fazer alterações

Exemplo:
```
BENCHMARK_NAMES=(
  "assin2rte"
  "assin2sts"
  "bluex"
  "enem"
  "hatebr"
  "portuguese_hate_speech"
  "faquad"
  "tweetsentbr"
  "oab"
)
```

<br>

### Rodando por Dockerfile

Por enquanto, o docker só roda o .sh isoladamente. Você ainda vai ter que seguir os passos acima para tudo funcionar. 

**Build (Rode na root do projeto)**: docker build -f .devcontainer/Dockerfile -t energygpt-eval ..
**Run**: docker run --rm --gpus=all --env-file .env energygpt-eval

## Subindo no Leaderboard
#### Se você está lendo isso, estou esperando o Lucas mandar o novo esquema. O código depende bastante do esquema que for dado para o output.

O script que faz isso é a `generate_leaderboard_info.py`.
O comando para rodar é a seguinte, onde:
- benchamrks-file: Repo onde vamos pegar os dados. No momento, seria os datasets "eval" que estão sendo gerados no final do nosso código de eval.
- output-repo: Repo do HuggingFace onde será salvo esses resultados.

Exemplo:
```
python generate_leaderboard_info.py --benchmarks-file pt-eval/eval_qwen2.5_5shot_3expa --output-repo pt-eval/leaderboard
```