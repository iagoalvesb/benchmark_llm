#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status.
echo "Starting..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# -------------------------
# Docker Settings
# -------------------------

if [ -f /.dockerenv ]; then
    python -c "import os; print('Running inside Docker – saving Hugging Face token...'); from huggingface_hub import HfFolder; HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))"
fi

# -------------------------
# Configuration Parameters
# -------------------------

CONFIG_FILE="${1:-}"

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Configuration file is required as first argument" >&2
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found" >&2
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

eval "$(python "${SCRIPT_DIR}/parse_input.py" "$CONFIG_FILE")"

echo "Configuration loaded successfully!"
echo "  NUM_SHOTS: $NUM_SHOTS"
echo "  NUM_EXPERIMENTS: $NUM_EXPERIMENTS"
echo "  RUN_ID: $RUN_ID"
echo "  USE_OUTLINES: $USE_OUTLINES"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  MULTI_GPU_ENABLED: $MULTI_GPU_ENABLED"
echo "  MULTI_GPU_NUM_GPUS: $MULTI_GPU_NUM_GPUS"
echo "  MULTI_GPU_MODE: $MULTI_GPU_MODE"
echo "  FLASH_ATTENTION_ENABLED: $FLASH_ATTENTION_ENABLED"
echo "  UPDATE_LEADERBOARD: $UPDATE_LEADERBOARD"
echo "  RUN_LOCAL: $RUN_LOCAL"
echo "  MODEL_PATHS: ${MODEL_PATHS[@]}"
echo "  MODEL_CUSTOM_FLAGS: ${MODEL_CUSTOM_FLAGS[@]}"
echo "  MODEL_TOKENIZERS: ${MODEL_TOKENIZERS[@]}"

# -------------------------
# Path Definitions
# -------------------------

PROMPTS_PATH="pt-eval/prompts_${RUN_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
ANSWERS_PATH="pt-eval/answers_${RUN_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
EVALUATION_PATH="pt-eval/eval_${RUN_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"

# ---------------------------------------------------------------------------------------------------------------
# ------------------------- RUNNING SCRIPTS TO GENERATE PROMPTS, ANSWERS AND EVALUATION -------------------------
# ---------------------------------------------------------------------------------------------------------------

# -------------------------
# Generating Prompts
# -------------------------

echo "Generating prompts for benchmarks: ${BENCHMARK_NAMES[*]}"

PROMPT_ARGS=(
  "--n_shots" "${NUM_SHOTS}"
  "--n_experiments" "${NUM_EXPERIMENTS}"
  "--model_paths" "${MODEL_PATHS[@]}"
  "--model_tokenizers" "${MODEL_TOKENIZERS[@]}"
  "--benchmark_names" "${BENCHMARK_NAMES[@]}"
  "--prompts_path" "${PROMPTS_PATH}"
)

if [ "$RUN_LOCAL" = "true" ]; then
    PROMPT_ARGS+=("--run_local")
fi

if [ "$USE_OUTLINES" = "true" ]; then
    PROMPT_ARGS+=("--use_outlines")
fi

python "${SCRIPT_DIR}/generate_prompts.py" "${PROMPT_ARGS[@]}"

echo "Prompt generation completed. Outputs saved to '${PROMPTS_PATH}'"


# -------------------------
# Generating Answers
# -------------------------


echo "Running answer generation..."

ANSWER_ARGS=(
  "--prompts_path" "${PROMPTS_PATH}"
  "--answers_path" "${ANSWERS_PATH}"
  "--model_path" "${MODEL_PATHS[@]}"
)

if [ "$RUN_LOCAL" = "true" ]; then
    ANSWER_ARGS+=("--run_local")
fi

if [ "$BACKEND" = "vllm" ]; then
    echo "Using vLLM backend"

    if [ "$MULTI_GPU_ENABLED" = "true" ] && [ "$MULTI_GPU_MODE" = "data" ]; then
        echo "Using multi-GPU with $MULTI_GPU_NUM_GPUS GPUs (data parallel)"
        # Determine GPU IDs from CUDA_VISIBLE_DEVICES if set, otherwise 0..N-1
        ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
        IFS=',' read -r -a GPU_LIST <<< "${ORIGINAL_CUDA_VISIBLE_DEVICES}"
        if [ -z "${ORIGINAL_CUDA_VISIBLE_DEVICES}" ]; then
            GPU_LIST=()
            for (( i=0; i<${MULTI_GPU_NUM_GPUS}; i++ )); do GPU_LIST+=("${i}"); done
        fi
        if [ ${#GPU_LIST[@]} -lt ${MULTI_GPU_NUM_GPUS} ]; then
            echo "Error: CUDA_VISIBLE_DEVICES has ${#GPU_LIST[@]} entries, but MULTI_GPU_NUM_GPUS=${MULTI_GPU_NUM_GPUS}" >&2
            exit 1
        fi

        BASE_NAME="${ANSWERS_PATH//\//_}"
        BASE_NAME="${BASE_NAME//-/_}"
        SHARD_DIR="eval_processing"

        # Iterate models sequentially to avoid overlapping model loads
        for MODEL in "${MODEL_PATHS[@]}"; do
            MODEL_NAME="${MODEL##*/}"
            # Common args for this model
            COMMON_ARGS=(
              "--prompts_path" "${PROMPTS_PATH}"
              "--answers_path" "${ANSWERS_PATH}"
              "--model_path" "${MODEL}"
              "--data_parallel_size" "${MULTI_GPU_NUM_GPUS}"
              "--max_new_tokens" "${MAX_NEW_TOKENS}"
              "--batch_size" "${BATCH_SIZE}"
            )
            if [ "$RUN_LOCAL" = "true" ]; then
                COMMON_ARGS+=("--run_local")
            fi
            if [ "$USE_OUTLINES" = "true" ]; then
                COMMON_ARGS+=("--use_outlines")
            fi

            SAFE_MODEL_SUFFIX="$(echo "_${MODEL_NAME}" | sed 's/[^A-Za-z0-9_]/_/g')"

            # Launch one rank per GPU, track PIDs
            PIDS=()
            for (( RANK=0; RANK<${MULTI_GPU_NUM_GPUS}; RANK++ )); do
                GPU_ID="${GPU_LIST[$RANK]}"
                SUFFIX="${SAFE_MODEL_SUFFIX}_shard${RANK}"
                echo "Launching data-parallel rank ${RANK} for ${MODEL_NAME} on GPU ${GPU_ID}"
                CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_DIR}/generate_answers_vllm.py" \
                  "${COMMON_ARGS[@]}" \
                  "--data_parallel_rank" "${RANK}" \
                  "--answers_shard_suffix" "${SUFFIX}" &
                PIDS+=("$!")
            done

            # Wait for all ranks and capture status
            FAIL=0
            for PID in "${PIDS[@]}"; do
                if ! wait "$PID"; then
                    FAIL=1
                fi
            done
            if [ "$FAIL" -ne 0 ]; then
                echo "Error: one or more ranks failed for model ${MODEL_NAME}" >&2
                exit 1
            fi

            # Merge shards for this model
            if [ "$RUN_LOCAL" = "true" ]; then
                FINAL_FILE="${SHARD_DIR}/${BASE_NAME}.csv"
                export BASE_NAME SHARD_DIR MODEL_NAME SAFE_MODEL_SUFFIX NUM_SHARDS="${MULTI_GPU_NUM_GPUS}" FINAL_FILE
                python - << 'PY'
import os, sys
import pandas as pd

base_name = os.environ['BASE_NAME']
shard_dir = os.environ['SHARD_DIR']
model_name = os.environ['MODEL_NAME']
suffix_prefix = os.environ['SAFE_MODEL_SUFFIX']
num_shards = int(os.environ['NUM_SHARDS'])
final_file = os.environ['FINAL_FILE']

dfs = []
for r in range(num_shards):
    shard_path = os.path.join(shard_dir, f"{base_name}{suffix_prefix}_shard{r}.csv")
    if os.path.exists(shard_path):
        dfs.append(pd.read_csv(shard_path))
    else:
        print(f"Warning: missing shard {shard_path}")

if not dfs:
    print("Error: no shards to merge", file=sys.stderr)
    sys.exit(1)

merged = pd.concat(dfs, ignore_index=True)

# Append into final file, replacing previous rows for this model
if os.path.exists(final_file):
    prev = pd.read_csv(final_file)
    prev = prev[prev.get('model_name', '') != model_name]
    merged = pd.concat([prev, merged], ignore_index=True)

merged.to_csv(final_file, index=False)
print(f"Merged shards for {model_name} into {final_file}")
PY
            else
                export ANSWERS_PATH BASE_NAME SHARD_DIR MODEL_NAME SAFE_MODEL_SUFFIX NUM_SHARDS="${MULTI_GPU_NUM_GPUS}"
                python - << 'PY'
import os
import pandas as pd
from datasets import Dataset, load_dataset

answers_path = os.environ['ANSWERS_PATH']
base_name = os.environ['BASE_NAME']
shard_dir = os.environ['SHARD_DIR']
model_name = os.environ['MODEL_NAME']
suffix_prefix = os.environ['SAFE_MODEL_SUFFIX']
num_shards = int(os.environ['NUM_SHARDS'])

dfs = []
for r in range(num_shards):
    shard_path = os.path.join(shard_dir, f"{base_name}{suffix_prefix}_shard{r}.csv")
    if os.path.exists(shard_path):
        dfs.append(pd.read_csv(shard_path))

if not dfs:
    raise RuntimeError("No shards found to push")

df = pd.concat(dfs, ignore_index=True)

# Merge with existing hub dataset, removing previous rows for this model
try:
    existing = load_dataset(answers_path, split='train')
    ex_df = existing.to_pandas()
    ex_df = ex_df[ex_df.get('model_name', '') != model_name]
    df = pd.concat([ex_df, df], ignore_index=True)
except Exception:
    pass

dataset = Dataset.from_pandas(df)
dataset.push_to_hub(answers_path)
print(f"Pushed merged dataset including {model_name} to {answers_path}")
PY
            fi

            # Optionally clean up shard files for this model
            for (( RANK=0; RANK<${MULTI_GPU_NUM_GPUS}; RANK++ )); do
                rm -f "${SHARD_DIR}/${BASE_NAME}${SAFE_MODEL_SUFFIX}_shard${RANK}.csv" || true
            done
        done

    else
        # Single process or tensor-parallel mode
        if [ "$MULTI_GPU_ENABLED" = "true" ] && [ "$MULTI_GPU_MODE" = "tensor" ]; then
            echo "Using multi-GPU with $MULTI_GPU_NUM_GPUS GPUs (tensor parallel)"
            ANSWER_ARGS+=("--tensor_parallel_size" "$MULTI_GPU_NUM_GPUS")
        fi

        # Append outlines opts if requested
        if [ "$USE_OUTLINES" = "true" ]; then
            ANSWER_ARGS+=("--use_outlines")
        fi
        ANSWER_ARGS+=("--max_new_tokens" "${MAX_NEW_TOKENS}")
        ANSWER_ARGS+=("--batch_size" "${BATCH_SIZE}")

        python "${SCRIPT_DIR}/generate_answers_vllm.py" "${ANSWER_ARGS[@]}"
    fi

elif [ "$BACKEND" = "hf" ]; then
    echo "Using HuggingFace backend"

    if [ "$FLASH_ATTENTION_ENABLED" = "true" ]; then
        ANSWER_ARGS+=("--use_flash_attention")
    fi

    if [ "$MULTI_GPU_ENABLED" = "true" ]; then
        echo "Using multi-GPU with $MULTI_GPU_NUM_GPUS GPUs (accelerate)"
        ANSWER_ARGS+=("--use_accelerate")
        accelerate launch --num_processes="$MULTI_GPU_NUM_GPUS" "${SCRIPT_DIR}/generate_answers.py" "${ANSWER_ARGS[@]}"
    else
        echo "Using single GPU/CPU"
        python "${SCRIPT_DIR}/generate_answers.py" "${ANSWER_ARGS[@]}"
    fi

else
    echo "Error: Unknown backend '$BACKEND'. Must be 'hf' or 'vllm'"
    exit 1
fi

echo "Answer generation completed. Outputs saved to '${ANSWERS_PATH}'"

# -------------------------
# Evaluating Results
# -------------------------

echo "Evaluating generated answers..."

EVAL_ARGS=(
  "--answers_path" "${ANSWERS_PATH}"
  "--eval_path" "${EVALUATION_PATH}"
)

if [ "$RUN_LOCAL" = "true" ]; then
    EVAL_ARGS+=("--run_local")
fi

python "${SCRIPT_DIR}/evaluate.py" "${EVAL_ARGS[@]}"

echo "Evaluation completed. Results saved to ${EVALUATION_PATH}"


# -------------------------
# Update Leaderboard (Optional)
# -------------------------

if [ "$UPDATE_LEADERBOARD" = "true" ]; then
    echo "Updating leaderboard..."

    LEADERBOARD_ARGS=(
      "--benchmarks-file" "${EVALUATION_PATH}"
      "--output-repo" "pt-eval/leaderboard_testing"
      "--model-paths" "${MODEL_PATHS[@]}"
      "--custom-flags" "${MODEL_CUSTOM_FLAGS[@]}"
      "--overwrite"
    )

    if [ "$RUN_LOCAL" = "true" ]; then
        LEADERBOARD_ARGS+=("--run_local")
    fi

    python "${SCRIPT_DIR}/generate_leaderboard_info.py" "${LEADERBOARD_ARGS[@]}"
    echo "Leaderboard updated successfully!"
else
    echo "Skipping leaderboard update (UPDATE_LEADERBOARD=false)"
fi

echo "Pipeline completed successfully!"