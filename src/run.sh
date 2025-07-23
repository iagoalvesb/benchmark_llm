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
echo "  MULTI_GPU_ENABLED: $MULTI_GPU_ENABLED"
echo "  MULTI_GPU_NUM_GPUS: $MULTI_GPU_NUM_GPUS"
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

    if [ "$MULTI_GPU_ENABLED" = "true" ]; then
        echo "Using multi-GPU with $MULTI_GPU_NUM_GPUS GPUs (tensor parallel)"
        ANSWER_ARGS+=("--tensor_parallel_size" "$MULTI_GPU_NUM_GPUS")
    fi

    python "${SCRIPT_DIR}/generate_answers_vllm.py" "${ANSWER_ARGS[@]}"

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