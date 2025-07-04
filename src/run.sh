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
python "${SCRIPT_DIR}/generate_prompts.py" \
  --n_shots "${NUM_SHOTS}" \
  --n_experiments "${NUM_EXPERIMENTS}" \
  --model_paths "${MODEL_PATHS[@]}" \
  --model_tokenizers "${MODEL_TOKENIZERS[@]}" \
  --benchmark_names "${BENCHMARK_NAMES[@]}" \
  --prompts_path "${PROMPTS_PATH}"

echo "Prompt generation completed. Outputs saved to '${PROMPTS_PATH}'"


# -------------------------
# Generating Answers
# -------------------------


echo "Running answer generation..."
if [ "$MULTI_GPU_ENABLED" = "true" ]; then
    echo "Using multi-GPU with $MULTI_GPU_NUM_GPUS GPUs"
    FLASH_ATTENTION_FLAG=""
    if [ "$FLASH_ATTENTION_ENABLED" = "true" ]; then
        FLASH_ATTENTION_FLAG="--use_flash_attention"
    fi
    accelerate launch --num_processes="$MULTI_GPU_NUM_GPUS" "${SCRIPT_DIR}/generate_answers.py" \
      --prompts_path "${PROMPTS_PATH}" \
      --answers_path "${ANSWERS_PATH}" \
      --model_path "${MODEL_PATHS[@]}" \
      --use_accelerate \
      $FLASH_ATTENTION_FLAG
else
    echo "Using single GPU/CPU"
    FLASH_ATTENTION_FLAG=""
    if [ "$FLASH_ATTENTION_ENABLED" = "true" ]; then
        FLASH_ATTENTION_FLAG="--use_flash_attention"
    fi
    python "${SCRIPT_DIR}/generate_answers.py" \
      --prompts_path "${PROMPTS_PATH}" \
      --answers_path "${ANSWERS_PATH}" \
      --model_path "${MODEL_PATHS[@]}" \
      $FLASH_ATTENTION_FLAG
fi

echo "Answer generation completed. Outputs saved to '${ANSWERS_PATH}'"

# -------------------------
# Evaluating Results
# -------------------------

echo "Evaluating generated answers..."
python "${SCRIPT_DIR}/evaluate.py" \
  --answers_path "${ANSWERS_PATH}" \
  --eval_path "${EVALUATION_PATH}"

echo "Evaluation completed. Results saved to ${EVALUATION_PATH}"


# -------------------------
# Update Leaderboard (Optional)
# -------------------------

if [ "$UPDATE_LEADERBOARD" = "true" ]; then
    echo "Updating leaderboard..."
    python "${SCRIPT_DIR}/generate_leaderboard_info.py" \
      --benchmarks-file "${EVALUATION_PATH}" \
      --output-repo "pt-eval/leaderboard_testing" \
      --model-paths "${MODEL_PATHS[@]}" \
      --custom-flags "${MODEL_CUSTOM_FLAGS[@]}"
    echo "Leaderboard updated successfully!"
else
    echo "Skipping leaderboard update (UPDATE_LEADERBOARD=false)"
fi

echo "Pipeline completed successfully!"