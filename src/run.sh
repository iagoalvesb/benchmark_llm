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

NUM_SHOTS=4
NUM_EXPERIMENTS=1
TOKENIZER_PATH="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID="qwen2.5"

# -------------------------
# Path Definitions
# -------------------------

PROMPTS_PATH="pt-eval/prompts_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
ANSWERS_PATH="pt-eval/answers_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
EVALUATION_PATH="pt-eval/eval_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"

MODEL_PATHS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
#   "Qwen/Qwen2.5-1.5B-Instruct"
#   "Qwen/Qwen2.5-3B-Instruct"
)


# -------------------------
# Benchmarks to run
# -------------------------

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

# ---------------------------------------------------------------------------------------------------------------
# ------------------------- RUNNING SCRIPTS TO GENERATE PROMPTS, ANSWERS AND EVALUATION -------------------------
# ---------------------------------------------------------------------------------------------------------------

# -------------------------
# Generating Prompts
# -------------------------

echo "Generating prompts for benchmarks: ${BENCHMARK_NAMES[*]}"
python "${SCRIPT_DIR}/generate_prompts.py" \
  --n_shots "${NUM_EXPERIMENTS}" \
  --n_experiments "${NUM_EXPERIMENTS}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --benchmark_names "${BENCHMARK_NAMES[@]}" \
  --prompts_path "${PROMPTS_PATH}"

echo "Prompt generation completed. Outputs saved to '${PROMPTS_PATH}'"


# -------------------------
# Generating Answers
# -------------------------


echo "Running answer generation..."
python "${SCRIPT_DIR}/generate_answers.py" \
  --prompts_path "${PROMPTS_PATH}" \
  --answers_path "${ANSWERS_PATH}" \
  --model_path "${MODEL_PATHS[@]}"

echo "Answer generation completed. Outputs saved to '${ANSWERS_PATH}'"


# -------------------------
# Evaluating Results
# -------------------------

echo "Evaluating generated answers..."
python "${SCRIPT_DIR}/evaluate.py" \
  --answers_path "${ANSWERS_PATH}" \
  --eval_path "${EVALUATION_PATH}"

echo "Evaluation completed. Results saved to ${EVALUATION_PATH}"
