#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# -------------------------
# Configuration Parameters
# -------------------------

NUM_SHOTS=5
NUM_EXPERIMENTS=3
MODEL_ID="Qwe"

# -------------------------
# Path Definitions
# -------------------------

PROMPTS_PATH="pt-eval/prompts_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
ANSWERS_PATH="pt-eval/answers_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
EVALUATION_PATH="pt-eval/eval_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"

MODEL_PATHS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

# -------------------------
# Generating Answers
# -------------------------

echo "Running answer generation..."
python generate_answers.py \
  --prompts_path "${PROMPTS_PATH}" \
  --output_path "${ANSWERS_PATH}" \
  --model_path "${MODEL_PATHS[@]}"

# -------------------------
# Evaluating Results
# -------------------------

echo "Evaluating generated answers..."
python evaluate.py \
  --answers_path "${ANSWERS_PATH}" \
  --output_evaluation_path "${EVALUATION_PATH}"

echo "Evaluation completed. Results saved to ${EVALUATION_PATH}"
