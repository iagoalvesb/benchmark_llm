#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status.

# -------------------------
# Configuration Parameters
# -------------------------

NUM_SHOTS=5
NUM_EXPERIMENTS=3
TOKENIZER_PATH="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID="qwen2.5"

# -------------------------
# Path Definitions
# -------------------------

PROMPTS_PATH="pt-eval/prompts_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}expa"
ANSWERS_PATH="pt-eval/answers_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}expa"
EVALUATION_PATH="pt-eval/eval_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}expa"

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
python get_bench_prompts.py \
  --n_shots "${NUM_EXPERIMENTS}" \
  --n_experiments "${NUM_EXPERIMENTS}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --benchmark_names "${BENCHMARK_NAMES[@]}" \
  --save_path_prompts "${PROMPTS_PATH}"

echo "Prompt generation completed. Outputs saved to '${PROMPTS_PATH}'"


# -------------------------
# Generating Answers
# -------------------------


echo "Running answer generation..."
python generate_answers.py \
  --prompts_path "${PROMPTS_PATH}" \
  --output_path "${ANSWERS_PATH}" \
  --model_path "${MODEL_PATHS[@]}"

echo "Answer generation completed. Outputs saved to '${ANSWERS_PATH}'"


# -------------------------
# Evaluating Results
# -------------------------

echo "Evaluating generated answers..."
python evaluate.py \
  --answers_path "${ANSWERS_PATH}" \
  --output_evaluation_path "${EVALUATION_PATH}"

echo "Evaluation completed. Results saved to ${EVALUATION_PATH}"
