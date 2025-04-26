#!/usr/bin/env bash

# ---------------------------------------------
# Script to generate prompt files for benchmarks
# ---------------------------------------------



# ---------------------------------------------
# Description: Runs `get_bench_prompts.py` with specified number of shots, experiments, tokenizer, and benchmarks.
# ---------------------------------------------


set -e  # Exit immediately if a command exits with a non-zero status.

# -------------------------
# Configuration Parameters
# -------------------------

N_SHOTS=5
N_EXPERIMENTS=3
TOKENIZER_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_ID="qwen2.5"

# -------------------------
# Path Definitions
# -------------------------

PROMPTS_PATH="pt-eval/prompts_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
ANSWERS_PATH="pt-eval/answers_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"
EVALUATION_PATH="pt-eval/eval_${MODEL_ID}_${NUM_SHOTS}shot_${NUM_EXPERIMENTS}exp"

MODEL_PATHS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
  # "Qwen/Qwen2.5-3B-Instruct"
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



# -------------------------
# Generating Prompts
# -------------------------

echo "Generating prompts for benchmarks: ${BENCHMARK_NAMES[*]}"
python get_bench_prompts.py \
  --n_shots "${N_SHOTS}" \
  --n_experiments "${N_EXPERIMENTS}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --benchmark_names "${BENCHMARK_NAMES[@]}" \
  --save_path_prompts "${PROMPTS_PATH}"

echo "Prompt generation completed. Outputs saved to '${PROMPTS_PATH}'"
