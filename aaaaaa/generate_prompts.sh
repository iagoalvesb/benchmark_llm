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
OUTPUT_HUB_REPO="pt-eval"

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
  --output_hub_repo "${OUTPUT_HUB_REPO}"

echo "Prompt generation completed. Outputs saved to '${OUTPUT_HUB_REPO}'"
