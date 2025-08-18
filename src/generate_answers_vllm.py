from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import list_datasets
import pandas as pd
import torch
import argparse
import os
import json
from utils import clean_index_columns
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
import logging
from logger_config import init_logger
from utils import parse_answer
import re
import torch
import random
import numpy as np

torch.backends.cudnn.deterministic = True
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_USE_V1"] = "1"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompts_path",
    type=str,
    required=True,
    help="Huggingface path to the prompt dataset [dataset must be in the format aquired by generate_prompts.sh]"
)

parser.add_argument(
    "--answers_path",
    type=str,
    required=True,
    help="Full Huggingface path to save models outputs"
)

parser.add_argument(
    "--model_path",
    type=str,
    nargs='+',
    required=True,
    help="Huggingface path to the models"
)

parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=1,
    help="Number of GPUs to use for tensor parallelism (replaces use_accelerate)"
)

parser.add_argument(
    "--run_local",
    action="store_true",
    help="If set, read/save results locally as CSV instead of using HuggingFace Hub"
)

args = parser.parse_args()

init_logger()

sampling_params = SamplingParams(
    max_tokens=4,
    temperature=0.0,  # Equivalent to do_sample=False
    n=1,
    seed=42
)

def get_prompt_for_model(prompt_json_str, model_path):
    try:
        prompt_data = json.loads(prompt_json_str)
        for tokenizer_path, tokenizer_info in prompt_data.items():
            if model_path in tokenizer_info['models']:
                return tokenizer_info['prompt']

        raise ValueError(f"Model {model_path} not found in prompt data")
    except json.JSONDecodeError:
        raise ValueError(f"Prompt is not a valid JSON: {prompt_json_str}")

def generate_answers_batch(prompts, model_path):
    actual_prompts = [get_prompt_for_model(prompt, model_path) for prompt in prompts]

    outputs = llm.generate(actual_prompts, sampling_params, use_tqdm=False)

    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)

    return actual_prompts, generated_texts

def map_answer_batch(examples, model_path):
    prompts = examples['prompt']
    actual_prompts, model_answers = generate_answers_batch(prompts, model_path)

    examples['prompt'] = actual_prompts
    examples['model_answer'] = model_answers

    parsed_answers = []
    for i in range(len(examples['benchmark'])):
        example = {
            'benchmark': examples['benchmark'][i],
            'model_answer': model_answers[i]
        }
        parsed_answers.append(parse_answer(example))

    examples['parsed_model_answer'] = parsed_answers
    return examples

possible_datasets = list_datasets(search=args.answers_path)
dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)

for model_path in args.model_path:
    logging.info(f"** RUNNING MODEL: {model_path}")
    model_name = model_path.split("/")[-1]
    current_model_path = model_path

    llm_kwargs = {
        "model": model_path,
        "dtype": "bfloat16",  # Equivalent to torch.bfloat16
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "seed": 42,
    }

    logging.info(f"Initializing vLLM with tensor_parallel_size={args.tensor_parallel_size}")

    llm = LLM(**llm_kwargs)

    if args.run_local:
        filename = args.prompts_path.replace("/", "_").replace("-", "_") + ".csv"
        filepath = os.path.join("eval_processing", filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompts file not found: {filepath}")

        df = pd.read_csv(filepath)
        df = clean_index_columns(df)
        dataset = Dataset.from_pandas(df)
        logging.info(f"Loaded prompts from local file: {filepath}")
    else:
        dataset = load_dataset(args.prompts_path, split='train')
        logging.info(f"Loaded prompts from HuggingFace Hub: {args.prompts_path}")

    dataset = dataset.map(lambda example: {"model_name": model_name})

    batch_size = 32
    dataset = dataset.map(
        lambda examples: map_answer_batch(examples, model_path),
        batched=True,
        batch_size=batch_size,
        desc=f"{model_path.split('/')[1] if '/' in model_path else model_path}"
    )

    all_datasets = []
    if args.run_local:
        answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
        answers_filepath = os.path.join("eval_processing", answers_filename)
        dataset_exists = os.path.exists(answers_filepath)

        if dataset_exists:
            original_df = pd.read_csv(answers_filepath)
            original_df = clean_index_columns(original_df)
            original_df = original_df.reset_index(drop=True)
            original_dataset = Dataset.from_pandas(original_df)

            current_benchmarks = set(dataset['benchmark'])
            filtered_df = original_df[
                ~((original_df['benchmark'].isin(current_benchmarks)) &
                (original_df['model_name'] == model_name))
            ]
            filtered_dataset = Dataset.from_pandas(filtered_df)
            all_datasets.append(filtered_dataset)
    else:
        # Same for huggingface hub
        possible_datasets = list_datasets(search=args.answers_path)
        dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)

        if dataset_exists:
            for attempt in range(3):
                try:
                    original_dataset = load_dataset(args.answers_path, split='train')
                    break
                except Exception as e:
                    if attempt == 2:
                        raise Exception(f"Failed to load existing dataset after 3 attempts: {e}")
                    logging.info(f"Tentativa {attempt + 1} falhou, retrying...")

            current_benchmarks = set(dataset['benchmark'])
            filtered_dataset = original_dataset.filter(
                lambda x: not (x['benchmark'] in current_benchmarks and x['model_name'] == model_name)
            )
            all_datasets.append(filtered_dataset)

    all_datasets.append(dataset)
    full_dataset = concatenate_datasets(all_datasets)

    full_dataset = full_dataset.map(
        lambda example, idx: {**example, "id": idx + 1},
        with_indices=True
    )

    if args.run_local:
        os.makedirs("eval_processing", exist_ok=True)
        answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
        answers_filepath = os.path.join("eval_processing", answers_filename)

        final_df = full_dataset.to_pandas()
        final_df.to_csv(answers_filepath, index=False)
        logging.info(f"**SAVED MODEL {model_name} AT: {answers_filepath}")
    else:
        full_dataset.push_to_hub(args.answers_path)
        logging.info(f"**SAVED MODEL {model_name} AT: {args.answers_path}")

    del llm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if args.run_local:
    answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
    answers_filepath = os.path.join("eval_processing", answers_filename)
    logging.info(f"**ALL MODELS LOCALLY SAVED AT: {answers_filepath}")
else:
    logging.info(f"**ALL MODELS SAVED AT: {args.answers_path}")