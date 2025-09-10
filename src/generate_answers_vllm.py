from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets, Dataset, Value
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
import csv
import torch
import time
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

def clean_data_for_csv(df):
    def clean_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        # Handle POSCOMP-specific Unicode escapes by doubling them
        text = re.sub(r'\\([uU][0-9a-fA-F]{4})', r'\\\\\1', text)
        # Handle other problematic backslashes
        text = re.sub(r'\\([^uU])', r'\\\\\1', text)
        return text
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_text)
    return df

def get_prompt_for_model(prompt_json_str, model_path):
    try:
        prompt_data = json.loads(prompt_json_str)
        for tokenizer_path, tokenizer_info in prompt_data.items():
            if model_path in tokenizer_info['models']:
                return tokenizer_info['prompt']

        raise ValueError(f"Model {model_path} not found in prompt data")
    except json.JSONDecodeError:
        raise ValueError(f"Prompt is not a valid JSON: {prompt_json_str}")

def generate_answers_batch(prompts, model_path, max_len):
	actual_prompts = [get_prompt_for_model(p, model_path) for p in prompts]
	if any(len(p) > max_len for p in actual_prompts):
		logging.warning(f"Batch contains prompts exceeding {max_len} tokens, middle-truncating prompts")

        # When the prompt is too long, we truncate it in the middle to avoid losing important information
		def mid_trunc(s, limit):
			ellipsis = " ... "
			extra = 5
			if len(s) <= limit:
				return s
			keep = max(limit - extra - len(ellipsis), 0)
			head = keep // 2
			tail = keep - head
			if keep == 0:
				return ellipsis.strip()
			return s[:head] + ellipsis + (s[-tail:] if tail > 0 else "")
		actual_prompts = [mid_trunc(p, max_len) if len(p) > max_len else p for p in actual_prompts]
	outputs = llm.generate(actual_prompts, sampling_params, use_tqdm=False)
	generated_texts = [o.outputs[0].text for o in outputs]
	return actual_prompts, generated_texts

def map_answer_batch(examples, model_path, max_len):
    prompts = examples['prompt']
    actual_prompts, model_answers = generate_answers_batch(prompts, model_path, max_len)

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

def ensure_string_columns(ds):
    target_cols = [
        'model_answer', 'parsed_model_answer', 'label',
        'prompt', 'shot_indices', 'benchmark', 'model_name', 'id_bench'
    ]
    for col in target_cols:
        if col in ds.column_names:
            ds = ds.cast_column(col, Value('string'))
    return ds

possible_datasets = list_datasets(search=args.answers_path)
dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)



for model_path in args.model_path:
    logging.info(f"** RUNNING MODEL: {model_path}")
    model_name = model_path.split("/")[-1]
    current_model_path = model_path

    llm_kwargs = {
        "model": model_path,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "seed": 42,
    }

    # Retry LLM initialization
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logging.info(f"Initializing vLLM attempt {attempt + 1}/3 with tensor_parallel_size={args.tensor_parallel_size}")
            llm = LLM(**llm_kwargs)
            max_len = llm.llm_engine.model_config.max_model_len
            break
        except Exception as e:
            if 'llm' in locals():
                del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == max_attempts - 1:
                raise Exception(f"Failed to initialize LLM after 3 attempts: {e}")
            logging.warning(f"LLM initialization attempt {attempt + 1} failed: {e}")
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)

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
        lambda examples: map_answer_batch(examples, model_path, max_len),
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
			filtered_dataset = ensure_string_columns(filtered_dataset)
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
			filtered_dataset = ensure_string_columns(filtered_dataset)
            all_datasets.append(filtered_dataset)

	dataset = ensure_string_columns(dataset)
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
        final_df = clean_data_for_csv(final_df)
        final_df.to_csv(
            answers_filepath,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
            escapechar='\\',
            lineterminator='\n'
        )
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
