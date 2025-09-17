from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, Dataset, Value
from huggingface_hub import list_datasets
import pandas as pd
import torch
import argparse
import os
import json
from accelerate import Accelerator
from utils import clean_index_columns
import csv
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
import logging
from logger_config import init_logger
from utils import parse_answer
import re
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


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
    "--use_accelerate",
    action="store_true",
    help="Use Accelerate for multi-GPU inference"
)

parser.add_argument(
    "--use_flash_attention",
    action="store_true",
    help="Enable Flash Attention 2 for faster inference"
)

parser.add_argument(
    "--run_local",
    action="store_true",
    help="If set, read/save results locally as CSV instead of using HuggingFace Hub"
)

args = parser.parse_args()

init_logger()

if args.use_accelerate:
    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f"Using Accelerate with device: {device}")
else:
    accelerator = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Usando single device: {device}")

def get_prompt_for_model(prompt_json_str, model_path):
    try:
        prompt_data = json.loads(prompt_json_str)
        for tokenizer_path, tokenizer_info in prompt_data.items():
            if model_path in tokenizer_info['models']:
                return tokenizer_info['prompt']

        raise ValueError(f"Model {model_path} not found in prompt data")
    except json.JSONDecodeError:
        raise ValueError(f"Prompt is not a valid JSON: {prompt_json_str}")

def generate_answer(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=4,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    # vamos pegar apenas os tokens da resposta (ou seja, descartamos os tokens do input)
    num_tokens_prompt = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[0][num_tokens_prompt:], skip_special_tokens=True)
    return generated_text


def map_answer(example, model_path):
    actual_prompt = get_prompt_for_model(example['prompt'], model_path)
    model_answer = generate_answer(actual_prompt)
    example['model_answer'] = model_answer
    example['parsed_model_answer'] = parse_answer(example)
    example['prompt'] = actual_prompt
    return example

def ensure_string_columns(ds):
    target_cols = [
        'model_answer', 'parsed_model_answer', 'label',
        'prompt', 'shot_indices', 'benchmark', 'model_name', 'id_bench'
    ]
    for col in target_cols:
        if col in ds.column_names:
            ds = ds.cast_column(col, Value('string'))
    return ds

# check if the dataset already exists in the hub
possible_datasets = list_datasets(search=args.answers_path)
dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)

for model_path in args.model_path:
    logging.info(f"** RUNNING MODEL: {model_path}")
    model_name = model_path.split("/")[-1]
    current_model_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.use_accelerate:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        model = accelerator.prepare(model)
    else:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        if args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

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
    dataset = dataset.map(
        lambda example: map_answer(example, model_path),
        desc=f"{model_path.split('/')[1]}"
    )

    all_datasets = []
    if args.run_local:
        # Loading local dataset to append results to (if a run has already been made, add new results to the end of the file)
        answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
        answers_filepath = os.path.join("eval_processing", answers_filename)
        dataset_exists = os.path.exists(answers_filepath)

        if dataset_exists:
            original_df = pd.read_csv(answers_filepath)
            original_df = clean_index_columns(original_df)
            original_df = original_df.reset_index(drop=True)
            original_dataset = Dataset.from_pandas(original_df)

            # Filter out duplicates for this model/benchmark combination
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

    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if args.run_local:
    answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
    answers_filepath = os.path.join("eval_processing", answers_filename)
    logging.info(f"**ALL MODELS LOCALLY SAVED AT: {answers_filepath}")
else:
    logging.info(f"**ALL MODELS SAVED AT: {args.answers_path}")
