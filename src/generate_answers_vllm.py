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
from typing import Optional

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

parser.add_argument(
    "--use_outlines",
    action="store_true",
    help="Use outlines to enforce structured JSON outputs with explanation and final answer"
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=4,
    help="Maximum number of new tokens to generate"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for prompt generation requests (dataset.map batched size)"
)

parser.add_argument(
    "--data_parallel_size",
    type=int,
    default=1,
    help="Number of data-parallel shards (use >1 when launching multiple ranks)"
)

parser.add_argument(
    "--data_parallel_rank",
    type=int,
    default=0,
    help="Rank index of this data-parallel worker (0..data_parallel_size-1)"
)

parser.add_argument(
    "--answers_shard_suffix",
    type=str,
    default="",
    help="Optional suffix to append to the local answers CSV filename for sharded saving"
)

args = parser.parse_args()

init_logger()

sampling_params = SamplingParams(
    max_tokens=args.max_new_tokens,
    temperature=0.0,  # Keep deterministic default to preserve prior behavior
    n=1,
    seed=42,
    frequency_penalty=1.2,
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
        'prompt', 'shot_indices', 'benchmark', 'model_name', 'id_bench', 'explanation'
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

    # Optional data-parallel sharding
    if args.data_parallel_size and args.data_parallel_size > 1:
        logging.info(f"Sharding dataset: data_parallel_size={args.data_parallel_size}, rank={args.data_parallel_rank}")
        try:
            dataset = dataset.shard(num_shards=args.data_parallel_size, index=args.data_parallel_rank, contiguous=True)
        except Exception as e:
            logging.warning(f"Failed to shard dataset contiguously, falling back to non-contiguous shard: {e}")
            dataset = dataset.shard(num_shards=args.data_parallel_size, index=args.data_parallel_rank)

    if args.use_outlines:
        # Lazy import to avoid hard dependency when not used
        try:
            import outlines
            from pydantic import BaseModel
        except Exception as e:
            raise ImportError(f"Outlines or pydantic not available but --use_outlines was set: {e}")

        class StructuredOutput(BaseModel):
            explicacao: str
            resposta: str

        def build_outlines_instruction(benchmark_name: str) -> str:
            benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
            base = (
                "IMPORTANTE: Responda APENAS no formato JSON com as chaves 'explicacao' e 'resposta'. "
                "A chave 'explicacao' deve conter um raciocínio breve e objetivo. "
                "A chave 'resposta' deve conter SOMENTE a resposta final no formato especificado abaixo.\n"
            )
            if benchmark.answer_pattern == "yes_no":
                spec = "Para 'resposta', use exatamente 'Sim' ou 'Não'."
            elif benchmark.answer_pattern == "multiple_choice":
                spec = "Para 'resposta', use exatamente UMA letra entre 'A', 'B', 'C', 'D' ou 'E'."
            elif benchmark.answer_pattern == "multiple_choice_full_word":
                spec = "Para 'resposta', use exatamente uma palavra entre 'Positivo', 'Negativo' ou 'Neutro'."
            elif benchmark.answer_pattern == "continue_value":
                spec = "Para 'resposta', use apenas um número (use ponto decimal se necessário)."
            else:
                spec = "Para 'resposta', forneça apenas o valor final esperado para o benchmark."
            example = "Exemplo: {\"explicacao\": \"...\", \"resposta\": \"...\"}"
            return base + spec + "\n" + example

        outlines_model = outlines.from_vllm_offline(llm)

        def map_answer_outlines(example, model_path_local, max_len_local, outlines_model_local):
            actual_prompt = get_prompt_for_model(example['prompt'], model_path_local)
            # No longer append instruction here; generate_prompts already embeds it in the user message

            result_json = outlines_model_local(
                actual_prompt,
                output_type=StructuredOutput,
                sampling_params=sampling_params
            )

            try:
                validated = StructuredOutput.model_validate_json(result_json)
                explanation = validated.explicacao
                final_answer_text = validated.resposta
            except Exception:
                explanation = None
                final_answer_text = result_json

            example['prompt'] = actual_prompt
            example['model_answer'] = result_json
            example['explanation'] = explanation
            parsed = parse_answer({'benchmark': example['benchmark'], 'model_answer': final_answer_text})
            example['parsed_model_answer'] = parsed
            return example
        batch_size = args.batch_size
        dataset = dataset.map(
            lambda example: map_answer_outlines(example, model_path, max_len, outlines_model),
            # batched=True,
            # batch_size=batch_size,
            desc=f"{model_path.split('/')[1] if '/' in model_path else model_path} (outlines)"
        )
    else:
        batch_size = args.batch_size
        dataset = dataset.map(
            lambda examples: map_answer_batch(examples, model_path, max_len),
            batched=True,
            batch_size=batch_size,
            desc=f"{model_path.split('/')[1] if '/' in model_path else model_path}"
        )

    all_datasets = []
    if args.run_local:
        # In data-parallel local mode, skip reading an existing merged file per shard
        if not (args.data_parallel_size and args.data_parallel_size > 1):
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
        base_name = args.answers_path.replace("/", "_").replace("-", "_")
        suffix = args.answers_shard_suffix if args.answers_shard_suffix else (f"_shard{args.data_parallel_rank}" if (args.data_parallel_size and args.data_parallel_size > 1) else "")
        answers_filename = f"{base_name}{suffix}.csv"
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
        if args.data_parallel_size and args.data_parallel_size > 1:
            # Save shard locally instead of pushing to hub
            os.makedirs("eval_processing", exist_ok=True)
            base_name = args.answers_path.replace("/", "_").replace("-", "_")
            suffix = args.answers_shard_suffix if args.answers_shard_suffix else f"_shard{args.data_parallel_rank}"
            answers_filename = f"{base_name}{suffix}.csv"
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
            logging.info(f"Saved shard locally at {answers_filepath}; skipping hub push in data-parallel mode")
        else:
            full_dataset.push_to_hub(args.answers_path)
            logging.info(f"**SAVED MODEL {model_name} AT: {args.answers_path}")

    del llm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if args.run_local:
    base_name = args.answers_path.replace("/", "_").replace("-", "_")
    suffix = args.answers_shard_suffix if args.answers_shard_suffix else (f"_shard{args.data_parallel_rank}" if (args.data_parallel_size and args.data_parallel_size > 1) else "")
    answers_filename = f"{base_name}{suffix}.csv"
    answers_filepath = os.path.join("eval_processing", answers_filename)
    logging.info(f"**ALL MODELS LOCALLY SAVED AT: {answers_filepath}")
else:
    if args.data_parallel_size and args.data_parallel_size > 1:
        base_name = args.answers_path.replace("/", "_").replace("-", "_")
        logging.info(f"**DATA-PARALLEL SHARDS SAVED LOCALLY UNDER eval_processing/{base_name}_shard*.csv; upstream merge/push handled by launcher.")
    else:
        logging.info(f"**ALL MODELS SAVED AT: {args.answers_path}")
