from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets, Value
import random
import pandas as pd
import torch
import json

import argparse

from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_shots",
    type=int,
    required=True,
    help="Number of shots used to build the prompt"
)

parser.add_argument(
    "--n_experiments",
    type=int,
    required=True,
    help="Number of prompts for each sample, changing the shots"
)

parser.add_argument(
    "--model_paths",
    type=str,
    nargs='+',
    required=True,
    help="Model paths from configuration"
)

parser.add_argument(
    "--model_tokenizers",
    type=str,
    nargs='+',
    required=True,
    help="Tokenizer paths for each model"
)

parser.add_argument(
    "--prompts_path",
    type=str,
    required=True,
    help="Huggingface repository to save (OBS: not the full path to save)"
)

parser.add_argument(
    "--benchmark_names",
    type=str,
    nargs='+',
    required=True,
    help="Names of benchmarks [must be in BENCHMARKS_PATTERNS.py]"
)

parser.add_argument(
    "--use_fixed_seed",
    type=bool,
    default=True,
)

args = parser.parse_args()

tokenizer_to_models = {}
for model_path, tokenizer_path in zip(args.model_paths, args.model_tokenizers):
    if tokenizer_path not in tokenizer_to_models:
        tokenizer_to_models[tokenizer_path] = []
    tokenizer_to_models[tokenizer_path].append(model_path)

tokenizers = {}
for tokenizer_path in tokenizer_to_models.keys():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizers[tokenizer_path] = tokenizer


def get_shots(id_query, dataset_fewshot, n_shots=args.n_shots, use_fixed_seed=args.use_fixed_seed, seed=42):
    possible_shots_indx = [i for i, example in enumerate(dataset_fewshot) if example['idx'] != id_query]
    
    if use_fixed_seed:
        random_state = random.Random(seed)
        shot_positions = random_state.sample(possible_shots_indx, n_shots)
    else:
        shot_positions = random.sample(possible_shots_indx, n_shots)

    # não vamos ter nossa query vazada no few_shot
    shots = dataset_fewshot.select(shot_positions)
    shot_id_benches = [shots[i]['id_bench'] for i in range(len(shots))]
    return shots, shot_id_benches


def get_prompt(example, benchmark, dataset_benchmark, n_shots=args.n_shots, n_experiments=args.n_experiments):
    for i in range(n_experiments):
        shots, shot_id_benches = get_shots(example['idx'], dataset, n_shots, args.use_fixed_seed)
        example_informations = benchmark.get_prompt_informations(example)

        chat = [{"role": "system", "content": example_informations['base_system_message']}]
        for shot in shots:
            shot_informations = benchmark.get_prompt_informations(shot)
            chat.append({"role": "user", "content": shot_informations['user_message']})
            chat.append({"role": "assistant", "content": shot_informations['assistant_message_with_answer']})

        chat.append({"role": "user", "content": example_informations['user_message']})
        chat.append({"role": "assistant", "content": example_informations['assistant_message_without_answer']})

        # Generate prompts for each unique tokenizer
        prompt_dict = {}
        for tokenizer_path, tokenizer in tokenizers.items():
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
            prompt_dict[tokenizer_path] = {
                "prompt": prompt,
                "models": tokenizer_to_models[tokenizer_path]
            }
        
        example[f"prompt_{i}"] = json.dumps(prompt_dict)
        example[f"shot_indices_{i}"] = shot_id_benches
    return example


all_benchmarks = []
for benchmark_name in args.benchmark_names:
    benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
    dataset = load_dataset(benchmark.dataset_path, benchmark.subset)
    if hasattr(benchmark, 'split'):
        dataset = dataset[benchmark.split]
    else:
        dataset = dataset['test'] if 'test' in dataset.keys() else dataset['train']

    # REMOVER QUALQUER LINHA QUE TENHA NAN
    if hasattr(benchmark, 'important_columns'):
        dataset = dataset.filter(lambda x: all((x[col] is not None) and (x[col] != "") for col in benchmark.important_columns if col in x))
    else:
        dataset = dataset.filter(lambda x: all((x[col] is not None) and (x[col] != "") for col in x))

    # dataset = dataset.select(list(range(15)))  # apenas para teste, depois tirar
    dataset = dataset.map(lambda example, idx: {"idx": int(idx)}, with_indices=True, desc="Adding index")

    # PADRONIZANDO TODOS OS BENCHMARKS
    if benchmark.label_column != "label":
        dataset = dataset.rename_column(benchmark.label_column, "label")
    dataset = dataset.add_column("benchmark", [benchmark_name] * len(dataset))
    dataset = dataset.add_column("id_bench", [f"{benchmark_name}_{i}" for i in range(len(dataset))])
    dataset = dataset.cast_column("label", Value("string"))

    # GERANDO PROMPTS
    get_prompt_partial = partial(get_prompt, benchmark=benchmark, dataset_benchmark=dataset)
    dataset = dataset.map(get_prompt_partial, num_proc=64, desc=benchmark_name)

    prompt_exp_columns = [f"prompt_{i}" for i in range(args.n_experiments)]
    shot_indices_columns = [f"shot_indices_{i}" for i in range(args.n_experiments)]
    columns_to_maintain = ['label', "benchmark", "id_bench"] + prompt_exp_columns + shot_indices_columns
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_maintain]
    dataset = dataset.remove_columns(columns_to_remove)
    all_benchmarks.append(dataset)


# --------------------------------------------------------------------------------
# AGORA, VAMOS FORMATAR OS EXPERIMENTOS PARA CADA PROMPT FICAR EM UMA LINHA
all_benchmarks = concatenate_datasets(all_benchmarks)

df = all_benchmarks.to_pandas()
id_vars = [col for col in df.columns if col not in prompt_exp_columns and col not in shot_indices_columns]

col_mapping = {f"prompt_{i}": f"shot_indices_{i}" for i in range(args.n_experiments)}

melted_df = []
for idx, row in df.iterrows():
    for prompt_col, shot_indices_col in col_mapping.items():
        new_row = {col: row[col] for col in id_vars}
        new_row['prompt'] = row[prompt_col]
        new_row['shot_indices'] = row[shot_indices_col]
        melted_df.append(new_row)

df = pd.DataFrame(melted_df)

df['id'] = list(range(len(df)))
column_order = ['id', 'id_bench', 'benchmark', 'prompt', 'shot_indices', 'label']
df = df[column_order]
dataset = Dataset.from_pandas(df)

dataset.push_to_hub(args.prompts_path)
