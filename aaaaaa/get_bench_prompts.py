from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets, Value
import random
import torch

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
    "--tokenizer_path",
    type=str,
    required=True,
    help="Huggingface path to the tokenizer"
)

parser.add_argument(
    "--output_hub_repo",
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

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

def get_shots(id_query, dataset_fewshot, n_shots=args.n_shots):
    # não vamos ter nossa query vazada no few_shot
    possible_shots_indx = [i for i, example in enumerate(dataset_fewshot) if example['idx'] != id_query]
    shots_idx = random.sample(possible_shots_indx, n_shots)
    shots = dataset_fewshot.select(shots_idx)
    return shots


def get_prompt(example, benchmark, dataset_benchmark, n_shots=args.n_shots, n_experiments=args.n_experiments):
    for i in range(n_experiments):

        shots = get_shots(example['idx'], dataset, n_shots)
        example_informations = benchmark.get_prompt_informations(example)

        chat = [{"role": "system", "content": example_informations['base_system_message']}]
        for shot in shots:
            shot_informations = benchmark.get_prompt_informations(shot)            
            chat.append({"role": "user", "content": shot_informations['user_message']})
            chat.append({"role": "assistant", "content": shot_informations['assistant_message_with_answer']})
        
        chat.append({"role": "user", "content": example_informations['user_message']})
        chat.append({"role": "assistant", "content": example_informations['assistant_message_without_answer']})

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        example[f"prompt_{i}"] = prompt
    return example


all_benchmarks = []
for benchmark_name in args.benchmark_names:
    benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
    dataset = load_dataset(benchmark.dataset_path, benchmark.subset)
    dataset = dataset['test'] if 'test' in dataset.keys() else dataset['train']
    # REMOVER QUALQUER LINHA QUE TENHA NAN
    dataset = dataset.filter(lambda x: all((x[col] is not None) and (x[col] != "") for col in x))
    # dataset = dataset.select(list(range(25)))  # apenas para teste, depois tirar
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

    columns_to_maintain = ['label', "prompt_0", "prompt_1", "prompt_2", "benchmark", "id_bench"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_maintain]
    dataset = dataset.remove_columns(columns_to_remove)
    all_benchmarks.append(dataset)





# --------------------------------------------------------------------------------
# AGORA, VAMOS FORMATAR OS EXPERIMENTOS PARA CADA PROMPT FICAR EM UMA LINHA
all_benchmarks = concatenate_datasets(all_benchmarks)

df = all_benchmarks.to_pandas()
prompt_exp_columns = ['prompt_0', 'prompt_1', 'prompt_2']
id_vars = [col for col in df.columns if col not in prompt_exp_columns]
df = df.melt(
    id_vars=id_vars,  # Columns to keep
    value_vars=prompt_exp_columns,  # Columns to unpivot
    var_name='prompt_type',   # Name for the new column indicating the prompt column
    value_name='prompt'       # Name for the new column that holds the prompt text
)

df = df.drop(columns='prompt_type')

df['id'] = list(range(len(df)))
column_order = ['id', 'id_bench', 'benchmark', 'prompt', 'label']
df = df[column_order]
dataset = Dataset.from_pandas(df)



model_name = args.tokenizer_path.split("/")[-1]
dataset_name = f'prompts_{model_name[:3]}_{args.n_shots}shot_{args.n_experiments}exp'
dataset_output_path = f"{args.output_hub_repo}/{dataset_name}"

dataset.push_to_hub(dataset_output_path)
print(f'\n\n** DATASET SAVED AT: {dataset_output_path}')
