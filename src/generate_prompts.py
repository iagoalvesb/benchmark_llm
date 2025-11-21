from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets, Value
import random
import os
import logging
import math
from logger_config import init_logger
import pandas as pd
import torch
import json
import ast
import re
import argparse

from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
from utils import build_api_user_block, _escape_xml
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
    "--model_types",
    type=str,
    nargs='+',
    required=True,
    help="Model types for each model (instruct or base)"
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

parser.add_argument(
    "--run_local",
    action="store_true",
    help="If set, save results locally as CSV instead of pushing to HuggingFace Hub"
)

parser.add_argument(
    "--use_outlines",
    action="store_true",
    help="When enabled, append JSON-format instruction to the final user message and leave assistant final message empty"
)

parser.add_argument(
    "--use_percentage_dataset",
    type=float,
    default=100.0,
    help="Percentage of the dataset to use (0-100), always rounds up"
)

args = parser.parse_args()
init_logger()

if args.use_outlines:
    for model_path, model_type in zip(args.model_paths, args.model_types):
        if model_type == 'base':
            raise ValueError(
                f"Error: Modelos base não podem usar o outlines. Modelo '{model_path}' é um modelo base.\n"
                f"Faça um dos seguintes:\n"
                f"  1. Deixe use_outlines como false no seu config (recomendado de qualquer forma, tem alguns bugs com o outlines), ou\n"
                f"  2. Mude o model_type para 'instruct' se este modelo suporta chat templates"
            )

tokenizer_to_model_info = {}
for model_path, tokenizer_path, model_type in zip(args.model_paths, args.model_tokenizers, args.model_types):
    if tokenizer_path not in tokenizer_to_model_info:
        tokenizer_to_model_info[tokenizer_path] = {'instruct': [], 'base': []}
    tokenizer_to_model_info[tokenizer_path][model_type].append(model_path)

tokenizers = {}
for tokenizer_path in tokenizer_to_model_info.keys():
    if tokenizer_path == "api":
        continue
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.pad_token_id = tok.eos_token_id
    tokenizers[tokenizer_path] = tok

def build_outlines_instruction(benchmark_obj) -> str:
    b = benchmark_obj
    base = (
        "IMPORTANTE: Responda APENAS no formato JSON com as chaves 'explicacao' e 'resposta'. "
        "A chave 'explicacao' deve conter um raciocínio breve e objetivo. "
        "A chave 'resposta' deve conter SOMENTE a resposta final no formato especificado abaixo.\n"
    )
    if b.answer_pattern == "yes_no":
        spec = "Para 'resposta', use exatamente 'Sim' ou 'Não'."
    elif b.answer_pattern == "multiple_choice":
        spec = "Para 'resposta', use exatamente UMA letra entre 'A', 'B', 'C', 'D' ou 'E'."
    elif b.answer_pattern == "multiple_choice_full_word":
        spec = "Para 'resposta', use exatamente uma palavra entre 'Positivo', 'Negativo' ou 'Neutro'."
    elif b.answer_pattern == "continue_value":
        spec = "Para 'resposta', use apenas um número (use ponto decimal se necessário)."
    elif b.answer_pattern == "integer_exact_math":
        spec = "Para 'resposta', forneça apenas um número inteiro (sem LaTeX)."
    else:
        spec = "Para 'resposta', forneça apenas o valor final esperado para o benchmark."
    example = "Exemplo: {\"explicacao\": \"...\", \"resposta\": \"...\"}"
    return base + spec + "\n" + example

def build_raw_text_prompt(chat, use_outlines=False):
    """
    Prompt seco para modelo base sem chat templates.
    """
    parts = []
    for msg in chat:
        role = msg['role'].capitalize()
        content = msg['content']
        if content:
            parts.append(f"{role}: {content}")
    
    return "\n\n".join(parts)

def get_shots(id_query, dataset_fewshot, n_shots=args.n_shots, use_fixed_seed=args.use_fixed_seed, seed=42):
    if n_shots == 0:
        return [], []

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
        # shots, shot_id_benches = get_shots(example['idx'], dataset, n_shots, args.use_fixed_seed)
        seed_i = 42 + i if args.use_fixed_seed else None
        shots, shot_id_benches = get_shots(example['idx'], dataset_benchmark, n_shots,
                                        use_fixed_seed=args.use_fixed_seed, seed=seed_i)
        example_informations = benchmark.get_prompt_informations(example)

        #example_informations['user_message'] = example_informations['user_message'].replace("\\uparrow$", "\\\\uparrow$")
        #example_informations['assistant_message_without_answer'] = example_informations['assistant_message_without_answer'].replace("\\uparrow$", "\\\\uparrow$")

        chat = [{"role": "system", "content": example_informations['base_system_message']}]

        if shots:
            for shot in shots:
                shot_informations = benchmark.get_prompt_informations(shot)
               # shot_informations['user_message'] = shot_informations['user_message'].replace("\\uparrow$", "\\\\uparrow$")
               # shot_informations['assistant_message_with_answer'] = shot_informations['assistant_message_with_answer'].replace("\\uparrow$", "\\\\uparrow$")
                chat.append({"role": "user", "content": shot_informations['user_message']})
                chat.append({"role": "assistant", "content": shot_informations['assistant_message_with_answer']})

        final_user_message = example_informations['user_message']
        if args.use_outlines:
            final_user_message = f"{final_user_message}\n\n{build_outlines_instruction(benchmark)}"
            final_assistant_message = ""
        else:
            final_assistant_message = example_informations['assistant_message_without_answer']

        chat.append({"role": "user", "content": final_user_message})
        if not args.use_outlines:
            chat.append({"role": "assistant", "content": final_assistant_message})

        # Generate prompts for each unique tokenizer
        prompt_dict = {}
        for tokenizer_path in tokenizer_to_model_info.keys():
            if tokenizer_path == "api":
                user_text, assistant_hint = build_api_user_block(
                    benchmark, shots, example_informations, args.use_outlines
                )
                # wrap with simple tags to keep structure explicit
                parts = []
                parts.append(f"<system>{_escape_xml(example_informations['base_system_message'])}</system>")
                parts.append(f"<user>{_escape_xml(user_text)}</user>")
                if assistant_hint is not None:
                    parts.append(f"<assistant>{_escape_xml(assistant_hint)}</assistant>")
                raw_prompt = "\n".join(parts)

                # All API models get the same prompt
                all_api_models = tokenizer_to_model_info[tokenizer_path]['instruct'] + tokenizer_to_model_info[tokenizer_path]['base']
                prompt_dict[tokenizer_path] = {
                    "prompt": raw_prompt,
                    "models": all_api_models,
                }
                continue

            tokenizer = tokenizers[tokenizer_path]
            
            instruct_models = tokenizer_to_model_info[tokenizer_path]['instruct']
            if instruct_models:
                if args.use_outlines:
                    rendered = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                else:
                    rendered = tokenizer.apply_chat_template(
                        chat, tokenize=False, continue_final_message=True
                    )
                prompt_dict[f"{tokenizer_path}_instruct"] = {
                    "prompt": rendered,
                    "models": instruct_models,
                }
            
            base_models = tokenizer_to_model_info[tokenizer_path]['base']
            if base_models:
                raw_prompt = build_raw_text_prompt(chat, args.use_outlines)
                prompt_dict[f"{tokenizer_path}_base"] = {
                    "prompt": raw_prompt,
                    "models": base_models,
                }

        example[f"prompt_{i}"] = json.dumps(prompt_dict, ensure_ascii=False)
        example[f"shot_indices_{i}"] = shot_id_benches
    return example

def load_benchmark_dataset(benchmark):
    path = benchmark.dataset_path
    subsets = benchmark.subset

    if subsets is None or isinstance(subsets, str):
        ddict = load_dataset(path, subsets)
        if hasattr(benchmark, 'split') and benchmark.split == "poscomp":
            ds = ddict['poscomp']
            ds = ds.map(sanitize_latex, num_proc=1, desc="Sanitizing alternatives for poscomp")
        elif hasattr(benchmark, 'split'):
            ds = ddict[benchmark.split]
        else:
            ds = ddict['test'] if 'test' in ddict.keys() else ddict['train']
        return ds

    if not isinstance(subsets, (list, tuple)):
        raise TypeError("benchmark.subset must be str | list[str] | None")

    loaded = []
    first_features = None
    for sub in subsets:
        ddict = load_dataset(path, sub)
        if hasattr(benchmark, 'split') and benchmark.split == "poscomp":
            ds = ddict['poscomp']
            ds = ds.map(sanitize_latex, num_proc=1, desc=f"Sanitizing alternatives for poscomp ({sub})")
        elif hasattr(benchmark, 'split'):
            ds = ddict[benchmark.split]
        else:
            ds = ddict['test'] if 'test' in ddict.keys() else ddict['train']

        # schema check
        if first_features is None:
            first_features = ds.features
        else:
            if ds.features != first_features:
                raise ValueError(
                    f"Subset '{sub}' has a different schema. "
                    f"Expected {list(first_features.keys())}, got {list(ds.features.keys())}."
                )

        ds = ds.add_column("subset_name", [sub] * len(ds))
        loaded.append(ds)

    return concatenate_datasets(loaded)

def sanitize_latex(example):
    """
    Esta função lê o campo 'alternatives', corrige problemas de sintaxe do LaTeX,
    converte para um dicionário para manipulação interna e
    converte o dicionário limpo de volta para uma string JSON padronizada.
    """
    alternatives_data = example.get("alternatives")

    # Se o campo não for uma string ou estiver vazio, retorna como está
    if not isinstance(alternatives_data, str) or not alternatives_data.strip():
        return example

    try:
        # 1. Corrige os backslashes para que a string seja "parseável" pelo Python
       # safe_string = alternatives_data.replace('\\', '\\\\')
        safe_string = alternatives_data.replace(r'\uparrow', r'\\uparrow')
        safe_string = safe_string.replace(r'\underline', r'\\underline') 
        # 2. Converte a string segura para um dicionário
        alternatives_dict = ast.literal_eval(safe_string)

        # 3. Garantir o formato
        if 'text' not in alternatives_dict:
            alternatives_dict['text'] = []
        if 'label' not in alternatives_dict:
            alternatives_dict['label'] = []

        # 4. Converte o dicionário limpo de volta para uma string.
        example['alternatives'] = json.dumps(alternatives_dict, ensure_ascii=False)

    except (ValueError, SyntaxError):
        # Se mesmo com a correção a string for inválida,
        # mantém o valor original para não quebrar o fluxo.
        pass

    return example

all_benchmarks = []
for benchmark_name in args.benchmark_names:
    benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]

    dataset = load_benchmark_dataset(benchmark)

    # REMOVER QUALQUER LINHA QUE TENHA NAN
    if hasattr(benchmark, 'important_columns'):
        dataset = dataset.filter(lambda x: all((x[col] is not None) and (x[col] != "") for col in benchmark.important_columns if col in x))
    else:
        dataset = dataset.filter(lambda x: all((x[col] is not None) and (x[col] != "") for col in x))

    if args.use_percentage_dataset < 100:
        total_samples = len(dataset)
        num_samples = math.ceil(total_samples * args.use_percentage_dataset / 100)
        selected_indices = list(range(min(num_samples, total_samples)))
        dataset = dataset.select(selected_indices)
        logging.info(f"Selected {len(dataset)} samples out of {total_samples} for {benchmark_name} ({args.use_percentage_dataset}% of dataset)")

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

if args.run_local:
    os.makedirs("eval_processing", exist_ok=True)
    
    filename = args.prompts_path.replace("/", "_").replace("-", "_") + ".csv"
    filepath = os.path.join("eval_processing", filename)
    
    df.to_csv(filepath, index=False)
    logging.info(f"Prompts saved locally to: {filepath}")
else:
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(args.prompts_path)
    logging.info(f"Prompts saved to HuggingFace Hub: {args.prompts_path}")
