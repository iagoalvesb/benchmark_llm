from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import datasets
import random
import torch

import argparse


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

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = load_dataset("eduagarcia/portuguese_benchmark", 'assin2-sts', split="test")
df = dataset.to_pandas()

def get_shots(id_query, n_shots=args.n_shots):
    # não vamos ter nossa query vazada no few_shot
    df_fewshot = df[df['idx'] != id_query]
    dataset_fewshot = datasets.Dataset.from_pandas(df_fewshot)
    possible_shots_indx = list(range(len(dataset_fewshot)))
    shots_idx = random.sample(possible_shots_indx, n_shots)
    shots = dataset_fewshot.select(shots_idx)
    return shots


def get_prompt(example, n_shots=args.n_shots, n_experiments=args.n_experiments):
    for i in range(n_experiments):

        shots = get_shots(example['idx'], n_shots)

        chat = [{"role": "system", "content": "Você é um assistente prestativo, responda de forma direta e objetiva."}]
        
        for shot in shots:
            user_shot_message = f"""Sentença 1: {shot["sentence1"]}.\Sentença 2: {shot["sentence2"]}.\n\nApós ler as sentenças, o quão similares elas são? Utilize valores entre 1 e 5, sendo 1 pouco similares e 5 muito similares."""
            assistant_shot_message = f"Resposta: Ao ler a Sentança 1 e a Sentença 2, o valor da similaridade é de {shot['label']}, ou seja, são {'pouco' if shot['label'] < 3 else 'muito'} similares."
            
            chat.append({"role": "user", "content": user_shot_message})
            chat.append({"role": "assistant", "content": assistant_shot_message})
        
        user_message = f"""Sentença 1: {example["sentence1"]}.\Sentença 2: {example["sentence2"]}.\n\nApós ler as sentenças, o quão similares elas são? Utilize valores entre 1 e 5, sendo 1 pouco similares e 5 muito similares."""

        chat.append({"role": "user", "content": user_message})
        chat.append({"role": "assistant", "content": "Resposta: Ao ler a Sentança 1 e a Sentença 2, o valor da similaridade é de "})

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        example[f"prompt_{i}"] = prompt
    return example


dataset = dataset.map(get_prompt, num_proc=32)
dataset_name = f'assin2sts_{args.n_shots}shot_{args.n_experiments}exp'
dataset.push_to_hub(f"mestras-valcir/{dataset_name}")