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
dataset = load_dataset("eduagarcia-temp/BLUEX_without_images", split="train")
df = dataset.to_pandas()

def get_shots(id_query, n_shots=args.n_shots):
    # não vamos ter nossa query vazada no few_shot
    df_fewshot = df[df['id'] != id_query]
    dataset_fewshot = datasets.Dataset.from_pandas(df_fewshot)
    possible_shots_indx = list(range(len(dataset_fewshot)))
    shots_idx = random.sample(possible_shots_indx, n_shots)
    shots = dataset_fewshot.select(shots_idx)
    return shots


def get_prompt(example, n_shots=args.n_shots, n_experiments=args.n_experiments):
    for i in range(n_experiments):

        shots = get_shots(example['id'], n_shots)

        chat = [{"role": "system", "content": "Você é um assistente prestativo, responda de forma direta e objetiva."}]
        
        for shot in shots:
            num_alternatives_shot = len(shot['choices']['text'])
            shot_alternatives = [f"({chr(65 + asc_letter)}): {shot['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_shot)]
            shot_alternatives = "\n".join(shot_alternatives)
            user_shot_message = f"""Pergunta:\n{shot["question"]}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{shot_alternatives}"""
            assistant_shot_message = f"""Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_shot)))} a alternativa que responde corretamente a pergunta é a letra ({shot['answerKey']})"""
            
            chat.append({"role": "user", "content": user_shot_message})
            chat.append({"role": "assistant", "content": assistant_shot_message})
        
        
        num_alternatives = len(example['choices']['text'])
        user_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives)]
        user_alternatives = "\n".join(user_alternatives)
        user_message = f"""Pergunta:\n{example["question"]}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{user_alternatives}"""
        assistant_message = f"""Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives)))} a alternativa que responde corretamente a pergunta é a letra ("""

        chat.append({"role": "user", "content": user_message})
        chat.append({"role": "assistant", "content": assistant_message})

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        example[f"prompt_{i}"] = prompt
    return example


dataset = dataset.map(get_prompt, num_proc=32)
dataset_name = f'bluex_{args.n_shots}shot_{args.n_experiments}exp'
dataset.push_to_hub(f"mestras-valcir/{dataset_name}")