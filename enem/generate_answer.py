from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="path to the dataset"
)

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="path to the model"
)
parser.add_argument(
    "--n_experiments",
    type=int,
    required=True,
    default=1,
    help="number of experiments on prompts"
)

args = parser.parse_args()

def generate_answer(example):
    base_assistant_message = 'Lendo as alternativas (A), (B), (C), (D), (E), a alternativa que responde corretamente a pergunta é a letra '
    for i in range(args.n_experiments):
        prompt_text = example[f'prompt_{i}']
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=3,        # Maximum number of tokens to generate
            do_sample=False,       # Use sampling to generate text (instead of greedy decoding)
            # temperature=0.7,      # Controls randomness: lower is less random
            # top_k=50,             # Limit to top-k tokens at each step
            # top_p=0.95            # Nucleus sampling: probability mass to consider
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        full_answer = generated_text.split(base_assistant_message)[-1] # formato é tipo (C)
        alternative_answer = full_answer[1] # pega o segundo caractere
        example[f'answer_{i}'] = alternative_answer
    return example


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer = tokenizer.to(device)
model = AutoModelForCausalLM.from_pretrained(args.model_path)
model = model.to(device)

dataset = load_dataset(args.dataset_path, split="train")
dataset = dataset.map(generate_answer)

dataset_name = args.dataset_path.split("/")[-1]
model_name = args.model_path.split("/")[-1]
dataset.push_to_hub(f"mestras-valcir/{dataset_name}_{model_name}")