from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import torch

import argparse

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = load_dataset("mestras-valcir/merged_5shot_3exp_v2", split='train')
df = dataset.to_pandas()
# df = df[df['benchmark']!='bluex']
df['model_name'] = model_name.split('/')[-1]
dataset = Dataset.from_pandas(df)

# dataset = dataset.select(list(range(5)))


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def generate_answer(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1,        # Maximum number of tokens to generate
        do_sample=False,       # Use sampling to generate text (instead of greedy decoding)
        # temperature=0.7,      # Controls randomness: lower is less random
        # top_k=50,             # Limit to top-k tokens at each step
        # top_p=0.95            # Nucleus sampling: probability mass to consider
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    # vamos pegar apenas a primeira letra
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def map_answer(example):
    model_answer = generate_answer(example['prompt'])
    model_answer =model_answer.split('assistant')[-1]
    final_prompt = example['prompt'].split('assistant')[-1]
    model_answer = model_answer[len(final_prompt):]


    example['model_answer'] = model_answer[0]
    return example


dataset = dataset.map(map_answer)

original_dataset = load_dataset("pt-eval/evaluate_5shot_3exp", split='train')
full_dataset = concatenate_datasets([original_dataset, dataset])
full_dataset.push_to_hub(f"pt-eval/evaluate_5shot_3exp")


# dataset.push_to_hub(f"pt-eval/evaluate_5shot_3exp")