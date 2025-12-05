import ast


class bluex_2022_2023:
    def __init__(self):
        self.dataset_path = "cemig-ceia/bluex_raw_noimg_2022_2023"
        self.subset = None
        self.split = "train"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        
        if isinstance(example['alternatives'], str):
            example['alternatives'] = ast.literal_eval(example['alternatives'])
        
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['alternatives']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['alternatives']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

