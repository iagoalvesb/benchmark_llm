class enem:
    def __init__(self):
        self.dataset_path = "eduagarcia/enem_challenge"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['choices']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

