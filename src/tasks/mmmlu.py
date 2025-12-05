class mmmlu:
    def __init__(self):
        self.dataset_path = "openai/MMMLU"
        self.subset = "PT_BR"
        self.split = "test"
        self.important_columns = ["Question", "A", "B", "C", "D", "Answer"]
        self.label_column = "Answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        texts = [example['A'], example['B'], example['C'], example['D']]
        num_alternatives_example = len(texts)
        example_alternatives = "\n".join(f"({chr(65 + i)}): {texts[i]}" for i in range(num_alternatives_example))

        prompt_informations['user_message'] = (
            f"Pergunta:\n{example['Question']}.\n\n"
            f"Leia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"
        )

        correct_label = example["label"]
        letters = ''.join(f"({chr(65 + i)})," for i in range(num_alternatives_example))

        prompt_informations['assistant_message_with_answer'] = (
            f"Lendo as alternativas {letters} a alternativa que responde corretamente a pergunta é a letra ({correct_label})"
        )
        prompt_informations['assistant_message_without_answer'] = (
            f"Lendo as alternativas {letters} a alternativa que responde corretamente a pergunta é a letra ("
        )
        return prompt_informations

