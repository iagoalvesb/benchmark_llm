import ast


class mmlu_pro_x:
    def __init__(self):
        self.dataset_path = "li-lab/MMLU-ProX"
        self.subset = ["pt", "en"]
        self.split = "test"
        self.important_columns = ["question", "option_0", "option_1", "option_2", "option_3", "option_4", "option_5", "option_6", "option_7", "option_8", "option_9", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        option_keys = [f"option_{i}" for i in range(10)]
        texts = [example[k] for k in option_keys if example.get(k) is not None]

        num_alternatives_example = len(texts)
        example_alternatives = [f"({chr(65 + i)}): {texts[i]}" for i in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = (
            f"Pergunta:\n{example['question']}\n\n"
            f"Leia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"
        )

        letters = ''.join(f"({chr(65 + i)})," for i in range(num_alternatives_example))
        prompt_informations['assistant_message_with_answer'] = (
            f"Lendo as alternativas {letters} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        )
        prompt_informations['assistant_message_without_answer'] = (
            f"Lendo as alternativas {letters} a alternativa que responde corretamente a pergunta é a letra ("
        )
        return prompt_informations