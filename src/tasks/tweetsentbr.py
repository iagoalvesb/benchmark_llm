class tweetsentbr:
    def __init__(self):
        self.dataset_path = "EdwardSJ151/tweetsentbr_fewshot_ptlabel"
        self.subset = None
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "multiple_choice_full_word"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nQual é o tom do comentário acima? Responda com apenas uma das seguintes opções: Positivo, Negativo ou Neutro."""

        label_map = {
            "Positivo": "Resposta: Positivo",
            "Negativo": "Resposta: Negativo",
            "Neutro": "Resposta: Neutro"
        }

        prompt_informations['assistant_message_with_answer'] = label_map.get(example['label'])

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

