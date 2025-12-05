class assin2rte:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "assin2-rte"
        self.split = "test"
        self.important_columns = ["idx", "sentence1", "sentence2", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Premissa: {example['sentence1']}.\nHipótese: {example['sentence2']}.\n\nCom base na nossa premissa, a hipótese acima é verdadeira ou falsa? Responda com Sim ou Não."

        assistant_message_positive = "Resposta: Sim, com base na nossa premissa, a hipótese acima é verdadeira."
        assistant_message_negative = "Resposta: Não, com base na nossa premissa, a hipótese acima não é verdadeira."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

