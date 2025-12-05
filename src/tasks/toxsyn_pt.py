class toxsyn_pt:
    def __init__(self):
        self.dataset_path = "ToxSyn/ToxSyn-PT"
        self.subset = None
        self.split = "test"
        self.important_columns = ["text", "is_toxic"]
        self.label_column = "is_toxic"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['text']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

