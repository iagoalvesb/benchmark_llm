class faquad:
    def __init__(self):
        self.dataset_path = "ruanchaves/faquad-nli"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question", "answer", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Pergunta: {example['question']}\n\nResposta: {example['answer']}\nA resposta dada satisfaz à pergunta? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, a resposta dada satisfaz à pergunta."
        assistant_message_negative = "Resposta: Não, a resposta dada não satisfaz à pergunta."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == 1 else assistant_message_negative

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

