class assin2sts:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "assin2-sts"
        self.split = "test"
        self.important_columns = ["idx", "sentence1", "sentence2", "label"]
        self.label_column = "label"
        self.answer_pattern = "continue_value"
        self.answer_type = "continue"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Sentença 1: {example['sentence1']}.\nSentença 2: {example['sentence2']}.\n\nApós ler as sentenças, o quão similares elas são? Utilize valores entre 1 e 5, sendo 1 pouco similares e 5 muito similares."
        prompt_informations['assistant_message_with_answer'] = f"Resposta: Ao ler a Sentença 1 e a Sentença 2, o valor da similaridade é de {example['label']}, ou seja, são {'pouco' if float(example['label']) < 3 else 'muito'} similares."
        prompt_informations['assistant_message_without_answer'] = "Resposta: Ao ler a Sentença 1 e a Sentença 2, o valor da similaridade é de "
        return prompt_informations

