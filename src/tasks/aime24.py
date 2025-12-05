class aime24:
    def __init__(self):
        self.dataset_path = "cemig-ceia/aime24-portuguese"
        self.subset = None
        self.split = "train"
        self.important_columns = ["problem_pt", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "integer_exact_math"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Problema: {example['problem_pt']}\n\nForneça apenas a resposta final como um número inteiro, sem formatação de LaTeX ou matemática."

        prompt_informations['assistant_message_with_answer'] = f"Resposta: {example['label']}"
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

