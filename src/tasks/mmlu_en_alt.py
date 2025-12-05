import ast


class mmlu_en_alt:
    def __init__(self):
        self.dataset_path = "cais/mmlu"
        self.subset = "all"
        self.split = "test"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "You are a helpful assistant. Answer directly and objectively."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives = ast.literal_eval(choices_data)
        else:
            alternatives = choices_data

        # choices is a list in the new format
        texts = list(alternatives)

        num_alternatives_example = len(texts)
        example_alternatives = "\n".join(f"({chr(65 + i)}): {texts[i]}" for i in range(num_alternatives_example))

        prompt_informations['user_message'] = (
            f"Question:\n{example['question']}.\n\n"
            f"Read the options below and answer correctly:\n\n{example_alternatives}"
        )

        correct_index = int(example["label"])
        correct_label = chr(65 + correct_index)

        letters = ''.join(f"({chr(65 + i)})," for i in range(num_alternatives_example))
        prompt_informations['assistant_message_with_answer'] = (
            f"Reading the options {letters} the letter that correctly answers the question is ({correct_label})"
        )
        prompt_informations['assistant_message_without_answer'] = (
            f"Reading the options {letters} the letter that correctly answers the question is ("
        )
        return prompt_informations

