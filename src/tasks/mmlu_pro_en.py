import ast


class mmlu_pro_en:
    def __init__(self):
        self.dataset_path = "cemig-ceia/MMLU-Pro-formatted"
        self.subset = None
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
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        # Filter out empty or placeholder values
        filtered_texts = [t for t in texts if (t is not None) and (str(t).strip() not in {'-', '–'}) and (str(t).strip() != '')]

        num_alternatives_example = len(filtered_texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {filtered_texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Question:\n{example['question']}.\n\nRead the options below and answer correctly or complete the blanks correctly:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = (
            f"Reading the options {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} "
            f"the letter that correctly answers the question is ({example['label']})"
        )
        prompt_informations['assistant_message_without_answer'] = (
            f"Reading the options {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} "
            f"the letter that correctly answers the question is ("
        )
        return prompt_informations

