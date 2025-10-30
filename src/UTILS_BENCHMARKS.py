import ast

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


class bluex:
    def __init__(self):
        self.dataset_path = "eduagarcia-temp/BLUEX_without_images"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['choices']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class bluex_2022_2023:
    def __init__(self):
        self.dataset_path = "cemig-ceia/bluex_raw_noimg_2022_2023"
        self.subset = None
        self.split = "train"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        
        if isinstance(example['alternatives'], str):
            example['alternatives'] = ast.literal_eval(example['alternatives'])
        
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['alternatives']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['alternatives']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class enem:
    def __init__(self):
        self.dataset_path = "eduagarcia/enem_challenge"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['choices']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class enem_2022_2023:
    def __init__(self):
        self.dataset_path = "cemig-ceia/vestibulares_concatened"
        self.subset = None
        self.split = "enem"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        if isinstance(example['alternatives'], str):
            example['alternatives'] = ast.literal_eval(example['alternatives'])

        alts = example['alternatives']
        num_alternatives_example = len(alts['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['alternatives']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class hatebr:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "HateBR_offensive_binary"
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


class portuguese_hate_speech:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "Portuguese_Hate_Speech_binary"
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative

        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


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


class oab:
    def __init__(self):
        self.dataset_path = "cemig-ceia/vestibulares_concatened"
        self.subset = None
        self.split = "oab"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        alts = example['alternatives']
        if isinstance(alts, str):
            alternatives_dict = ast.literal_eval(alts)
        else:
            alternatives_dict = alts

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class poscomp:
    def __init__(self):
        self.dataset_path = "cemig-ceia/vestibulares_concatened"
        self.subset = None
        self.split = "poscomp"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        alts = example['alternatives']
        if isinstance(alts, str):
            alternatives_dict = ast.literal_eval(alts)
        else:
            alternatives_dict = alts

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)


        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class energy_regulacao:
    def __init__(self):
        self.dataset_path = "cemig-ceia/energy-eval"
        self.subset = None
        self.split = "train"
        self.important_columns = ["right_context", "question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)


        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class gpqa_diamond:
    def __init__(self):
        self.dataset_path = "cemig-ceia/GPQA-portuguese"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


# Novos benchmarks a partir daqui
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

class aime25:
    def __init__(self):
        self.dataset_path = "cemig-ceia/aime25-portuguese"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question_pt", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "integer_exact_math"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Problema: {example['question_pt']}\n\nForneça apenas a resposta final como um número inteiro, sem formatação de LaTeX ou matemática."

        prompt_informations['assistant_message_with_answer'] = f"Resposta: {example['label']}"
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

class mmlu:
    def __init__(self):
        self.dataset_path = "cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        num_alternatives_example = len(texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class mmlu_english:
    def __init__(self):
        self.dataset_path = "cemig-ceia/MMLU-formatted"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        num_alternatives_example = len(texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class mmlu_hard:
    def __init__(self):
        self.dataset_path = "cemig-ceia/mmlu-pt"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        num_alternatives_example = len(texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class mmlu_redux_english:
    def __init__(self):
        self.dataset_path = "cemig-ceia/MMLU-Redux-formatted"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        num_alternatives_example = len(texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class mmlu_pro_english:
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
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        # Filtrar respostas vazias/placeholder " - "
        filtered_texts = [t for t in texts if (t is not None) and (str(t).strip() not in {'-', '–'}) and (str(t).strip() != '')]

        num_alternatives_example = len(filtered_texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {filtered_texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta ou complete corretamente as lacunas:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class supergpqa:
    def __init__(self):
        self.dataset_path = "cemig-ceia/benchmark_Qwen3-235B-A22B_SuperGPQA_v1"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"

        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class supergpqa_lucas:
    def __init__(self):
        self.dataset_path = "cemig-ceia/SuperGPQA-formatted"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question", "choices", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        choices_data = example['choices']
        if isinstance(choices_data, str):
            alternatives_dict = ast.literal_eval(choices_data)
        else:
            alternatives_dict = choices_data

        texts = alternatives_dict['text']
        filtered_texts = [t for t in texts if (t is not None) and (str(t).strip() not in {'-', '–'}) and (str(t).strip() != '')]

        num_alternatives_example = len(filtered_texts)
        example_alternatives = [f"({chr(65 + asc_letter)}): {filtered_texts[asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)

        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

BENCHMARKS_INFORMATIONS = {
    "assin2rte": assin2rte(),
    "assin2sts": assin2sts(),
    "bluex": bluex_2022_2023(),
    "enem": enem_2022_2023(),
    "hatebr": hatebr(),
    "portuguese_hate_speech": portuguese_hate_speech(),
    "toxsyn_pt": toxsyn_pt(),
    "faquad": faquad(),
    "tweetsentbr": tweetsentbr(),
    "oab": oab(),
    "poscomp": poscomp(),
    "energy_regulacao": energy_regulacao(),
    #"gpqa_diamond": gpqa_diamond(),
    #Inserir a partir daqui
    "aime24": aime24(),
    "aime25": aime25(),
    "mmlu": mmlu(), # Modifiquei para usar a do Emanuel
    "mmlu_hard": mmlu_hard(),
    "mmlu_redux": mmlu_redux_english(),
    "mmlu_pro": mmlu_pro_english(),
    "supergpqa": supergpqa(),
    }
