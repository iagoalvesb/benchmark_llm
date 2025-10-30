import pandas as pd
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
import re

# Mapeamento de benchmarks para áreas do conhecimento
BENCHMARK_TO_AREA = {
    'hatebr':                 'Discurso de Ódio',
    'portuguese_hate_speech': 'Discurso de Ódio',
    'tweetsentbr':            'Discurso de Ódio',
    'toxsyn_pt':              'Discurso de Ódio',
    'oab':                    'Área do Direito',
    'enam':                   'Área do Direito',
    'revalida':               'Área Médica',
    'mrex':                   'Área Médica',
    'afa':                    'Provas Militares',
    'ita':                    'Provas Militares',
    'ime':                    'Provas Militares',
    'poscomp':                'Computação',
    'obi':                    'Computação',
    'bcb':                    'Economia e Contabilidade',
    'cfces':                  'Economia e Contabilidade',
    'assin2rte':              'Semântica e Inferência',
    'assin2sts':              'Semântica e Inferência',
    'faquad':                 'Semântica e Inferência',
    'bluex':                  'Multidisciplinar',
    'enem':                   'Multidisciplinar',
    'cnpu':                   'Multidisciplinar',
    'enade':                  'Multidisciplinar',
    'bndes':                  'Multidisciplinar',
    'cacd_1_fase':            'Multidisciplinar',
    'cacd_2_fase':            'Multidisciplinar',
    'energy_regulacao':       'Energy',
    'gpqa_diamond':           'STEM',
    'aime24':                 'STEM',
    'aime25':                 'STEM',
    'mmlu':                   'General Tasks (Inglês)',
    'mmlu_hard':              'General Tasks (Inglês)',
    'mmlu_redux':             'General Tasks (Inglês)',
    'mmlu_pro':               'General Tasks (Inglês)',
    'supergpqa':              'General Tasks (Inglês)',
}

# Mapeamento de benchmarks para formatação do leaderboard (temporario provavelmente)
BENCHMARK_TO_COLUMN = {
    'hatebr':                 'HateBR',
    'portuguese_hate_speech': 'PT Hate Speech',
    'tweetsentbr':            'tweetSentBR',
    'toxsyn_pt':              'ToxSyn-PT',
    'oab':                    'OAB',
    'revalida':               'Revalida',
    'mrex':                   'MREX',
    'enam':                   'ENAM',
    'afa':                    'AFA',
    'ita':                    'ITA',
    'ime':                    'IME',
    'poscomp':                'POSCOMP',
    'obi':                    'OBI',
    'bcb':                    'BCB',
    'cfces':                  'CFCES',
    'assin2rte':              'ASSIN2 RTE',
    'assin2sts':              'ASSIN2 STS',
    'faquad':                 'FAQUAD NLI',
    'bluex':                  'BLUEX',
    'enem':                   'ENEM',
    'cnpu':                   'CNPU',
    'enade':                  'ENADE',
    'bndes':                  'BNDES',
    'cacd_1_fase':            'CACD (1ª fase)',
    'cacd_2_fase':            'CACD (2ª fase)',
    'energy_regulacao':       'ENERGY-REGULAÇÃO',
    'gpqa_diamond':           'GPQA-Diamond',
    'aime24':                 'AIME 24',
    'aime25':                 'AIME 25',
    'mmlu':                   'MMLU',
    'mmlu_hard':                   'MMLU-Hard',
    'mmlu_redux':             'MMLU-Redux',
    'mmlu_pro':               'MMLU-Pro',
    'supergpqa':              'SuperGPQA'
}

# Mapeamento de benchmarks para métricas
BENCHMARK_TO_METRIC = {
    'hatebr':                 ['accuracy'],
    'portuguese_hate_speech': ['accuracy'],
    'tweetsentbr':            ['accuracy'],
    'toxsyn_pt':              ['accuracy'],
    'oab':                    ['accuracy'],
    'enam':                   ['accuracy'],
    'revalida':               ['accuracy'],
    'mrex':                   ['accuracy'],
    'afa':                    ['accuracy'],
    'ita':                    ['accuracy'],
    'ime':                    ['accuracy'],
    'poscomp':                ['accuracy'],
    'obi':                    ['accuracy'],
    'bcb':                    ['accuracy'],
    'cfces':                  ['accuracy'],
    'assin2rte':              ['accuracy'],
    'assin2sts':              ['pearson_correlation'],
    'faquad':                 ['accuracy'],
    'bluex':                  ['accuracy'],
    'enem':                   ['accuracy'],
    'cnpu':                   ['accuracy'],
    'enade':                  ['accuracy'],
    'bndes':                  ['accuracy'],
    'cacd_1_fase':            ['accuracy'],
    'cacd_2_fase':            ['accuracy'],
    'energy_regulacao':       ['accuracy'],
    'gpqa_diamond':           ['accuracy'],
    'aime24':                 ['accuracy'],
    'aime25':                 ['accuracy'],
    'mmlu':                   ['accuracy'],
    'mmlu_hard':              ['accuracy'],
    'mmlu_redux':             ['accuracy'],
    'mmlu_pro':               ['accuracy'],
    'supergpqa':              ['accuracy'],
}

##############################################################################

MODEL_PARAMS = {
    'Qwen2.5-1.5B-Instruct': {
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research'
    },

    'Qwen2.5-0.5B-Instruct': {
        'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research'
    },

    'Qwen2.5-3B-Instruct': {
        'model_id': 'Qwen/Qwen2.5-3B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research'
    },

    'Llama-3.2-1B-Instruct': {
        'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'llama3.2'
    },

    'Llama-3.2-3B-Instruct': {
        'model_id': 'meta-llama/Llama-3.2-3B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'llama3.2'
    },

    'Qwen2.5-7B-Instruct': {
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research',

        'precisao': 'BF16',
        'arquitetura': 'N/A',
        'params_b': 7.0,
        'hub_likes': 0,
        'disponivel_no_hub': False,
        'sha_modelo': 'N/A',
    },

    'qwen2.5-7B-1E_fulltrain': {
        'model_id': 'qwen2.5-7B-2E_fulltrain',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research',

        'precisao': 'BF16',
        'arquitetura': 'N/A',
        'params_b': 7.0,
        'hub_likes': 0,
        'disponivel_no_hub': False,
        'sha_modelo': 'N/A',
    },

    'qwen2.5-7B-2E_fulltrain': {
        'model_id': 'qwen2.5-7B-2E_fulltrain',
        't': 'SFT',
        'tipo': 'SFT : Supervised Finetuning',
        'tipo_peso': 'Original',
        'licenca': 'qwen-research',

        'precisao': 'BF16',
        'arquitetura': 'N/A',
        'params_b': 7.0,
        'hub_likes': 0,
        'disponivel_no_hub': False,
        'sha_modelo': 'N/A',
    },

}

#######################################################################
#OK
def add_additional_info(data):
    benchmarks = {
        "Datasets Área Médica": "Revalida, MREX",
        "Datasets Área do Direito": "OAB, ENAM",
        "Datasets Provas Militares": "AFA, ITA, IME",
        "Datasets Computação": "POSCOMP, OBI",
        "Datasets Discurso de Ódio": "HateBR, PT Hate Speech, tweetSentBR, ToxSyn-PT",
        "Datasets Economia e Contabilidade": "BCB, CFCES",
        "Datasets Semântica e Inferência": "FAQUAD NLI, ASSIN2 RTE, ASSIN2 STS",
        "Datasets Multidisciplinar": "ENEM, BLUEX, CNPU, ENADE, BNDES, CACD (1ª fase), CACD (2ª fase)",
        "energy_dataset": "ENERGY-REGULAÇÃO",
        "STEM": "AIME 24, AIME 25, GPQA-Diamond",
        "General Tasks (Inglês)": "MMLU, MMLU-Redux, MMLU-Pro, SuperGPQA",
        "reasoning_dataset": 0.5
    }

    for area, value in benchmarks.items():
        data[area] = value

    return data

def clean_index_columns(df):
    """Remove any index-related columns that may have been created by CSV round-trips"""
    index_cols = [col for col in df.columns if col.startswith('__index_level_') or col == 'Unnamed: 0']
    if index_cols:
        df = df.drop(columns=index_cols)
    return df

##############################################################################

# Generate Answers utils
def parse_yes_no(text):
    text_lower = text.lower()
    if "sim" in text_lower:
        return "1"
    elif "não" in text_lower:
        return "0"

    text = text.strip()[0].upper()
    answer = 1 if text == 'S' else 0 if text == 'N' else None
    if answer is None:
        return None
    return str(answer)

def parse_multiple_choice(text):
    # Extract the first character of the answer
    text = text.strip()[0].upper()
    return text[0]

def parse_multiple_choice_full_word(text):
    # Extract the first word of the answer
    stripped_text = text.strip()
    words = stripped_text.split()
    first_word = words[0].capitalize() # Capitalize the first word just in case
    first_word = words[0].rstrip('.,!?;:').capitalize()
    return first_word

def parse_continue_value(text):
    # Extract the first character of the answer
    text = text.strip()
    match = re.search(r"\d+([\.,]?\d+)?", text)
    text = match.group()
    text = text.replace(',', '.')
    return text

def parse_integer_math_format(text):
    """Extrai a parte inteira de uma resposta possivelmente formatada em LaTeX/matemática.

    Exemplos aceitos: "336^\circ", "336^{\circ}", "336°", "\(336^\circ\)", "-12.0".
    Retorna a string do inteiro normalizado (sem zeros à esquerda) ou None se não encontrar.
    """
    if text is None:
        return None

    s = str(text).strip()

    # Remover delimitadores LaTeX e marcadores comuns
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", s)  # \( \) \[ \]
    s = re.sub(r"\$+", "", s)  # $ ou $$
    s = re.sub(r"\\mathrm\{", "", s)  # \mathrm{
    s = s.replace("}", "")

    # Remover marcações de graus e similares
    s = re.sub(r"\^\s*\{\\?circ\}", "", s)  # ^{\circ}
    s = re.sub(r"\^\s*\\?circ", "", s)       # ^\circ ou ^circ
    s = re.sub(r"\\(deg|degree|circ)", "", s)  # \deg, \degree, \circ
    s = s.replace("°", "")

    # Capturar número (opcionalmente com decimal) e sinal
    match = re.search(r"[-+]?\d+(?:[\.,]\d+)?", s)
    if not match:
        return None
    num_str = match.group(0)

    # Normalizar separador decimal e obter parte inteira
    if ',' in num_str and '.' in num_str:
        # Caso raro: manter apenas antes do primeiro separador
        integral_part = re.split(r"[\.,]", num_str)[0]
    else:
        integral_part = num_str.split('.')[0].split(',')[0]

    try:
        return str(int(integral_part))
    except Exception:
        cleaned = re.sub(r"[^\d+-]", "", integral_part)
        try:
            return str(int(cleaned))
        except Exception:
            return None

def parse_answer(example):
    # Extract the answer in the correct format (e.g. anser "Resposta: E" to "E")
    benchmark_name = example['benchmark']
    benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
    try:
        if benchmark.answer_pattern == "yes_no":
            return parse_yes_no(example['model_answer'])
        elif benchmark.answer_pattern == "multiple_choice":
            return parse_multiple_choice(example['model_answer'])
        elif benchmark.answer_pattern == "multiple_choice_full_word":
            return parse_multiple_choice_full_word(example['model_answer'])
        elif benchmark.answer_pattern == "continue_value":
            return parse_continue_value(example['model_answer'])
        elif benchmark.answer_pattern == "integer_exact_math":
            return parse_integer_math_format(example['model_answer'])
        else:
            raise ValueError(f"Unknown answer pattern: {benchmark.answer_pattern}")
    except:
        return None
