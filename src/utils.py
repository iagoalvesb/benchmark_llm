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
        "energy_dataset": 0.5,
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
        else:
            raise ValueError(f"Unknown answer pattern: {benchmark.answer_pattern}")
    except:
        return None