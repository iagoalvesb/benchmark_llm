import pandas as pd

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