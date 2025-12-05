from tasks.assin2rte import assin2rte
from tasks.assin2sts import assin2sts
from tasks.bluex import bluex
from tasks.bluex_2022_2023 import bluex_2022_2023
from tasks.enem import enem
from tasks.enem_2022_2023 import enem_2022_2023
from tasks.hatebr import hatebr
from tasks.portuguese_hate_speech import portuguese_hate_speech
from tasks.toxsyn_pt import toxsyn_pt
from tasks.faquad import faquad
from tasks.tweetsentbr import tweetsentbr
from tasks.oab import oab
from tasks.poscomp import poscomp
from tasks.energy_regulacao import energy_regulacao
from tasks.gpqa_diamond import gpqa_diamond
from tasks.aime24 import aime24
from tasks.aime25 import aime25
from tasks.mmlu import mmlu
from tasks.mmlu_en import mmlu_en
from tasks.mmlu_en_alt import mmlu_en_alt
from tasks.mmlu_hard import mmlu_hard
from tasks.mmlu_redux_en import mmlu_redux_en
from tasks.mmlu_pro_en import mmlu_pro_en
from tasks.supergpqa import supergpqa
from tasks.supergpqa_en import supergpqa_en
from tasks.include import include
from tasks.mmmlu import mmmlu
from tasks.mmlu_pro_x import mmlu_pro_x


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
    "aime24": aime24(),
    "aime25": aime25(),
    "mmlu": mmlu(),
    "mmlu_en": mmlu_en(),
    "mmlu_hard": mmlu_hard(),
    "mmlu_redux_en": mmlu_redux_en(),
    "mmlu_pro_en": mmlu_pro_en(),
    "mmlu_pro_x": mmlu_pro_x(),
    "supergpqa": supergpqa(),
    "supergpqa_en": supergpqa_en(),
    "include": include(),
    "mmmlu": mmmlu(),
}

if __name__ == "__main__":

    # Dei add nisso para verificar a estrutura de datasets sem ter que rodar alguma coisa
    # Ex: python src/UTILS_BENCHMARKS.py assin2rte assin2sts bluex enem hatebr portuguese_hate_speech toxsyn_pt faquad tweetsentbr oab poscomp energy_regulacao aime24 aime25 mmlu mmlu_en mmlu_hard mmlu_redux_en mmlu_pro_en supergpqa supergpqa_en include mmmlu
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description='Inspect benchmark datasets and test prompt structures')
    parser.add_argument(
        'datasets',
        nargs='*',
        default=['aime25', 'mmlu'],
        help='Dataset names to inspect (default: aime25 mmlu)'
    )

    args = parser.parse_args()

    for dataset_name in args.datasets:
        if dataset_name not in BENCHMARKS_INFORMATIONS:
            print(f"Dataset {dataset_name} not found")
            continue

        benchmark = BENCHMARKS_INFORMATIONS[dataset_name]
        print(f"\n================= {dataset_name.upper()} =================")
        print(f"Path: {benchmark.dataset_path}")
        print(f"Subset: {benchmark.subset}")
        print(f"Split: {benchmark.split}")
        print(f"Label column: {benchmark.label_column}")
        print(f"Important columns: {benchmark.important_columns}")

        try:
            subset = benchmark.subset
            if isinstance(subset, (list, tuple)):
                subset = subset[0]
            dataset = load_dataset(benchmark.dataset_path, subset)[benchmark.split]
            if len(dataset) > 0:
                real_sample = dataset[0]
                if benchmark.label_column != "label":
                    real_sample = {k: v for k, v in real_sample.items()}
                    if benchmark.label_column in real_sample:
                        real_sample["label"] = real_sample.pop(benchmark.label_column)
                prompt_info = benchmark.get_prompt_informations(real_sample)

                print("System Prompt:")
                print(prompt_info['base_system_message'])
                print("\nUser Message:")
                print(prompt_info['user_message'])
                print("\nAssistant Message (with answer):")
                print(prompt_info['assistant_message_with_answer'])
                print("\nAssistant Message (without answer):")
                print(prompt_info['assistant_message_without_answer'])
                print("\n")

        except Exception as e:
            print(f"Error inspecting {dataset_name}: {e}")
