from datasets import load_dataset, concatenate_datasets, Dataset, Value

datasets_path = {"bluex": "mestras-valcir/bluex_5shot_3exp",
                  "enem": "mestras-valcir/enem_5shot_3exp",
                  "assin2rte": "mestras-valcir/assin2rte_5shot_3exp",
                  "assin2sts": "mestras-valcir/assin2sts_5shot_3exp",
                  "hatebr": "mestras-valcir/hatebr_5shot_3exp",
                  "phs": "mestras-valcir/phs_5shot_3exp",
                  }


datasets_loaded = {name: load_dataset(path, split='train') for name, path in datasets_path.items()}
datasets_processed = []

for name, dataset in datasets_loaded.items():
    if name in ["bluex", "enem"]:
        label_column = "answerKey"
        dataset = dataset.rename_column("answerKey", "label")
    columns_to_maintain = ['label', "prompt_0", "prompt_1", "prompt_2"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_maintain]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.add_column("benchmark", [name] * len(dataset))
    dataset = dataset.add_column("id_bench", [f"{name}_{i}" for i in range(len(dataset))])
    dataset = dataset.cast_column("label", Value("string"))
    datasets_processed.append(dataset)

full_dataset = concatenate_datasets(datasets_processed)

df = full_dataset.to_pandas()

id_vars = [col for col in df.columns if col not in ['prompt_0', 'prompt_1', 'prompt_2']]
df = df.melt(
    id_vars=id_vars,  # Columns to keep
    value_vars=['prompt_0', 'prompt_1', 'prompt_2'],  # Columns to unpivot
    var_name='prompt_type',   # Name for the new column indicating the prompt column
    value_name='prompt'       # Name for the new column that holds the prompt text
)

df = df.drop(columns='prompt_type')

df['id'] = list(range(len(df)))
column_order = ['id', 'id_bench', 'benchmark', 'prompt', 'label']
df = df[column_order]
full_dataset = Dataset.from_pandas(df)

print(full_dataset)

full_dataset.push_to_hub("mestras-valcir/merged_5shot_3exp_v2")