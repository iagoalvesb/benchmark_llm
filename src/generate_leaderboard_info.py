import argparse
import pandas as pd
import warnings
from huggingface_hub import HfApi, list_datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import BENCHMARK_TO_AREA, BENCHMARK_TO_COLUMN, BENCHMARK_TO_METRIC, MODEL_PARAMS, add_additional_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks-file', required=True,
                        help="HuggingFace dataset path containing benchmark results")
    parser.add_argument('--output-repo', required=True,
                        help="HuggingFace repo name to save the leaderboard")
    parser.add_argument('--exclude-models', nargs='+', default=[],
                        help="List of model names to exclude from the leaderboard")
    parser.add_argument('--overwrite', action='store_true',
                        help="If True, overwrite existing model results")
    parser.add_argument('--save-csv', action='store_true',
                        help="If set, save the final results to a CSV file")
    return parser.parse_args()

def compute_score(row, benchmark):
    metrics = BENCHMARK_TO_METRIC.get(benchmark, ['accuracy'])

    for metric in metrics:
        if metric not in row or pd.isna(row[metric]):
            warnings.warn(f"Metric '{metric}' not found or is NULL for benchmark '{benchmark}'")

    vals = [row.get(m, None) for m in metrics]
    valid = [v for v in vals if pd.notnull(v)]
    if not valid:
        return 0.0
    return float(valid[0] if len(valid) == 1 else sum(valid) / len(valid))

if __name__ == "__main__":
    args = parse_args()

    try:
        dataset = load_dataset(args.benchmarks_file, split='train')
        df = dataset.to_pandas()
    except:
        df = pd.read_csv(args.benchmarks_file)

    df['score'] = df.apply(lambda r: compute_score(r, r['benchmark']), axis=1)

    if args.exclude_models:
        df = df[~df['model_name'].isin(args.exclude_models)]

    benchmark_exists = {}
    for bench in BENCHMARK_TO_AREA.keys():
        has_scores = False
        for _, row in df.iterrows():
            if row['benchmark'] == bench and row['score'] > 0:
                has_scores = True
                break
        benchmark_exists[bench] = has_scores

    all_results = []
    model_names = df['model_name'].unique()

    api = HfApi()
    possible_datasets = list_datasets(search=args.output_repo)
    dataset_exists = any(ds.id == args.output_repo for ds in possible_datasets)

    models_to_skip = set()
    if dataset_exists and not args.overwrite:
        existing_dataset = load_dataset(args.output_repo, split='train')
        models_to_skip = set(existing_dataset['Modelo'])

    for model_name in model_names:
        if model_name in models_to_skip:
            print(f"Pulando o modelo {model_name} (já existe, overwrite está como False)") # pula modelo se já existe e overwrite tá setado como falso
            continue

        print(f"Processando o modelo {model_name}")
        model_df = df[df['model_name'] == model_name]
        model_params = MODEL_PARAMS[model_name]

        try:
            hf_model_id = model_params['model_id']
            info = api.model_info(hf_model_id)
            cfg = info.config or {}
            st = info.safetensors or {}

            modelo = info.modelId
            sha_modelo = info.sha
            hub_likes = info.likes
            disponivel = not info.private
            arquiteturas = cfg.get('architectures', [])
            precisao = next(iter(st.get('parameters', {})), None)
            params_B = st.get('total', 0) / 1e9
        except Exception as e:
            warnings.warn(f"Modelo não encontrado no HF: {e}")
            modelo = model_name
            sha_modelo = "unknown"
            hub_likes = 0
            disponivel = False
            arquiteturas = []
            precisao = None
            params_B = 0

        out = {
            'T': model_params.get('t', 'n/a'),
            'Modelo': modelo,
            'Tipo': model_params.get('tipo', 'n/a'),
            'Arquitetura': ','.join(arquiteturas) if arquiteturas else None,
            'Tipo de Peso': model_params.get('tipo_peso', 'Original'),
            'Precisão': precisao,
            'Licença': model_params.get('licenca', 'n/a'),
            '#Params (B)': round(params_B, 3) if params_B else 0,
            'Hub Likes': hub_likes,
            'Disponível no hub': disponivel,
            'SHA do modelo': sha_modelo,
        }

        area_scores = {area: [] for area in set(BENCHMARK_TO_AREA.values())}
        for _, row in model_df.iterrows():
            bench = row['benchmark']
            sc = row['score']
            area = BENCHMARK_TO_AREA.get(bench)
            if area and benchmark_exists[bench]:
                area_scores[area].append(sc)

        for area, scores in area_scores.items():
            out[f'{area}'] = (sum(scores)/len(scores)) if scores else 0.0

        for bench, col in BENCHMARK_TO_COLUMN.items():
            matched = model_df.loc[model_df['benchmark'] == bench, 'score']
            out[col] = float(matched.iloc[0]) if len(matched) > 0 else 0.0

        benchmark_values = [out[col] for bench, col in BENCHMARK_TO_COLUMN.items() 
                           if benchmark_exists[bench]]
        out['Média Geral'] = sum(benchmark_values)/len(benchmark_values) if benchmark_values else 0.0

        all_results.append(out)

    results_df = pd.DataFrame(all_results)
    results_df = add_additional_info(results_df) # Lucas pediu para ter isso aqui

    if args.save_csv:
        output_csv_path = "leaderboard_results.csv"
        results_df.to_csv(output_csv_path, index=False)

    if dataset_exists:
        existing_dataset = load_dataset(args.output_repo, split='train')

        if args.overwrite:
            new_model_names = set(results_df['Modelo'])
            existing_df = existing_dataset.to_pandas()
            filtered_df = existing_df[~existing_df['Modelo'].isin(new_model_names)]

            combined_df = pd.concat([filtered_df, results_df], ignore_index=True)
            combined_dataset = Dataset.from_pandas(combined_df)
        else:
            combined_dataset = concatenate_datasets([existing_dataset, Dataset.from_pandas(results_df)])

        combined_dataset.push_to_hub(args.output_repo)
        print(f"Atualizou resultados com {len(results_df)} modelos novos")
    else:
        new_dataset = Dataset.from_pandas(results_df)
        new_dataset.push_to_hub(args.output_repo)
        print(f"Repo criado com {len(results_df)} modelos")