import argparse
import pandas as pd
import warnings
from huggingface_hub import HfApi, list_datasets
from datasets import load_dataset, Dataset, concatenate_datasets
import logging
import os
from logger_config import init_logger
from utils import BENCHMARK_TO_AREA, BENCHMARK_TO_COLUMN, BENCHMARK_TO_METRIC, MODEL_PARAMS, add_additional_info, clean_index_columns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks-file', required=True,
                        help="HuggingFace dataset path containing benchmark results")
    parser.add_argument('--output-repo', required=True,
                        help="HuggingFace repo name to save the leaderboard")
    parser.add_argument('--exclude-models', nargs='+', default=[],
                        help="List of model names to exclude from the leaderboard")
    parser.add_argument('--custom-flags', nargs='+', default=[],
                        help="Custom flags for each model (true/false)")
    parser.add_argument('--model-paths', nargs='+', default=[],
                        help="Model paths corresponding to custom flags")
    parser.add_argument('--overwrite', action='store_true',
                        help="If True, overwrite existing model results")
    parser.add_argument('--save-csv', action='store_true',
                        help="If set, save the final results to a CSV file")
    parser.add_argument('--run_local', action='store_true',
                        help="If set, read/save results locally as CSV instead of using HuggingFace Hub")
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
    init_logger()

    if args.run_local:
        benchmarks_filename = args.benchmarks_file.replace("/", "_").replace("-", "_") + ".csv"
        benchmarks_filepath = os.path.join("eval_processing", benchmarks_filename)
        
        if not os.path.exists(benchmarks_filepath):
            raise FileNotFoundError(f"Benchmarks file not found: {benchmarks_filepath}")
        
        df = pd.read_csv(benchmarks_filepath)
        df = clean_index_columns(df)
        logging.info(f"Loaded benchmarks from local file: {benchmarks_filepath}")
    else:
        try:
            dataset = load_dataset(args.benchmarks_file, split='train')
            df = dataset.to_pandas()
            logging.info(f"Loaded benchmarks from HuggingFace Hub: {args.benchmarks_file}")
        except:
            df = pd.read_csv(args.benchmarks_file)
            df = clean_index_columns(df)
            logging.info(f"Loaded benchmarks from CSV file: {args.benchmarks_file}")

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
    models_to_skip = set()
    actually_processed = []
    if args.run_local:
        leaderboard_filename = args.output_repo.replace("/", "_").replace("-", "_") + ".csv"
        leaderboard_filepath = os.path.join("leaderboard_results", leaderboard_filename)
        dataset_exists = os.path.exists(leaderboard_filepath)
        
        if dataset_exists and not args.overwrite:
            existing_df = pd.read_csv(leaderboard_filepath)
            existing_df = clean_index_columns(existing_df)
            models_to_skip = set(existing_df['Modelo'])
            logging.info(f"Found existing leaderboard data: {leaderboard_filepath}")
    else:
        possible_datasets = list_datasets(search=args.output_repo)
        dataset_exists = any(ds.id == args.output_repo for ds in possible_datasets)

        if dataset_exists and not args.overwrite:
            existing_dataset = load_dataset(args.output_repo, split='train')
            models_to_skip = set(existing_dataset['Modelo'])
            logging.info(f"Found existing leaderboard data on HuggingFace Hub: {args.output_repo}")

    custom_flag_mapping = {}
    if args.custom_flags and args.model_paths:
        if len(args.custom_flags) == len(args.model_paths):
            for model_path, custom_flag in zip(args.model_paths, args.custom_flags):
                custom_flag_mapping[model_path] = custom_flag.lower() == 'true'
        else:
            warnings.warn(f"Custom flags count ({len(args.custom_flags)}) doesn't match model paths count ({len(args.model_paths)})")

    skipped_existing = []
    skipped_not_run = []
    for model_name in model_names:
        if model_name in models_to_skip:
            skipped_existing.append(model_name)
            logging.info(f"Pulando o modelo {model_name} (já existe, overwrite está como False)")
            continue

        model_params = MODEL_PARAMS.get(model_name, {})
        if 'model_id' not in model_params:
            skipped_not_run.append(model_name)
            logging.info(f"Modelo {model_name} não foi executado nesta rodada, pulando processamento de metadados")
            continue

        actually_processed.append(model_name)  # Track what we actually process
        logging.info(f"Processando o modelo {model_name}")
        model_df = df[df['model_name'] == model_name]
        model_params = MODEL_PARAMS.get(model_name, {})

        modelo = model_name
        sha_modelo = license = determined_tipo = "unknown"
        hub_likes = 0
        disponivel = False
        is_adapter = "Original"
        arquiteturas = []
        precisao = None
        params_B = 0
        t = "SFT"

        try:
            hf_model_id = model_params['model_id']
            info = api.model_info(hf_model_id)
        except Exception as e:
            warnings.warn(f"Erro ao obter informações do modelo {model_name}: {e}")

        def safe_get(getter_func, default=None):
            try:
                return getter_func()
            except Exception:
                return default

        modelo = safe_get(lambda: info.modelId, modelo)
        sha_modelo = safe_get(lambda: info.sha, sha_modelo)
        hub_likes = safe_get(lambda: info.likes, hub_likes)
        disponivel = safe_get(lambda: not info.private, disponivel)
        license = safe_get(lambda: info.card_data.get('license'), license)

        cfg = safe_get(lambda: info.config, {}) or {}
        st = safe_get(lambda: info.safetensors, {}) or {}

        arquiteturas = safe_get(lambda: cfg.get('architectures', []), arquiteturas)
        precisao = safe_get(lambda: next(iter(st.get('parameters', {})), None), precisao)
        params_B = safe_get(lambda: st.get('total', 0) / 1e9, params_B)

        is_adapter = "Original"
        if safe_get(lambda: info.config and (info.config.get('adapter_type') or info.config.get('peft_config')), False):
            is_adapter = "Adapter"

        is_custom_model = custom_flag_mapping.get(model_name, False)

        if is_custom_model:
            t = "Custom"
            determined_tipo = "Custom"
        else:
            determined_tipo = t = "Base"
            if safe_get(lambda: info and info.card_data and info.card_data.get('base_model'), False):
                base_spec = info.card_data.get('base_model')
                actual_base_id = safe_get(lambda: (base_spec[0] if isinstance(base_spec, list) and base_spec else base_spec) if isinstance(base_spec, (list, str)) else None, None)
                if actual_base_id and actual_base_id != info.modelId:
                    t = "SFT"
                    determined_tipo = "SFT: Supervised Finetuning"

        out = {
            'T': t,
            'Modelo': modelo,
            'Tipo': determined_tipo,
            'Arquitetura': ','.join(arquiteturas) if arquiteturas else None,
            'Tipo de Peso': is_adapter,
            'Precisão': precisao,
            'Licença': license,
            '#Params (B)': round(params_B, 3) if params_B else 0,
            'Hub Likes': hub_likes,
            'Disponível no hub': disponivel,
            'SHA do modelo': sha_modelo,
        }

        # Adicionado para casos onde você quer sobrescrever os valores de 'out' com os valores hardcoded
        override_mapping = {
            't': 'T',
            'modelo': 'Modelo',
            'tipo': 'Tipo',
            'arquitetura': 'Arquitetura',
            'tipo_peso': 'Tipo de Peso',
            'precisao': 'Precisão',
            'licenca': 'Licença',
            'params_b': '#Params (B)',
            'hub_likes': 'Hub Likes',
            'disponivel_no_hub': 'Disponível no hub',
            'sha_modelo': 'SHA do modelo',
        }

        for param_key, out_key in override_mapping.items():
            if param_key in model_params:
                if param_key in ['t', 'tipo'] and is_custom_model:
                    continue
                
                value = model_params[param_key]
                if param_key == 'arquitetura' and isinstance(value, list):
                    out[out_key] = ','.join(value)
                else:
                    out[out_key] = value

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

    # if args.save_csv:
    #     output_csv_path = "leaderboard_results_test.csv"
    #     results_df.to_csv(output_csv_path, index=False)

    new_models_count = len(actually_processed)
    existing_models_count = len(skipped_existing) + len(skipped_not_run)

    if args.run_local:
        os.makedirs("leaderboard_results", exist_ok=True)
        leaderboard_filename = args.output_repo.replace("/", "_").replace("-", "_") + ".csv"
        leaderboard_filepath = os.path.join("leaderboard_results", leaderboard_filename)
        
        if dataset_exists:
            existing_df = pd.read_csv(leaderboard_filepath)
            existing_df = clean_index_columns(existing_df)
            
            if args.overwrite:
                new_model_names = set(results_df['Modelo'])
                print(new_model_names)
                print(existing_df)
                filtered_df = existing_df[~existing_df['Modelo'].isin(new_model_names)]
                combined_df = pd.concat([filtered_df, results_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['Modelo'], keep='last')
            else:
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            
            combined_df.to_csv(leaderboard_filepath, index=False)
            logging.info(f"Atualizou resultados com {new_models_count} modelos novos e {existing_models_count} modelos antigos em: {leaderboard_filepath}")
        else:
            results_df.to_csv(leaderboard_filepath, index=False)
            logging.info(f"Criou leaderboard local com {len(results_df)} modelos em: {leaderboard_filepath}")
    else:
        if dataset_exists:
            existing_dataset = load_dataset(args.output_repo, split='train')

            if args.overwrite:
                new_model_names = set(results_df['Modelo'])
                existing_df = existing_dataset.to_pandas()
                filtered_df = existing_df[~existing_df['Modelo'].isin(new_model_names)]

                combined_df = pd.concat([filtered_df, results_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['Modelo'], keep='last')
                combined_dataset = Dataset.from_pandas(combined_df)
            else:
                combined_dataset = concatenate_datasets([existing_dataset, Dataset.from_pandas(results_df)])

            combined_dataset.push_to_hub(args.output_repo)
            logging.info(f"Atualizou resultados com {new_models_count} modelos novos e {existing_models_count} modelos antigos")
        else:
            new_dataset = Dataset.from_pandas(results_df)
            new_dataset.push_to_hub(args.output_repo)
            logging.info(f"Repo criado com {len(results_df)} modelos")

