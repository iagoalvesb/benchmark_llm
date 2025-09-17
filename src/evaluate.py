from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.stats import pearsonr
from huggingface_hub import list_datasets
import argparse
import os
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
from utils import BENCHMARK_TO_METRIC, clean_index_columns, parse_integer_math_format
import logging
from logger_config import init_logger


parser = argparse.ArgumentParser()

parser.add_argument(
    "--answers_path",
    type=str,
    required=True,
    help="Dataset path with the label and the model answers"
)

parser.add_argument(
    "--eval_path",
    type=str,
    required=True,
    help="Huggingface path to save final metrics"
)

parser.add_argument(
    "--run_local",
    action="store_true",
    help="If set, read/save results locally as CSV instead of using HuggingFace Hub"
)

args = parser.parse_args()

init_logger()

if args.run_local:
    eval_filename = args.eval_path.replace("/", "_").replace("-", "_") + ".csv"
    eval_filepath = os.path.join("eval_processing", eval_filename)
    eval_dataset_exists = os.path.exists(eval_filepath)
    
    if eval_dataset_exists:
        eval_df = pd.read_csv(eval_filepath)
        eval_df = clean_index_columns(eval_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        logging.info(f"Loaded existing evaluation data from: {eval_filepath}")

else:
    eval_dataset = list_datasets(search=args.eval_path)
    eval_dataset_exists = any(ds.id == args.eval_path for ds in eval_dataset)
    if eval_dataset_exists:
        eval_dataset = load_dataset(args.eval_path, split='train')
        logging.info(f"Loaded existing evaluation data from HuggingFace Hub: {args.eval_path}")

if args.run_local:
    answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
    answers_filepath = os.path.join("eval_processing", answers_filename)
    
    if not os.path.exists(answers_filepath):
        raise FileNotFoundError(f"Answers file not found: {answers_filepath}")
    
    df = pd.read_csv(answers_filepath)
    df = clean_index_columns(df)
    dataset = Dataset.from_pandas(df)
    logging.info(f"Loaded answers from local file: {answers_filepath}")
else:
    dataset = load_dataset(args.answers_path, split='train')
    logging.info(f"Loaded answers from HuggingFace Hub: {args.answers_path}")


df = dataset.to_pandas()
metrics_list = []

for model_name in df['model_name'].unique():
    df_model = df[df['model_name'] == model_name]
    # para não ter duplicatas, vamos remover os modelos que rodamos o eval novamente no dataset de eval
    if eval_dataset_exists:
        eval_dataset = eval_dataset.filter(lambda x: x['model_name'] != model_name)

    for benchmark_name in df_model['benchmark'].unique():
        df_model_benchmark = df_model[df_model['benchmark'] == benchmark_name]
        y_true = df_model_benchmark['label'].values
        y_pred = df_model_benchmark['parsed_model_answer'].values

        
        # Remove None values from y_true and y_pred
        mask = [True if p is not None else False for p in y_pred]
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        non_parsed_pct = mask.count(False) / len(mask)

        benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]

        # Normalização específica para benchmarks de exato inteiro com formatação matemática (AIME 24/25)
        if getattr(benchmark, 'answer_pattern', None) == "integer_exact_math":
            y_true_list = [parse_integer_math_format(str(x)) for x in y_true]
            y_pred_list = list(y_pred)
            # Refiltrar removendo casos inválidos em y_true OU y_pred
            mask_both = [(p is not None) and (t is not None) for p, t in zip(y_pred_list, y_true_list)]
            y_pred = [p for p, m in zip(y_pred_list, mask_both) if m]
            y_true = [t for t, m in zip(y_true_list, mask_both) if m]
            if len(mask_both) > 0:
                non_parsed_pct = 1 - (sum(mask_both) / len(mask_both))
        
        # Calculate various metrics
        if benchmark.answer_type == "category":
            # Garantir tipos homogêneos (strings) para classificadores
            y_true = [str(x) for x in y_true]
            y_pred = [str(x) for x in y_pred]
            accuracy  = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            corr = None
        elif benchmark.answer_type == "continue":
            y_true = [float(x) for x in y_true]
            y_pred = [float(x) for x in y_pred]
            accuracy  = None
            precision = None
            recall    = None
            f1        = None
            corr, p_value = pearsonr(y_true, y_pred)

        # Store results in a list of dicts
        metrics_list.append({
            'model_name': model_name,
            'benchmark': benchmark_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pearson_correlation': corr,
            'non_parsed_rate': non_parsed_pct,
        })

# Convert list of dicts to a DataFrame for easy viewing
results_df = pd.DataFrame(metrics_list)
print(results_df)
results_dataset = Dataset.from_pandas(results_df)
if eval_dataset_exists:
    # remove the models that are already in the eval dataset
    results_dataset = concatenate_datasets([eval_dataset, results_dataset])

if args.run_local:
    os.makedirs("eval_processing", exist_ok=True)
    eval_filename = args.eval_path.replace("/", "_").replace("-", "_") + ".csv"
    eval_filepath = os.path.join("eval_processing", eval_filename)
    
    final_df = results_dataset.to_pandas()
    final_df.to_csv(eval_filepath, index=False)
    logging.info(f"\n\n** EVALUATION RESULTS SAVED AT: {eval_filepath}")
else:
    results_dataset.push_to_hub(args.eval_path)
    logging.info(f"\n\n** EVALUATION RESULTS SAVED AT: {args.eval_path}")

# Criar print no CLI dos resultados (forma simples)
metrics_data = []
parsing_data = []

for model_name in results_df['model_name'].unique():
    model_metrics = {'Model': model_name}
    model_parsing = {'Model': model_name}
    model_df = results_df[results_df['model_name'] == model_name]

    for _, row in model_df.iterrows():
        benchmark = row['benchmark']
        metric_name = BENCHMARK_TO_METRIC.get(benchmark, ['accuracy'])[0]

        metric_value = row[metric_name] if row[metric_name] is not None else 0.0
        model_metrics[benchmark] = round(metric_value, 3)
        model_parsing[benchmark] = round(row['non_parsed_rate'], 3)

    metrics_data.append(model_metrics)
    parsing_data.append(model_parsing)

metrics_df = pd.DataFrame(metrics_data)
parsing_df = pd.DataFrame(parsing_data)

benchmark_columns = [col for col in metrics_df.columns if col != 'Model']
metrics_df['avg'] = metrics_df[benchmark_columns].mean(axis=1).round(3)

with pd.option_context('display.max_columns', None, 'display.width', None):
    print("\n" + "="*80)
    print("** METRICAS DE PERFORMANCE **")
    print("="*80)
    print(metrics_df)
    print("\n" + "="*80)
    print("** TAXA DE ERRO DE PARSING **")
    print("="*80)
    print(parsing_df)
    print("="*80)
