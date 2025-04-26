from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.stats import pearsonr

import argparse
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS


parser = argparse.ArgumentParser()

parser.add_argument(
    "--answers_path",
    type=str,
    required=True,
    help="Dataset path with the label and the model answers"
)

parser.add_argument(
    "--output_evaluation_path",
    type=str,
    required=True,
    help="Huggingface path to save final metrics"
)

args = parser.parse_args()

dataset = load_dataset(args.answers_path, split='train')
df = dataset.to_pandas()

metrics_list = []

for model_name in df['model_name'].unique():
    df_model = df[df['model_name'] == model_name]
    for benchmark_name in df_model['benchmark'].unique():
        df_model_benchmark = df_model[df_model['benchmark'] == benchmark_name]
        y_true = df_model_benchmark['label'].values
        y_pred = df_model_benchmark['parsed_model_answer'].values

        
        # Remove None values from y_true and y_pred
        mask = [True if p is not None else False for p in y_pred]
        y_pred = y_pred[mask]
        non_parsed_pct = mask.count(False) / len(mask)

        benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
        
        # Calculate various metrics
        if benchmark.answer_type == "category":
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
            'non_parsed': non_parsed_pct,
        })

# Convert list of dicts to a DataFrame for easy viewing
results_df = pd.DataFrame(metrics_list)
print(results_df)
results_dataset = Dataset.from_pandas(results_df)
results_dataset.push_to_hub(args.output_evaluation_path)

print(f"\n\n** EVALUATION RESULTS SAVED AT: {args.output_evaluation_path}")