import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import matplotlib.pyplot as plt

df = pd.read_csv("parsed_answer.csv")
df = df[df['label'].isin(['A', 'B', 'C', 'D', 'E', 'a', 'b', 'c', 'd', 'e'])]  # Filter for valid labels
df = df[df['answer_parsed'].isin(['A', 'B', 'C', 'D', 'E', 'a', 'b', 'c', 'd', 'e'])]  # Filter for valid labels


models = df['model'].unique()
metrics_list = []

for model_name in models:
    # Subset for this particular model
    df_model = df[df['model'] == model_name]
    
    # Ground truth (y_true) and predictions (y_pred)
    y_true = df_model['label']
    y_pred = df_model['answer_parsed']
    
    # Calculate various metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Store results in a list of dicts
    metrics_list.append({
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Convert list of dicts to a DataFrame for easy viewing
results_df = pd.DataFrame(metrics_list)

models = ['0.5b-traduzido', '0.5b-sintetico', '1.5b-traduzido', '1.5b-sintetico', '3b-sintetico', '3b-traduzido']
plt.figure(figsize=(10, 6))
plt.title("Model Performance Comparison - F1 Score")
plt.bar(models, results_df['f1_score'], color='blue', label='Accuracy')
plt.savefig("f1_plot.png")