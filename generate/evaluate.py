from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = load_dataset(f"pt-eval/evaluate_5shot_3exp", split='train')

df = dataset.to_pandas()
df = df.dropna()



true_labels = df['label']
predicted_labels = df['model_answer']

# Calculate Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

class_report = classification_report(true_labels, predicted_labels, labels=['A', 'B', 'C', 'D', 'E'])
print('Classification Report:')
print(class_report)