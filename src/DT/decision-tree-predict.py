import os
import pandas as pd
import joblib
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

# paths
MODEL_DIR = 'DecisionTree-model'
RESULTS_DIR = 'data/results/Eval-DT'
CLASSIFIED_DIR = 'data/corpus_class_DT'
TEST_FILE = 'cleaned_dataset_test.csv'

# load model
model = joblib.load(os.path.join(MODEL_DIR, 'decision_tree_model.joblib'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))

# load data
df = pd.read_csv(TEST_FILE, sep=';')
if 'Review' not in df.columns:
    raise ValueError("Missing 'Review' column in test file.")

df['Review'] = df['Review'].astype(str).str.strip()
X_test_raw = df['Review']
X_test = vectorizer.transform(X_test_raw)
y_pred = model.predict(X_test)
labels_pred = label_encoder.inverse_transform(y_pred)

# save predictions
df['Prediction'] = labels_pred
df.to_csv('decision_tree_predictions.csv', index=False)
print("Predictions saved to 'decision_tree_predictions.csv'")

# evaluate model
if 'Category' in df.columns:
    df['Category'] = df['Category'].str.strip()
    y_true = label_encoder.transform(df['Category'])

    report_dict = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    # Save text report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'decision_tree_results.txt'), 'w') as f:
        f.write("=== Confusion Matrix ===\n")
        f.write(str(matrix) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Save CSV report
    with open(os.path.join(RESULTS_DIR, 'decision_tree_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        for cls in label_encoder.classes_:
            scores = report_dict[cls]
            writer.writerow([
                cls,
                f"{scores['precision']:.2f}",
                f"{scores['recall']:.2f}",
                f"{scores['f1-score']:.2f}",
                int(scores['support'])
            ])
    print("Evaluation reports saved in:", RESULTS_DIR)

############## Plot confusion matrix#########################
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()
##############################################################

# save classified reviews in folders
os.makedirs(CLASSIFIED_DIR, exist_ok=True)
for label in label_encoder.classes_:
    os.makedirs(os.path.join(CLASSIFIED_DIR, label.lower()), exist_ok=True)

for i, (text, label) in enumerate(zip(X_test_raw, labels_pred)):
    folder = os.path.join(CLASSIFIED_DIR, label.lower())
    path = os.path.join(folder, f"review_{i+1}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

print(f"Reviews sorted into subfolders under: {CLASSIFIED_DIR}")
