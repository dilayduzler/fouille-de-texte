import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TRAIN_DATASET = 'cleaned_dataset.csv'
MODEL_DIR = 'DecisionTree-model'
os.makedirs(MODEL_DIR, exist_ok=True)

# load data 
df = pd.read_csv(TRAIN_DATASET, sep=';')
df.drop(columns=[col for col in df.columns if "ID" in col], inplace=True, errors='ignore')
df['Category'] = df['Category'].str.strip()

X_raw = df['Review']
y_raw = df['Category']

custom_stopwords = ["game", 'way', 'play', 'people', 'games', "the", "everyone"]
combined_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

# tf-idf vectorization
vectorizer = TfidfVectorizer(
    #stop_words=combined_stopwords,
    stop_words=custom_stopwords,
    max_features=500,
    ngram_range=(1, 1),
    #max_df=0.9,
    token_pattern=r"(?u)\b\w+\b|[!?]" # include ! and ? in tokens
)
X = vectorizer.fit_transform(X_raw)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# train model
model = DecisionTreeClassifier(
    criterion='entropy',
    #criterion='gini',
    max_depth=7, 
    min_samples_leaf=3,
    class_weight='balanced',
    ccp_alpha=0.03,
    #random_state=42
)
model.fit(X, y)

# evaluate model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=skf)

print("\n=== Classification Report ===")
print(f"Accuracy : {accuracy_score(y, y_pred)*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")

label_order = np.arange(len(label_encoder.classes_))

# show a few misclassified reviews
df["true_label"] = y
predicted_labels = y_pred
df["predicted_label"] = predicted_labels
misclassified = df[df["true_label"] != df["predicted_label"]].sample(n=5, random_state=42)

#print("\nExamples:")
#for idx, row in misclassified.iterrows():
#    true_cls = label_encoder.inverse_transform([row["true_label"]])[0]
#    pred_cls = label_encoder.inverse_transform([row["predicted_label"]])[0]
#    print(f"\nReview: {row['Review']}\n  True: {true_cls} | Predicted: {pred_cls}")

# Confusion matrix
cm = confusion_matrix(y, y_pred, labels=label_order)

print("Confusion Matrix:")
print(pd.DataFrame(cm, index=[f"True: {cls}" for cls in label_encoder.classes_], columns=[f"Pred: {cls}" for cls in label_encoder.classes_]))

####################### Displays #######################

# CONFUSION MATRIX
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# IMPORTANCES
importances = model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
nonzero_indices = np.where(importances > 0)[0]
top_features = sorted(
    [(feature_names[i], importances[i]) for i in nonzero_indices],
    key=lambda x: x[1],
    reverse=True
)[:20] 
df_features = pd.DataFrame(top_features, columns=["Word", "Importance"])
plt.figure(figsize=(10, 6))
sns.barplot(data=df_features, y="Word", x="Importance", palette="magma")
plt.title("Top 20 Most Important Words in Decision Tree", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

##########################################################

# save model
joblib.dump(model, os.path.join(MODEL_DIR, 'decision_tree_model.joblib'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.joblib'))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

