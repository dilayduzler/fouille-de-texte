
#######################################################

# Import

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import csv


######################################################

# Chargin model & data

### Charging trained model ###
model = joblib.load('SVC-model/svm_model.joblib')
vectorizer = joblib.load('SVC-model/vectorizer.joblib')
label_encoder = joblib.load('SVC-model/label_encoder.joblib')

### charging test corpus ###
try:
    test_df = pd.read_csv('cleaned_dataset_test.csv', sep=';')
    if 'Review' not in test_df.columns:
        raise ValueError("Le fichier 'test.csv' doit contenir une colonne 'Review'.")
    X_test_raw = test_df['Review']
        ### Apply model ###
    X_test = vectorizer.transform(X_test_raw)
    y_test_pred = model.predict(X_test)

    # === 4. Décodage des prédictions en labels textuels ===
    labels_pred = label_encoder.inverse_transform(y_test_pred)



 ####################################################################

# Save predictions

    ### Save in CSV doc ###
    test_df['Prediction'] = labels_pred
    test_df.to_csv('test_resultats.csv', index=False)

    print("✅ Prédictions sauvegardées dans 'test_resultats.csv'")



 ####################################################################

 # Report

    ### testing data
    evaluation_txt = ""
    evaluation_csv = []

    if 'Category' in test_df.columns:
        test_df['Category'] = test_df['Category'].str.strip()
        y_test_true = label_encoder.transform(test_df['Category'])

        report = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_, output_dict=True)
        matrix = confusion_matrix(y_test_true, y_test_pred)
    
        # Rapport texte
        evaluation_txt += "=== Matrice de confusion ===\n"
        evaluation_txt += str(matrix) + "\n\n"
        evaluation_txt += "=== Rapport de classification ===\n"
        evaluation_txt += classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_)

        # Rapport CSV
        evaluation_csv.append(["Classe", "Precision", "Recall", "F1-score", "Support"])
        for cls in label_encoder.classes_:
            scores = report[cls]
            evaluation_csv.append([
                cls,
                f"{scores['precision']:.2f}",
                f"{scores['recall']:.2f}",
                f"{scores['f1-score']:.2f}",
                f"{int(scores['support'])}"
            ])

        # === Sauvegarde du rapport d'évaluation dans des fichiers ===
        evaluation_path_txt = os.path.join("data/results/Eval-SVC/SVC-results.txt")
        evaluation_path_csv = os.path.join("data/results/Eval-SVC/SVC-results.csv")

        if evaluation_txt:
            # Sauvegarde du rapport au format texte
            with open(evaluation_path_txt, 'w', encoding='utf-8') as f:
                f.write(evaluation_txt)
                print(f"📄 Rapport texte enregistré dans : {evaluation_path_txt}")

        if evaluation_csv:
            # Sauvegarde du rapport au format CSV
            with open(evaluation_path_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(evaluation_csv)
                print(f"📊 Rapport CSV enregistré dans : {evaluation_path_csv}")

    
except Exception as e:
    print(f"Erreur lors du chargement de 'cleaned_dataset_test.csv' : {e}")
    exit()



###########################################################

# Classification

### create output dir ### 
output_base = 'data/corpus_class_SVC'
os.makedirs(output_base, exist_ok=True)
for label in ['bug', 'feature', 'feedback']:
    os.makedirs(os.path.join(output_base, label), exist_ok=True)

### save review in dir ###
for i, (text, label) in enumerate(zip(X_test_raw, labels_pred)):
    folder = label.lower()  # ex: "Bug" -> "bug"
    filename = f"review_{i+1}.txt"
    path = os.path.join(output_base, folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

print(f"✅ Reviews classées dans : {output_base}")

