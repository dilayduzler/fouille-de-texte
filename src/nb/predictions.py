
###########################################################

 #Import

import pandas as pd
import joblib
from sklearn.metrics import classification_report


#############################################################

# Parameter and charging model

### Parameters ###
MODEL_PATH = "naive_bayes_model.joblib"         # Trained model
TEST_CSV_PATH = "data/intermediate/cleaned_dataset_test.csv"    # Test corpus
OUTPUT_PATH = "test_resultats.csv"          # Output file 
OUTPUT_DIR = "data/corpus_class_nb"          # Output dir

### Charging trained model ###
print("Chargement du modèle...")
model = joblib.load(MODEL_PATH)

### Charging test corpus ###
print(f"Chargement du fichier test : {TEST_CSV_PATH}")
df_test = pd.read_csv(TEST_CSV_PATH)



###########################################################

### Verif ###
if 'Review' not in df_test.columns:
    raise ValueError("La colonne 'Review' est requise dans le fichier test.")

### Predictions ###
print("✍️ Prédiction en cours...")
df_test['Prediction'] = model.predict(df_test['Review'])

### Affichage des résultats ###
#print("\nRésultats :\n")
#for review, pred in zip(df_test['Review'], df_test['Prediction']):
#    print(f"[{pred.upper()}] {review}")



########################################################

### Evaluation (if 'Category' true) ###
if 'Category' in df_test.columns:
    # Nettoyage : suppression des lignes sans catégorie réelle
    df_test_clean = df_test.dropna(subset=['Category'])

    # normalize name
    df_test_clean['Category'] = df_test_clean['Category'].str.strip().str.capitalize()
    df_test_clean['Prediction'] = df_test_clean['Prediction'].str.strip().str.capitalize()

    # print report
    print("\nÉvaluation du modèle sur les données de test (nettoyées) :\n")
    report = classification_report(df_test_clean['Category'], df_test_clean['Prediction'])
    print(classification_report(df_test_clean['Category'], df_test_clean['Prediction']))

    # Save report txt/csv
    with open("data/results/report_NB.txt", "w") as f:
        f.write(report)

    print("✅ Rapport texte sauvegardé dans 'data/results/report_NB.txt'")

    # Générer le rapport sous forme de dictionnaire
    report_dict = classification_report(
        df_test_clean['Category'],
        df_test_clean['Prediction'],
        output_dict=True
    )

    # Conversion en DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Sauvegarde en CSV
    report_df.to_csv("data/results/report_NB.csv")

    print("✅ Rapport CSV sauvegardé dans 'data/results/report_NB.csv'")

    

#############################################################################@

### Predicions Save ###
OUTPUT_PATH = 'data/results/NB_test_results.csv'
df_test.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Résultats sauvegardés dans : {OUTPUT_PATH}")



############################################################################@

### classification ###


test_data = pd.read_csv("data/results/NB_test_results.csv")


# create dir
categories = test_data['Prediction'].unique()
for category in categories:
    os.makedirs(f"{OUTPUT_DIR}/{category}", exist_ok=True)

# save review in dir
for index, row in test_data.iterrows():
    category = row['Prediction']
    content = row['Review']
    filename = f"{OUTPUT_DIR}/{category}/comment_{index+1}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

print(f"\n✅ Tous les commentaires ont été classés dans : '{OUTPUT_DIR}/[Bug|Feature|Feedback]/'")




