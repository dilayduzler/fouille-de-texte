
#########################################

# Import

import pandas as pd
import os
import csv
import re


##########################################

# Comparaispon

### Charger les deux fichiers CSV ###
df1 = pd.read_csv("data/results/Eval-nb/NB_test_results.csv")
df2 = pd.read_csv("data/results/Eval_SVC/test_resultats.csv")


### Standardiser les noms de colonnes ###
df1.columns = [col.strip().lower() for col in df1.columns]
df2.columns = [col.strip().lower() for col in df2.columns]

### Nettoyer toutes les valeurs (minuscules + trim) ###
for col in df1.columns:
    df1[col] = df1[col].astype(str).str.lower().str.strip()
for col in df2.columns:
    df2[col] = df2[col].astype(str).str.lower().str.strip()

### Fusionner sur 'id' ###
merged = pd.merge(df1, df2, on="id", how="outer", suffixes=('_file1', '_file2'))

### Comparaison des prédictions ###
def comparer(row):
    pred1 = row.get("prediction_file1")
    pred2 = row.get("prediction_file2")
    
    if pred1 != pred2:
        return f"[{row['id']}] Différent : fichier1 = {pred1}, fichier2 = {pred2}"
    else:
        return f"[{row['id']}] Identique : {pred1}"

### Appliquer la comparaison ###
merged["résultat"] = merged.apply(comparer, axis=1)

### save in txt ###
with open("data/results/comparaison_resultat.txt", "w", encoding="utf-8") as f:
    for line in merged["résultat"]:
        f.write(line + "\n")

print("✅ Résultat enregistré dans 'comparaison_resultat.txt'")

### save in csv ###
try :
    merged.to_csv("data/results/comparaison_resultat.csv", index=False)
except:
    exit()
print("✅ Résultat enregistré dans 'comparaison_resultat.csv'")


##################################################################

# Split conflict

diff_list = []
autres_list = []

pattern = re.compile(r"\[(\d+)\]\s*(\w+)\s*:\s*(.+)")

with open("data/results/comparaison_resultat.txt", "r", encoding="utf-8") as f:
    for ligne in f:
        ligne = ligne.strip()
        match = pattern.match(ligne)
        if match:
            id = int(match.group(1))
            resultat = match.group(2)
            cat = match.group(3)
            ligne_dict = {"id": id, "résultat": resultat, "catégorie": cat}

            if resultat.lower() == "différent":
                diff_list.append(ligne_dict)
            else:
                autres_list.append(ligne_dict)

print("Différents :", diff_list)
print("Autres :", autres_list)


#################################################################

# Reclassement, Cross_corpus

cross_data = pd.read_csv("data/results/comparaison_resultat.csv")
OUTPUT_DIR = "data/cross-model_corpus"
categories = cross_data['category_file1'].unique()
conflict_path = "data/cross-model_corpus/conflict"
for category in categories:
    os.makedirs(f"{OUTPUT_DIR}/{category}", exist_ok=True)

ids_al = [item["id"] for item in autres_list]
ids_d = [item["id"] for item in diff_list]

# Boucle sur toutes les lignes du DataFrame
for index, row in cross_data.iterrows():
    if str(row['id']) in [str(e) for e in ids_al]:  # check id
        category = row['category_file1']
        content = row['review_file1']

        # create dir if doesn't exist
        category_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(category_path, exist_ok=True)

        # save the review in the dir
        filename = os.path.join(category_path, f"comment_{index+1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    elif str(row['id']) in [str(e) for e in ids_d]: # check id (again)
        content = row['review_file1']

        # save the review in conflict dir
        filename = os.path.join(conflict_path, f"comment_{index+1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

print("✅ cross_corpus assembled! ")


##############################🌙##################################