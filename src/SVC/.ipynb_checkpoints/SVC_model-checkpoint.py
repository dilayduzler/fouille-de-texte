# Import

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Charging Data
df = pd.read_csv('cleaned_dataset.csv', sep=';')  # Attention au séparateur !

# prep data
X_raw = df['Review']
y_raw = df['Category']

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X_raw)

# Encodage des labels (texte → nombre)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# === 3. Séparation en train/test ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Entraînement du modèle SVM ===
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# === 5. Évaluation ===
y_pred = model.predict(X_val)

print("=== Évaluation du modèle ===")
print(f"Exactitude : {accuracy_score(y_val, y_pred)*100:.2f}%\n")
print("Rapport de classification :")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
print("Matrice de confusion :")
print(confusion_matrix(y_val, y_pred))


### Save ###

joblib.dump(model, 'svm_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

