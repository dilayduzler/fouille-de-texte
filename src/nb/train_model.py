import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# charging trained model
TRAIN_CSV_PATH = "data/intermediate/cleaned_dataset.csv"
print(f"Chargement du fichier : {TRAIN_CSV_PATH}")
df = pd.read_csv(TRAIN_CSV_PATH, sep=';')  # Assure-toi que le fichier est bien formaté

# verif rows
assert 'Review' in df.columns and 'Category' in df.columns, "Colonnes 'Review' et 'Category' requises."

# split data (train/test) to eval interne
X_train, X_test, y_train, y_test = train_test_split(
    df['Review'], df['Category'], test_size=0.2, random_state=42)

# create model and train
print("Entraînement du modèle Naïve Bayes...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# eval interne
y_pred = model.predict(X_test)
print("\nÉvaluation sur un échantillon de validation :\n")
print(classification_report(y_test, y_pred))

# save model
MODEL_PATH = "./model/naive_bayes_model.joblib"
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Modèle sauvegardé dans : {MODEL_PATH}")
