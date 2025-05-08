# Import

import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

### Charging trained model ###
model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

### charging test corpus ###
test_df = pd.read_csv('../cleaned_dataset_test.csv')  # colonne "Review" uniquement
X_test_raw = test_df['Review']

### Apply model ###
X_test = vectorizer.transform(X_test_raw)
y_test_pred = model.predict(X_test)

# === 4. Décodage des prédictions en labels textuels ===
labels_pred = label_encoder.inverse_transform(y_test_pred)

### Save in CSV doc ###
test_df['Prediction'] = labels_pred
test_df.to_csv('test_resultats.csv', index=False)

print("✅ Prédictions sauvegardées dans 'test_resultats.csv'")
