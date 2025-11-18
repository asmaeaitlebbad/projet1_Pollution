import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1️⃣ Charger ton dataset
df = pd.read_csv("pollution.csv")  # ton fichier avec PM2.5, NO2, etc.

# 2️⃣ Séparer les features et la cible
X = df[['PM2.5','PM10','NO2','SO2','CO']]  # supprime 'O3'
y = df['Qualite_air']


# 3️⃣ Entraîner le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 4️⃣ Sauvegarder le modèle dans un fichier .pkl
with open("random_forest_air.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Modèle sauvegardé dans random_forest_air.pkl ✅")
