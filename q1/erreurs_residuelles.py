import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Définition du dataset
data = pd.DataFrame({
    "nombre_de_tweets": [1087, 1200, 1000, 890, 500, 950],
    "similarité_tweets": [3.10, 0.1, 1.6, 1.5, 2.5, 1.0],
    "fréquence_tweets": [2, 1.25, 0.05, 0.9, 1.2, 0.7],
    "classe": [1, 0, 0, 1, 0, 1]
})

# Séparation des features et de la cible
X = data[["nombre_de_tweets", "similarité_tweets", "fréquence_tweets"]]
y = data["classe"]

# Entraînement de l'arbre de décision
dt_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_clf.fit(X, y)

# Calcul des probabilités pour la classe 1
data["Probabilité"] = dt_clf.predict_proba(X)[:, 1]

# Calcul des pseudo résidus pour la classification avec la log-loss :
# r_i = y_i - p_i
data["Pseudo Résidu"] = data["classe"] - data["Probabilité"]

# Affichage des résultats avec l'identifiant de la ligne
for idx, row in data.iterrows():
    print(f"Ligne {idx}: y = {row['classe']}, p = {row['Probabilité']:.6f}, pseudo résidu = {row['Pseudo Résidu']:.6f}")
