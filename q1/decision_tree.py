import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Define the dataset
data = pd.DataFrame({
    "nombre_de_tweets": [1087, 1200, 1000, 890, 500, 950],
    "similarité_tweets": [3.10, 0.1, 1.6, 1.5, 2.5, 1.0], 
    "fréquence_tweets": [2, 1.25, 0.05, 0.9, 1.2, 0.7], 
    "classe": [1, 0, 0, 1, 0, 1]  # More class mix
})


# 2. Separate features and target
X = data[["nombre_de_tweets", "similarité_tweets", "fréquence_tweets"]]
y = data["classe"]

# 3. Train a simple Decision Tree with max_depth=2
dt_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_clf.fit(X, y)


# 1. Predict the classes of the training data
y_pred = dt_clf.predict(X)

# 2. Compute accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 3. Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 4. Plot the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the Decision Tree without 'gini', 'samples', and 'value'
plt.figure(figsize=(8, 5))
plot_tree(dt_clf, 
          filled=True, 
          feature_names=X.columns, 
          class_names=["Legitime", "Pollueur"], 
          impurity=False,   # Removes 'gini'
          proportion=False, # Keeps absolute values
          rounded=True, 
          fontsize=10)
plt.show()

