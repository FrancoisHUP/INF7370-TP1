import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score

# Load the dataset
df = pd.read_csv("preprocessed_data.csv")

# Ensure avg_norm_levenshtein_distance and z_score_similarity are numeric
df["avg_norm_levenshtein_distance"] = pd.to_numeric(df["avg_norm_levenshtein_distance"], errors="coerce")
df["z_score_similarity"] = pd.to_numeric(df["z_score_similarity"], errors="coerce")

# Drop rows where z_score_similarity is NaN
df = df.dropna(subset=["z_score_similarity"])

# Extract target (class) and feature (z_score_similarity)
y_true = df["class"]  # 1 = spammer, 0 = legitimate user
y_scores = df["z_score_similarity"]

# INVERT TARGET LABELS because lower Z-score means legitimate, higher means spam
y_true_inverted = 1 - y_true  # Swap spammer (1) ↔ legitimate (0)

### Compute Corrected AUC-ROC Score
auc = roc_auc_score(y_true_inverted, y_scores)  # Use inverted labels
print(f"Corrected AUC-ROC Score: {auc:.4f} (1.0 = perfect, 0.5 = random)")

### Find the Best Classification Threshold
precisions, recalls, thresholds = precision_recall_curve(y_true_inverted, y_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # Avoid division by zero
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best Threshold for classification: {best_threshold:.4f}")

### Compute F1-score at the Best Threshold
y_pred = (y_scores >= best_threshold).astype(int)  # Apply threshold
f1 = f1_score(y_true_inverted, y_pred)
print(f"F1-score at best threshold: {f1:.4f}")

### Plot Corrected ROC Curve
fpr, tpr, _ = roc_curve(y_true_inverted, y_scores)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')  # Random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (Corrected)")
plt.legend()
plt.show()

### Plot Histogram of `z_score_similarity` by Class
plt.figure(figsize=(8, 5))
plt.hist(df[df["class"] == 1]["z_score_similarity"], bins=30, alpha=0.6, label="Spammers (1)", color='r')
plt.hist(df[df["class"] == 0]["z_score_similarity"], bins=30, alpha=0.6, label="Legitimate Users (0)", color='b')
plt.axvline(best_threshold, color='black', linestyle='dashed', label=f"Best Threshold ({best_threshold:.4f})")
plt.xlabel("Z-score Similarity")
plt.ylabel("Count")
plt.legend()
plt.title("Distribution of Z-score Similarity")
plt.show()
