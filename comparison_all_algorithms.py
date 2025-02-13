import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# Define base directories
BASE_MODEL_DIR = "output/models/"
BASE_GRAPH_DIR = "output/graphs/"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(BASE_GRAPH_DIR, exist_ok=True)

class DataLoader:
    """
    Handles data loading, missing value handling, class balancing, and train-test splitting.
    """
    def __init__(self, file_path, test_size=0.2, random_state=42, imbalance_ratio=None):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.imbalance_ratio = imbalance_ratio
        self.df = self.load_data()

    def load_data(self):
        """Loads and preprocesses the dataset."""
        df = pd.read_csv(self.file_path)

        # Drop unnecessary columns
        columns_to_drop = ['user_id', 'created_at', 'collected_at', 'followings', 'tweet']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Remove rows with NaN values
        df = df.dropna()

        return df

    def balance_classes(self, df):
        """Applies class imbalance only to the training set."""
        legit_users = df[df['class'] == 0]
        polluters = df[df['class'] == 1]

        num_polluters = int(len(legit_users) * self.imbalance_ratio)
        polluters_subset = polluters.sample(n=num_polluters, random_state=self.random_state)

        return pd.concat([legit_users, polluters_subset])

    def preprocess_data(self):
        """
        Splits the dataset into training and testing sets.
        If imbalance is requested, only the training set is modified.
        """
        X = self.df.drop(columns=['class'])
        y = self.df['class']

        # Split dataset while keeping the test set unchanged
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Apply class imbalance **only to training set**
        if self.imbalance_ratio is not None:
            train_df = pd.concat([X_train, y_train], axis=1)
            train_df = self.balance_classes(train_df)
            X_train, y_train = train_df.drop(columns=['class']), train_df['class']

        return X_train, X_test, y_train, y_test

class ModelTrainer:
    """
    Handles model training, evaluation, and performance visualization.
    """
    def __init__(self, name, model, imbalance_ratio=None):
        self.name = name
        self.model = model
        self.imbalance_ratio = imbalance_ratio

        # Create unique subdirectories for models and graphs
        self.model_dir = os.path.join(BASE_MODEL_DIR, self.name)
        self.graph_dir = os.path.join(BASE_GRAPH_DIR, self.name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

        # Append suffix for imbalance cases
        self.suffix = f"_imbalanced_{self.imbalance_ratio}" if self.imbalance_ratio else ""

    def train_model(self, X_train, y_train):
        """Trains the model and saves it."""
        print(f"Training {self.name} model {'(Imbalanced)' if self.imbalance_ratio else '(Balanced)'}...")
        self.model.fit(X_train, y_train)

        # Save trained model
        model_path = os.path.join(self.model_dir, f"{self.name}_model{self.suffix}.pkl")
        joblib.dump(self.model, model_path)

        print(f"Model saved to: {model_path}")

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model and generates reports."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros_like(y_test)

        # Compute metrics
        cm = confusion_matrix(y_test, y_pred)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        TP_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
        FP_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
        F_measure = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        AUC = roc_auc_score(y_test, y_prob) if hasattr(self.model, "predict_proba") else 0

        # Save results
        self.results = {
            "Model": self.name,
            "TP Rate": TP_rate,
            "FP Rate": FP_rate,
            "F-measure": F_measure,
            "AUC": AUC
        }

        # Save ROC curve
        self.plot_roc_curve(y_test, y_prob)

        # Save Confusion Matrix heatmap
        self.plot_confusion_matrix(cm)

    def plot_roc_curve(self, y_test, y_prob):
        """Plots and saves the ROC curve."""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob) if hasattr(self.model, "predict_proba") else ([], [])
        plt.plot(fpr, tpr, label=f"{self.name} (AUC = {self.results['AUC']:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.name}")
        plt.legend()

        # Save ROC curve with correct suffix
        roc_curve_path = os.path.join(self.graph_dir, f"{self.name}_roc_curve{self.suffix}.png")
        plt.savefig(roc_curve_path)
        plt.close()

    def plot_confusion_matrix(self, cm):
        """Plots and saves the Confusion Matrix heatmap."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Polluter"], yticklabels=["Legitimate", "Polluter"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {self.name}")

        # Save Confusion Matrix with correct suffix
        conf_matrix_path = os.path.join(self.graph_dir, f"{self.name}_confusion_matrix{self.suffix}.png")
        plt.savefig(conf_matrix_path)
        plt.close()

def comparaison_algorithmes(file_path, imbalance_ratio=None):
    """Main function to train and evaluate models with different settings."""
    # Load and preprocess data
    data_loader = DataLoader(file_path, imbalance_ratio=imbalance_ratio)
    X_train, X_test, y_train, y_test = data_loader.preprocess_data()

    # Define models
    models = [
        ModelTrainer("DecisionTree", DecisionTreeClassifier(random_state=42), imbalance_ratio),
        ModelTrainer("Bagging", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42), imbalance_ratio),
        ModelTrainer("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=42), imbalance_ratio),
        ModelTrainer("GradientBoosting", GradientBoostingClassifier(n_estimators=50, random_state=42), imbalance_ratio),
        ModelTrainer("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42), imbalance_ratio),
        ModelTrainer("NaiveBayes", GaussianNB(), imbalance_ratio)
    ]

    # Train and evaluate all models
    for model in models:
        model.train_model(X_train, y_train)
        model.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    comparaison_algorithmes("preprocessed_data.csv")
    comparaison_algorithmes("preprocessed_data.csv", imbalance_ratio=0.05)
