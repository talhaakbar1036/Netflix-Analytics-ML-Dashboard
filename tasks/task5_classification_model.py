"""
Task 5 (Advanced) - Machine Learning Classification Model
Auspify Technologies - Data Science Internship

Goal: Build a machine learning model to classify Netflix content
(Movie vs TV Show) based on available features.

Workflow:
 Step 1: Select and prepare features.
 Step 2: Split data into training and testing sets.
 Step 3: Train classification models.
 Step 4: Evaluate model performance.
 Step 5: Compare model accuracy.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task5")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH)
    else:
        from task1_data_cleaning import run as clean_run
        df = clean_run(save_report=False)
    return df


def run():
    print("=" * 70)
    print("TASK 5: MACHINE LEARNING CLASSIFICATION MODEL")
    print("=" * 70)
    print("\nTarget: predict content 'type' (Movie vs TV Show) from metadata.")

    df = load_data().copy()

    # ------------------------------------------------------------------
    # Step 1: Select and prepare features
    # ------------------------------------------------------------------
    # Primary genre (first item in listed_in)
    df["primary_genre"] = df["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
    df["primary_country"] = df["country"].apply(lambda x: str(x).split(",")[0].strip())
    df["director_known"] = (df["director"] != "Unknown").astype(int)

    # duration_value/unit may already exist from Task 1 cleaning
    if "duration_value" not in df.columns:
        dur_split = df["duration"].str.extract(r"(\d+)")
        df["duration_value"] = pd.to_numeric(dur_split[0], errors="coerce")
    df["duration_value"] = df["duration_value"].fillna(df["duration_value"].median())

    if "year_added" not in df.columns:
        df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year
    df["year_added"] = df["year_added"].fillna(df["year_added"].median())

    feature_cols_cat = ["primary_genre", "primary_country", "rating"]
    feature_cols_num = ["release_year", "duration_value", "year_added", "director_known"]

    le_dict = {}
    df_model = df.copy()
    for col in feature_cols_cat:
        le = LabelEncoder()
        df_model[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    target_le = LabelEncoder()
    df_model["type_enc"] = target_le.fit_transform(df_model["type"])

    feature_cols = [c + "_enc" for c in feature_cols_cat] + feature_cols_num
    X = df_model[feature_cols]
    y = df_model["type_enc"]

    print(f"\n[Step 1] Features selected: {feature_cols}")
    print(f"Target classes: {list(target_le.classes_)}")
    print(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------
    # Step 2: Split data into training and testing sets
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Step 2] Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Step 3: Train classification models
    # ------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }

    results = []
    trained_models = {}
    print("\n[Step 3] Training models...")
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        trained_models[name] = (model, preds)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        results.append({"model": name, "accuracy": acc, "precision": prec,
                         "recall": rec, "f1_score": f1})
        print(f" - {name} trained. Accuracy: {acc:.3f}")

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

    # ------------------------------------------------------------------
    # Step 4: Evaluate model performance
    # ------------------------------------------------------------------
    best_model_name = results_df.iloc[0]["model"]
    best_model, best_preds = trained_models[best_model_name]
    print(f"\n[Step 4] Best model: {best_model_name}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, best_preds, target_names=target_le.classes_))

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    # Feature importance (Random Forest)
    if "Random Forest" in trained_models:
        rf_model = trained_models["Random Forest"][0]
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances.values, y=importances.index, color="#E50914")
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"Feature importance chart saved -> {fi_path}")

    # ------------------------------------------------------------------
    # Step 5: Compare model accuracy
    # ------------------------------------------------------------------
    print("\n[Step 5] Model comparison (sorted by accuracy):")
    print(results_df.to_string(index=False))
    print(
        "\nNote: accuracy is near-perfect because 'duration_value' is measured in "
        "different units per type (minutes for Movies, seasons for TV Shows), which "
        "makes it an almost perfect proxy for the target. This is a classic example "
        "of a near-leaky feature -- worth flagging in any real business report."
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x="accuracy", y="model", data=results_df, color="#221f1f")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.xlim(0, 1)
    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"Model comparison chart saved -> {comp_path}")

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("TASK 5 - CLASSIFICATION MODEL REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Features used: {feature_cols}\n\n")
        f.write("Model comparison:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_model_name}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, best_preds, target_names=target_le.classes_))
    print(f"\nFull report saved -> {report_path}")

    return results_df


if __name__ == "__main__":
    run()
