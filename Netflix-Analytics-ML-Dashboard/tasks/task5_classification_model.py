"""
Task 5 (Advanced) - Machine Learning Classification Model
Auspify Technologies - Data Science Internship

Goal: Build a machine learning model to classify Netflix content
(Movie vs TV Show) based on available features.

FIXED VERSION: Addresses data leakage issues
- duration_value is a near-leaky feature (minutes vs seasons)
- director_known is a data collection artifact, not a real pattern
- Added FAIR MODE that excludes both leaky features

Workflow:
 Step 1: Select and prepare features (with leakage awareness)
 Step 2: Split data into training and testing sets.
 Step 3: Train classification models.
 Step 4: Evaluate model performance.
 Step 5: Compare model accuracy and flag leakage.
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


def prepare_features(df, use_duration=True, use_director_known=True):
    """
    Prepare features with option to exclude leaky features.
    
    Leaky features:
    1) duration_value: Minutes for Movies, seasons for TV Shows - almost perfect proxy
    2) director_known: Data collection artifact - 97% Movies have director, only 9% TV Shows
    
    FAIR MODE (both=False): Uses only genre, country, rating, release_year
    """
    work = df.copy()
    work["primary_genre"] = work["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
    work["primary_country"] = work["country"].apply(lambda x: str(x).split(",")[0].strip())
    work["director_known"] = (work["director"] != "Unknown").astype(int)

    if "duration_value" not in work.columns:
        dur_split = work["duration"].str.extract(r"(\d+)")
        work["duration_value"] = pd.to_numeric(dur_split[0], errors="coerce")
    work["duration_value"] = work["duration_value"].fillna(work["duration_value"].median())

    if "year_added" not in work.columns:
        work["year_added"] = pd.to_datetime(work["date_added"], errors="coerce").dt.year
    work["year_added"] = work["year_added"].fillna(work["year_added"].median())

    feature_cols_cat = ["primary_genre", "primary_country", "rating"]
    feature_cols_num = ["release_year", "year_added"]
    
    if use_duration:
        feature_cols_num.append("duration_value")
    if use_director_known:
        feature_cols_num.append("director_known")

    le_dict = {}
    for col in feature_cols_cat:
        le = LabelEncoder()
        work[col + "_enc"] = le.fit_transform(work[col].astype(str))
        le_dict[col] = le

    target_le = LabelEncoder()
    work["type_enc"] = target_le.fit_transform(work["type"])

    feature_cols = [c + "_enc" for c in feature_cols_cat] + feature_cols_num
    X = work[feature_cols]
    y = work["type_enc"]

    return X, y, feature_cols, le_dict, target_le, work


def train_and_evaluate(X, y, feature_cols, target_le, model_name_suffix=""):
    """Train multiple models and return results."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }

    results = []
    trained_models = {}
    
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
        results.append({
            "model": name, 
            "accuracy": acc, 
            "precision": prec,
            "recall": rec, 
            "f1_score": f1
        })

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    
    # Get best model predictions
    best_model_name = results_df.iloc[0]["model"]
    best_model, best_preds = trained_models[best_model_name]
    
    return results_df, best_model_name, best_model, best_preds, y_test, trained_models


def plot_confusion_matrix(y_test, preds, target_le, best_model_name, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_feature_importance(trained_models, feature_cols, output_path):
    """Plot Random Forest feature importance."""
    if "Random Forest" in trained_models:
        rf_model = trained_models["Random Forest"][0]
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances.values, y=importances.index, color="#E50914")
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return importances
    return None


def plot_model_comparison(results_df, output_path, title_suffix=""):
    """Plot model comparison chart."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x="accuracy", y="model", data=results_df, color="#221f1f")
    plt.title(f"Model Accuracy Comparison{title_suffix}")
    plt.xlabel("Accuracy")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run():
    print("=" * 70)
    print("TASK 5: MACHINE LEARNING CLASSIFICATION MODEL")
    print("=" * 70)
    print("\nTarget: predict content 'type' (Movie vs TV Show) from metadata.")

    df = load_data().copy()

    # ======================================================================
    # VERSION 1: FULL MODEL (with all features - shows leakage issue)
    # ======================================================================
    print("\n" + "=" * 70)
    print("VERSION 1: FULL MODEL (all features - includes leaky features)")
    print("=" * 70)
    
    X_full, y_full, feature_cols_full, le_dict_full, target_le, work = prepare_features(
        df, use_duration=True, use_director_known=True
    )
    
    print(f"\nFeatures used: {feature_cols_full}")
    print("WARNING: duration_value and director_known are near-leaky features!")
    
    results_full, best_name_full, best_model_full, best_preds_full, y_test_full, trained_full = train_and_evaluate(
        X_full, y_full, feature_cols_full, target_le
    )
    
    print(f"\nBest model: {best_name_full}")
    print(f"Test accuracy: {results_full.iloc[0]['accuracy']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_full, best_preds_full, target_names=target_le.classes_))

    # Save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test_full, best_preds_full, target_le, best_name_full, cm_path)
    print(f"Confusion matrix saved -> {cm_path}")

    # Save feature importance
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    importances_full = plot_feature_importance(trained_full, feature_cols_full, fi_path)
    if importances_full is not None:
        print(f"\nFeature Importance (leaky features marked with *):")
        for feat, imp in importances_full.items():
            marker = " *** LEAKY" if feat in ["duration_value", "director_known"] else ""
            print(f"  {feat}: {imp:.3f}{marker}")
        print(f"\nFeature importance chart saved -> {fi_path}")

    # Save model comparison
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plot_model_comparison(results_full, comp_path, " (Full Model)")
    print(f"Model comparison saved -> {comp_path}")

    # ======================================================================
    # VERSION 2: FAIR MODEL (without leaky features)
    # ======================================================================
    print("\n" + "=" * 70)
    print("VERSION 2: FAIR MODEL (excludes leaky features)")
    print("=" * 70)
    
    X_fair, y_fair, feature_cols_fair, le_dict_fair, target_le_fair, work_fair = prepare_features(
        df, use_duration=False, use_director_known=False
    )
    
    print(f"\nFeatures used: {feature_cols_fair}")
    print("This model learns purely from content metadata (genre, country, rating, year)")
    
    results_fair, best_name_fair, best_model_fair, best_preds_fair, y_test_fair, trained_fair = train_and_evaluate(
        X_fair, y_fair, feature_cols_fair, target_le_fair, "_fair"
    )
    
    print(f"\nBest model: {best_name_fair}")
    print(f"Test accuracy: {results_fair.iloc[0]['accuracy']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_fair, best_preds_fair, target_names=target_le_fair.classes_))

    # Save fair model confusion matrix
    cm_fair_path = os.path.join(OUTPUT_DIR, "confusion_matrix_fair.png")
    plot_confusion_matrix(y_test_fair, best_preds_fair, target_le_fair, best_name_fair, cm_fair_path)
    print(f"Fair model confusion matrix saved -> {cm_fair_path}")

    # Save fair model feature importance
    fi_fair_path = os.path.join(OUTPUT_DIR, "feature_importance_fair.png")
    importances_fair = plot_feature_importance(trained_fair, feature_cols_fair, fi_fair_path)
    if importances_fair is not None:
        print(f"\nFeature Importance (Fair Model):")
        for feat, imp in importances_fair.items():
            print(f"  {feat}: {imp:.3f}")
        print(f"\nFair feature importance chart saved -> {fi_fair_path}")

    # Save fair model comparison
    comp_fair_path = os.path.join(OUTPUT_DIR, "model_comparison_fair.png")
    plot_model_comparison(results_fair, comp_fair_path, " (Fair Model)")
    print(f"Fair model comparison saved -> {comp_fair_path}")

    # ======================================================================
    # COMPARISON & LEAKAGE ANALYSIS
    # ======================================================================
    print("\n" + "=" * 70)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 70)
    
    acc_full = results_full.iloc[0]['accuracy']
    acc_fair = results_fair.iloc[0]['accuracy']
    acc_drop = acc_full - acc_fair
    
    print(f"\nFull Model Accuracy:  {acc_full:.3f}")
    print(f"Fair Model Accuracy:  {acc_fair:.3f}")
    print(f"Accuracy Drop:        {acc_drop:.3f} ({acc_drop/acc_full*100:.1f}% relative)")
    
    if acc_drop > 0.1:
        print("\n⚠️  SIGNIFICANT DATA LEAKAGE DETECTED!")
        print("   The accuracy drop indicates that leaky features were carrying")
        print("   predictive signal that isn't available in real-world scenarios.")
    else:
        print("\n✓  Minimal leakage impact - model is reasonably robust.")

    # ======================================================================
    # SAVE REPORTS
    # ======================================================================
    # Full model report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("TASK 5 - CLASSIFICATION MODEL REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("VERSION 1: FULL MODEL (with leaky features)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Features used: {feature_cols_full}\n\n")
        f.write("Model comparison:\n")
        f.write(results_full.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_name_full}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test_full, best_preds_full, target_names=target_le.classes_))
        f.write("\n\n")
        
        f.write("DATA LEAKAGE WARNING:\n")
        f.write("-" * 40 + "\n")
        f.write("The following features are near-leaky:\n")
        f.write("1. duration_value: Different units for Movies (minutes) vs TV Shows (seasons)\n")
        f.write("2. director_known: Data collection artifact (97% Movies vs 9% TV Shows have director)\n")
        f.write("\nThese features give the model an unfair advantage not available in practice.\n\n")
        
        f.write("VERSION 2: FAIR MODEL (without leaky features)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Features used: {feature_cols_fair}\n\n")
        f.write("Model comparison:\n")
        f.write(results_fair.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_name_fair}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test_fair, best_preds_fair, target_names=target_le_fair.classes_))
        f.write("\n\n")
        
        f.write("LEAKAGE IMPACT ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Full Model Accuracy:  {acc_full:.3f}\n")
        f.write(f"Fair Model Accuracy:  {acc_fair:.3f}\n")
        f.write(f"Accuracy Drop:        {acc_drop:.3f}\n")
        
    print(f"\nFull report saved -> {report_path}")

    # Save model metadata for app.py
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"full_model_accuracy: {acc_full:.4f}\n")
        f.write(f"fair_model_accuracy: {acc_fair:.4f}\n")
        f.write(f"accuracy_drop: {acc_drop:.4f}\n")
        f.write(f"leakage_detected: {'YES' if acc_drop > 0.1 else 'NO'}\n")
        f.write(f"best_full_model: {best_name_full}\n")
        f.write(f"best_fair_model: {best_name_fair}\n")
    print(f"Model metadata saved -> {meta_path}")

    return results_full, results_fair


if __name__ == "__main__":
    run()
