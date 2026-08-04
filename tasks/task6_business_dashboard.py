"""
Task 6 (Advanced) - Data Science Business Insights Dashboard
Auspify Technologies - Data Science Internship

Goal: Create an end-to-end data science project that combines analytics,
machine learning, and business insights.

Workflow:
 Step 1: Perform complete data analysis.
 Step 2: Develop predictive models.
 Step 3: Create interactive visualizations.
 Step 4: Generate business recommendations.
 Step 5: Present findings in a professional report.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task6")

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
    print("TASK 6: DATA SCIENCE BUSINESS INSIGHTS DASHBOARD")
    print("=" * 70)

    df = load_data().copy()

    # ------------------------------------------------------------------
    # Step 1: Perform complete data analysis
    # ------------------------------------------------------------------
    print("\n[Step 1] Running complete analysis...")
    type_counts = df["type"].value_counts()
    top_countries = df[df["country"] != "Unknown"]["country"].value_counts().head(5)
    top_ratings = df["rating"].value_counts().head(5)

    yearly = df.groupby("release_year").size()
    yearly = yearly[(yearly.index >= 2000) & (yearly.index <= 2021)]

    print(f"Total titles: {len(df)} | Movies: {type_counts.get('Movie',0)} | "
          f"TV Shows: {type_counts.get('TV Show',0)}")
    print(f"Top country: {top_countries.index[0]} ({top_countries.iloc[0]} titles)")
    print(f"Most common rating: {top_ratings.index[0]} ({top_ratings.iloc[0]} titles)")

    # ------------------------------------------------------------------
    # Step 2: Develop predictive models
    # ------------------------------------------------------------------
    print("\n[Step 2] Developing predictive models...")

    # 2a. Forecast future content growth (regression)
    X_year = yearly.index.values.reshape(-1, 1)
    y_count = yearly.values
    growth_model = LinearRegression().fit(X_year, y_count)
    future_years = np.arange(2022, 2026).reshape(-1, 1)
    future_forecast = np.clip(growth_model.predict(future_years), 0, None).round().astype(int)
    forecast_df = pd.DataFrame({"release_year": future_years.flatten(),
                                 "predicted_titles": future_forecast})
    print("Content growth forecast (2022-2025):")
    print(forecast_df.to_string(index=False))

    # 2b. Classification model (type prediction) - condensed from Task 5
    df["primary_genre"] = df["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
    df["primary_country"] = df["country"].apply(lambda x: str(x).split(",")[0].strip())
    le_genre, le_country, le_rating, le_type = (LabelEncoder() for _ in range(4))
    df["genre_enc"] = le_genre.fit_transform(df["primary_genre"])
    df["country_enc"] = le_country.fit_transform(df["primary_country"])
    df["rating_enc"] = le_rating.fit_transform(df["rating"].astype(str))
    df["type_enc"] = le_type.fit_transform(df["type"])

    X = df[["genre_enc", "country_enc", "rating_enc", "release_year"]]
    y = df["type_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    clf_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\nContent-type classifier (genre/country/rating/year only, no duration) "
          f"accuracy: {clf_acc:.3f}")

    # ------------------------------------------------------------------
    # Step 3: Create interactive-style visualizations (dashboard image)
    # ------------------------------------------------------------------
    print("\n[Step 3] Building dashboard visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Netflix Content Strategy - Business Insights Dashboard",
                 fontsize=16, fontweight="bold")

    # Movie vs TV
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
                    colors=["#E50914", "#221f1f"], textprops={"color": "white"})
    axes[0, 0].set_title("Content Mix")

    # Top countries
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[0, 1], color="#E50914")
    axes[0, 1].set_title("Top 5 Content Markets")
    axes[0, 1].set_xlabel("Titles")

    # Growth trend + forecast
    axes[0, 2].plot(yearly.index, yearly.values, marker="o", color="#221f1f", label="Actual")
    axes[0, 2].plot(forecast_df["release_year"], forecast_df["predicted_titles"],
                     marker="o", linestyle="--", color="#E50914", label="Forecast")
    axes[0, 2].set_title("Release Trend & Forecast")
    axes[0, 2].legend(fontsize=8)

    # Ratings distribution
    sns.barplot(x=top_ratings.values, y=top_ratings.index, ax=axes[1, 0], color="#221f1f")
    axes[1, 0].set_title("Top 5 Content Ratings")
    axes[1, 0].set_xlabel("Titles")

    # Content added per year
    if "year_added" in df.columns:
        added = df["year_added"].dropna().astype(int).value_counts().sort_index()
        added = added[added.index >= 2015]
        axes[1, 1].bar(added.index.astype(str), added.values, color="#E50914")
        axes[1, 1].set_title("Titles Added to Platform / Year")
        axes[1, 1].tick_params(axis="x", rotation=45)

    # Feature importance for classifier
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=importances.values, y=importances.index, ax=axes[1, 2], color="#E50914")
    axes[1, 2].set_title("Type-Classifier Feature Importance")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dash_path = os.path.join(OUTPUT_DIR, "business_dashboard.png")
    plt.savefig(dash_path, dpi=150)
    plt.close()
    print(f"Dashboard image saved -> {dash_path}")

    # ------------------------------------------------------------------
    # Step 4: Generate business recommendations
    # ------------------------------------------------------------------
    print("\n[Step 4] Generating business recommendations...")
    recommendations = [
        f"1. Content mix: Movies dominate the catalogue ({type_counts.get('Movie',0)/len(df)*100:.0f}%). "
        f"Given TV Shows drive longer subscriber engagement/retention industry-wide, consider "
        f"increasing TV Show investment, especially in high-performing markets.",
        f"2. Market focus: {top_countries.index[0]} and {top_countries.index[1]} lead content "
        f"production. Underrepresented but populous markets could be prioritized for regional "
        f"originals to diversify the catalogue and reduce single-market dependency.",
        f"3. Catalogue growth: the forecast model projects roughly "
        f"{forecast_df['predicted_titles'].iloc[-1]} new titles/year by {int(forecast_df['release_year'].iloc[-1])} "
        f"if historical trends continue; treat this as a baseline for content-budget planning, "
        f"not a guarantee, since 2020-2021 already broke the historical growth pattern.",
        f"4. Ratings strategy: '{top_ratings.index[0]}' is the most common content rating, "
        f"indicating a mature-audience skew; family-friendly ('TV-G', 'TV-Y') content is "
        f"comparatively under-represented and could support subscriber-base broadening.",
        f"5. Data quality: {(df['director']=='Unknown').mean()*100:.0f}% of titles are missing "
        f"director metadata. Improving metadata completeness would materially improve future "
        f"recommendation-engine and analytics accuracy.",
    ]
    for r in recommendations:
        print(" - " + r)

    # ------------------------------------------------------------------
    # Step 5: Present findings in a professional report
    # ------------------------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "business_insights_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NETFLIX CONTENT STRATEGY - BUSINESS INSIGHTS REPORT\n")
        f.write("Auspify Technologies Data Science Internship - Task 6\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATA OVERVIEW\n" + "-" * 40 + "\n")
        f.write(f"Total titles analyzed: {len(df)}\n")
        f.write(f"Movies: {type_counts.get('Movie',0)} | TV Shows: {type_counts.get('TV Show',0)}\n\n")

        f.write("2. KEY ANALYTICS\n" + "-" * 40 + "\n")
        f.write(f"Top 5 markets:\n{top_countries.to_string()}\n\n")
        f.write(f"Top 5 ratings:\n{top_ratings.to_string()}\n\n")

        f.write("3. PREDICTIVE MODELS\n" + "-" * 40 + "\n")
        f.write(f"Growth forecast (2022-2025):\n{forecast_df.to_string(index=False)}\n\n")
        f.write(f"Content-type classifier accuracy: {clf_acc:.3f}\n\n")

        f.write("4. BUSINESS RECOMMENDATIONS\n" + "-" * 40 + "\n")
        for r in recommendations:
            f.write(r + "\n\n")

        f.write("5. SUPPORTING VISUALS\n" + "-" * 40 + "\n")
        f.write(f"See: {os.path.basename(dash_path)}\n")

    print(f"\n[Step 5] Full professional report saved -> {report_path}")
    print(f"Dashboard visualization saved -> {dash_path}")

    return {
        "forecast": forecast_df,
        "classifier_accuracy": clf_acc,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    run()
