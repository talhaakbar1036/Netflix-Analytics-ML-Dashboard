"""
Task 2 (Easy) - Exploratory Data Analysis (EDA)
Auspify Technologies - Data Science Internship

Goal: Explore the Netflix dataset to uncover trends, patterns, and insights.

Workflow:
 Step 1: Analyze dataset structure and statistics.
 Step 2: Study content distribution by type.
 Step 3: Identify top countries and categories.
 Step 4: Create visualizations for key metrics.
 Step 5: Summarize findings.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task2")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Use the cleaned dataset from Task 1 if it exists, else clean on the fly."""
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH)
    else:
        from task1_data_cleaning import run as clean_run
        df = clean_run(save_report=False)
    return df


def run():
    print("=" * 70)
    print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    df = load_data()

    # ------------------------------------------------------------------
    # Step 1: Analyze dataset structure and statistics
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn info:")
    print(df.dtypes.to_string())
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe().to_string())

    # ------------------------------------------------------------------
    # Step 2: Study content distribution by type
    # ------------------------------------------------------------------
    type_counts = df["type"].value_counts()
    print("\n[Step 2] Content distribution by type:")
    print(type_counts.to_string())
    print(f"Movies: {type_counts.get('Movie',0)} ({type_counts.get('Movie',0)/len(df)*100:.1f}%)")
    print(f"TV Shows: {type_counts.get('TV Show',0)} ({type_counts.get('TV Show',0)/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 3: Identify top countries and categories
    # ------------------------------------------------------------------
    top_countries = (
        df[df["country"] != "Unknown"]["country"]
        .value_counts()
        .head(10)
    )
    print("\n[Step 3] Top 10 content-producing countries:")
    print(top_countries.to_string())

    # listed_in / genres — explode genres column
    if "genres" in df.columns:
        # genres column was saved as a stringified list -> reparse
        import ast
        genre_series = df["genres"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else str(x).split(",")
        )
    else:
        genre_series = df["listed_in"].apply(lambda x: str(x).split(","))

    all_genres = pd.Series([g.strip() for sub in genre_series for g in sub])
    top_genres = all_genres.value_counts().head(10)
    print("\nTop 10 genres/categories:")
    print(top_genres.to_string())

    top_ratings = df["rating"].value_counts().head(10)
    print("\nContent rating distribution (top 10):")
    print(top_ratings.to_string())

    # ------------------------------------------------------------------
    # Step 4: Create visualizations for key metrics
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a: Movie vs TV Show
    axes[0, 0].pie(
        type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
        colors=["#E50914", "#221f1f"], startangle=90,
        textprops={"color": "white"}
    )
    axes[0, 0].set_title("Content Distribution: Movie vs TV Show")

    # 4b: Top 10 countries
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[0, 1], color="#E50914")
    axes[0, 1].set_title("Top 10 Countries by Content Count")
    axes[0, 1].set_xlabel("Number of Titles")

    # 4c: Content added per year
    if "year_added" in df.columns:
        year_counts = df["year_added"].dropna().astype(int).value_counts().sort_index()
        year_counts = year_counts[year_counts.index >= 2008]
        axes[1, 0].plot(year_counts.index, year_counts.values, marker="o", color="#E50914")
        axes[1, 0].set_title("Content Added to Netflix Per Year")
        axes[1, 0].set_xlabel("Year Added")
        axes[1, 0].set_ylabel("Number of Titles")

    # 4d: Top genres
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=axes[1, 1], color="#221f1f")
    axes[1, 1].set_title("Top 10 Genres")
    axes[1, 1].set_xlabel("Number of Titles")

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "eda_overview.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Step 4] Visualization dashboard saved -> {chart_path}")

    # Release year trend (separate chart)
    plt.figure(figsize=(10, 5))
    release_counts = df["release_year"].value_counts().sort_index()
    release_counts = release_counts[release_counts.index >= 1990]
    plt.plot(release_counts.index, release_counts.values, color="#E50914")
    plt.fill_between(release_counts.index, release_counts.values, alpha=0.2, color="#E50914")
    plt.title("Number of Titles by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    chart_path2 = os.path.join(OUTPUT_DIR, "release_year_trend.png")
    plt.savefig(chart_path2, dpi=150)
    plt.close()
    print(f"Release year trend chart saved -> {chart_path2}")

    # ------------------------------------------------------------------
    # Step 5: Summarize findings
    # ------------------------------------------------------------------
    summary_lines = [
        "TASK 2 - EDA SUMMARY OF FINDINGS",
        "=" * 40,
        f"Total titles analyzed: {len(df)}",
        f"Movies make up {type_counts.get('Movie',0)/len(df)*100:.1f}% of the catalogue, "
        f"TV Shows {type_counts.get('TV Show',0)/len(df)*100:.1f}%.",
        f"Top content-producing country: {top_countries.index[0]} ({top_countries.iloc[0]} titles).",
        f"Most common genre: {top_genres.index[0]} ({top_genres.iloc[0]} titles).",
        f"Most common content rating: {top_ratings.index[0]} ({top_ratings.iloc[0]} titles).",
        f"Peak content-addition year: {int(year_counts.idxmax())} ({int(year_counts.max())} titles added).",
    ]
    summary = "\n".join(summary_lines)
    print("\n[Step 5] " + summary)

    summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved -> {summary_path}")

    return {
        "type_counts": type_counts,
        "top_countries": top_countries,
        "top_genres": top_genres,
    }


if __name__ == "__main__":
    run()
