"""
Task 3 (Medium) - Recommendation System Analysis
Auspify Technologies - Data Science Internship

Goal: Develop a basic content recommendation model based on Netflix titles
and categories.

Workflow:
 Step 1: Extract relevant content features.
 Step 2: Perform text preprocessing.
 Step 3: Calculate content similarity.
 Step 4: Generate recommendations for selected titles.
 Step 5: Evaluate recommendation quality.
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task3")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH)
    else:
        from task1_data_cleaning import run as clean_run
        df = clean_run(save_report=False)
    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_recommendations(title, df, cosine_sim, indices, top_n=5):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    rec_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]
    result = df.iloc[rec_indices][["title", "type", "listed_in"]].copy()
    result["similarity_score"] = scores
    return result


def run():
    print("=" * 70)
    print("TASK 3: RECOMMENDATION SYSTEM ANALYSIS")
    print("=" * 70)

    df = load_data().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1: Extract relevant content features
    # ------------------------------------------------------------------
    # Combine genre + director + type into one "content soup" feature
    df["director"] = df["director"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("")
    df["content_features"] = (
        df["listed_in"].astype(str) + " " +
        df["director"].astype(str) + " " +
        df["type"].astype(str)
    )
    print(f"\n[Step 1] Combined content features from: genre (listed_in), director, type")
    print(df[["title", "content_features"]].head(3).to_string())

    # ------------------------------------------------------------------
    # Step 2: Perform text preprocessing
    # ------------------------------------------------------------------
    df["content_features_clean"] = df["content_features"].apply(clean_text)
    print("\n[Step 2] Text cleaned (lowercased, punctuation removed).")
    print(df[["title", "content_features_clean"]].head(3).to_string())

    # ------------------------------------------------------------------
    # Step 3: Calculate content similarity (TF-IDF + cosine similarity)
    # ------------------------------------------------------------------
    # Use a subset for performance/demo purposes if dataset is large
    sample_size = min(3000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_sample["content_features_clean"])
    print(f"\n[Step 3] TF-IDF matrix shape: {tfidf_matrix.shape} "
          f"(on a sample of {sample_size} titles for speed)")

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_sample.index, index=df_sample["title"]).drop_duplicates()
    print("Cosine similarity matrix computed.")

    # ------------------------------------------------------------------
    # Step 4: Generate recommendations for selected titles
    # ------------------------------------------------------------------
    preferred_titles = ["Ganglands", "Midnight Mass", "Dick Johnson Is Dead"]
    sample_titles = [t for t in preferred_titles if t in indices]
    remaining_needed = 3 - len(sample_titles)
    if remaining_needed > 0:
        extra = list(df_sample["title"].sample(remaining_needed, random_state=1))
        sample_titles += [t for t in extra if t not in sample_titles]

    all_recs = {}
    for t in sample_titles:
        recs = get_recommendations(t, df_sample, cosine_sim, indices, top_n=5)
        all_recs[t] = recs
        print(f"\n[Step 4] Top 5 recommendations for '{t}':")
        if recs is not None:
            print(recs.to_string(index=False))
        else:
            print("Title not found in sample.")

    # ------------------------------------------------------------------
    # Step 5: Evaluate recommendation quality
    # ------------------------------------------------------------------
    # Simple proxy metric: genre overlap ratio between the seed title and
    # its recommendations (higher = more thematically consistent recs).
    eval_rows = []
    for t, recs in all_recs.items():
        if recs is None:
            continue
        seed_genres = set(g.strip() for g in df_sample.loc[indices[t], "listed_in"].split(","))
        overlaps = []
        for genre_str in recs["listed_in"]:
            rec_genres = set(g.strip() for g in str(genre_str).split(","))
            overlap = len(seed_genres & rec_genres) / max(len(seed_genres), 1)
            overlaps.append(overlap)
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        eval_rows.append({"seed_title": t, "avg_genre_overlap": round(avg_overlap, 2),
                           "avg_similarity_score": round(recs["similarity_score"].mean(), 3)})

    eval_df = pd.DataFrame(eval_rows)
    print("\n[Step 5] Recommendation quality evaluation (genre overlap proxy):")
    print(eval_df.to_string(index=False))

    # Save outputs
    out_path = os.path.join(OUTPUT_DIR, "sample_recommendations.txt")
    with open(out_path, "w") as f:
        for t, recs in all_recs.items():
            f.write(f"Recommendations for '{t}':\n")
            if recs is not None:
                f.write(recs.to_string(index=False))
            f.write("\n\n")
        f.write("Evaluation:\n")
        f.write(eval_df.to_string(index=False))
    print(f"\nRecommendations + evaluation saved -> {out_path}")

    return all_recs, eval_df


if __name__ == "__main__":
    run()
