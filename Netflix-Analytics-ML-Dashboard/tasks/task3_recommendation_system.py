"""
Task 3 (Medium) - Recommendation System Analysis
Auspify Technologies - Data Science Internship

Goal: Develop a content recommendation model based on Netflix titles
and categories.

FIXED VERSION:
- Uses FULL dataset instead of 3000 sample
- Better content feature engineering
- Added diversity in recommendations
- Added evaluation metrics

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


def build_content_features(df):
    """
    Build rich content features for recommendation.
    FIXED: Uses full dataset and richer feature set.
    """
    work = df.copy().reset_index(drop=True)
    
    # Fill missing values
    work["director"] = work["director"].fillna("Unknown")
    work["listed_in"] = work["listed_in"].fillna("")
    work["cast"] = work.get("cast", pd.Series([""] * len(work))).fillna("")
    work["description"] = work.get("description", pd.Series([""] * len(work))).fillna("")
    work["country"] = work["country"].fillna("Unknown")
    
    # Rich content soup: genres + director + cast + description + country
    # Note: We intentionally do NOT include 'type' to allow cross-type recommendations
    work["content_features"] = (
        work["listed_in"].astype(str) + " " +
        work["director"].astype(str) + " " +
        work["cast"].astype(str) + " " +
        work["country"].astype(str) + " " +
        work["description"].astype(str).str[:200]  # First 200 chars of description
    )
    
    work["content_features_clean"] = work["content_features"].apply(clean_text)
    return work


def get_recommendations(title, work, cosine_sim, indices, top_n=5, diversity_boost=True):
    """
    Get content-based recommendations with optional diversity boost.
    
    FIXED: Added diversity parameter to avoid too-similar recommendations.
    """
    if title not in indices:
        return None
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get seed title info
    seed_type = work.loc[idx, "type"]
    seed_genres = set(g.strip() for g in str(work.loc[idx, "listed_in"]).split(","))
    
    # Filter out self and apply diversity
    candidates = []
    for i, score in sim_scores:
        if i == idx:
            continue
        
        candidate = work.iloc[i]
        cand_type = candidate["type"]
        cand_genres = set(g.strip() for g in str(candidate["listed_in"]).split(","))
        
        # Genre overlap
        genre_overlap = len(seed_genres & cand_genres) / max(len(seed_genres), 1)
        
        # Diversity boost: slightly penalize same-type recommendations
        # to encourage discovery of different content types
        diversity_penalty = 0.0
        if diversity_boost and cand_type == seed_type:
            diversity_penalty = 0.02  # Small penalty for same type
        
        adjusted_score = score - diversity_penalty
        
        candidates.append({
            "idx": i,
            "similarity": round(score, 3),
            "adjusted_similarity": round(adjusted_score, 3),
            "genre_overlap": round(genre_overlap, 2),
            "type": cand_type
        })
        
        if len(candidates) >= top_n * 2:  # Get more candidates for filtering
            break
    
    # Sort by adjusted similarity and take top N
    candidates.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
    selected = candidates[:top_n]
    
    rec_indices = [c["idx"] for c in selected]
    result = work.iloc[rec_indices][["title", "type", "listed_in", "release_year", "director"]].copy()
    result["similarity_score"] = [c["similarity"] for c in selected]
    result["genre_overlap"] = [c["genre_overlap"] for c in selected]
    
    return result


def evaluate_recommendations(seed_title, recs, work, indices):
    """Evaluate recommendation quality with multiple metrics."""
    if recs is None or recs.empty:
        return None
    
    idx = indices[seed_title]
    seed_row = work.loc[idx]
    seed_genres = set(g.strip() for g in str(seed_row["listed_in"]).split(","))
    seed_type = seed_row["type"]
    
    metrics = {
        "seed_title": seed_title,
        "avg_similarity": recs["similarity_score"].mean(),
        "avg_genre_overlap": recs["genre_overlap"].mean(),
        "same_type_ratio": (recs["type"] == seed_type).mean(),
        "diversity_score": recs["type"].nunique() / len(recs) if len(recs) > 0 else 0
    }
    
    return metrics


def run():
    print("=" * 70)
    print("TASK 3: RECOMMENDATION SYSTEM ANALYSIS")
    print("=" * 70)

    df = load_data()
    
    # FIXED: Use full dataset instead of 3000 sample
    print(f"\nDataset loaded: {len(df)} titles")
    print("Using FULL dataset for recommendations (fixed from 3000 sample)")

    # ------------------------------------------------------------------
    # Step 1: Extract relevant content features
    # ------------------------------------------------------------------
    work = build_content_features(df)
    print(f"\n[Step 1] Combined content features from: genre, director, cast, country, description")

    # ------------------------------------------------------------------
    # Step 2: Perform text preprocessing
    # ------------------------------------------------------------------
    print("\n[Step 2] Text cleaned (lowercased, punctuation removed).")

    # ------------------------------------------------------------------
    # Step 3: Calculate content similarity (TF-IDF + cosine similarity)
    # ------------------------------------------------------------------
    # FIXED: Full dataset, no sampling
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(work["content_features_clean"])
    print(f"\n[Step 3] TF-IDF matrix shape: {tfidf_matrix.shape} (FULL dataset)")

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(work.index, index=work["title"]).drop_duplicates()
    print("Cosine similarity matrix computed.")

    # ------------------------------------------------------------------
    # Step 4: Generate recommendations for selected titles
    # ------------------------------------------------------------------
    # Test with diverse titles
    test_titles = [
        "Ganglands",
        "Midnight Mass", 
        "Dick Johnson Is Dead",
        "Stranger Things",
        "The Crown"
    ]
    
    available_titles = [t for t in test_titles if t in indices]
    
    # If preferred titles not found, sample some popular ones
    if len(available_titles) < 3:
        extra = list(work["title"].sample(5, random_state=1))
        available_titles = list(dict.fromkeys(available_titles + extra))[:5]
    
    print(f"\n[Step 4] Generating recommendations for {len(available_titles)} titles:")
    
    all_recs = {}
    all_metrics = []
    
    for t in available_titles:
        recs = get_recommendations(t, work, cosine_sim, indices, top_n=5, diversity_boost=True)
        all_recs[t] = recs
        
        print(f"\n  Top 5 recommendations for '{t}':")
        if recs is not None:
            print(recs.to_string(index=False))
            
            # Evaluate
            metrics = evaluate_recommendations(t, recs, work, indices)
            if metrics:
                all_metrics.append(metrics)
                print(f"  -> Avg similarity: {metrics['avg_similarity']:.3f}, "
                      f"Genre overlap: {metrics['avg_genre_overlap']:.2f}, "
                      f"Type diversity: {metrics['diversity_score']:.2f}")
        else:
            print("  Title not found in dataset.")

    # ------------------------------------------------------------------
    # Step 5: Evaluate recommendation quality
    # ------------------------------------------------------------------
    if all_metrics:
        eval_df = pd.DataFrame(all_metrics)
        print("\n[Step 5] Overall Recommendation Quality:")
        print(f"  Average similarity: {eval_df['avg_similarity'].mean():.3f}")
        print(f"  Average genre overlap: {eval_df['avg_genre_overlap'].mean():.2f}")
        print(f"  Average type diversity: {eval_df['diversity_score'].mean():.2f}")

    # Save outputs
    out_path = os.path.join(OUTPUT_DIR, "sample_recommendations.txt")
    with open(out_path, "w") as f:
        f.write("TASK 3 - RECOMMENDATION SYSTEM REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("FIXES APPLIED:\n")
        f.write("- Using FULL dataset instead of 3000 sample\n")
        f.write("- Richer content features (cast, description, country)\n")
        f.write("- Added diversity boost for cross-type recommendations\n")
        f.write("- Added evaluation metrics\n\n")
        
        for t, recs in all_recs.items():
            f.write(f"\nRecommendations for '{t}':\n")
            f.write("-" * 40 + "\n")
            if recs is not None:
                f.write(recs.to_string(index=False))
            f.write("\n\n")
        
        if all_metrics:
            f.write("\nEvaluation Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(pd.DataFrame(all_metrics).to_string(index=False))
            
    print(f"\nRecommendations + evaluation saved -> {out_path}")

    return all_recs


if __name__ == "__main__":
    run()
