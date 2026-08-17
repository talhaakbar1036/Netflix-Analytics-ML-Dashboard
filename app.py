"""
================================================================================
 NETFLIX ANALYTICS & ML DASHBOARD -- STREAMLIT UI
 Auspify Technologies - Data Science Internship (Talha Akbar)
================================================================================

WHAT THIS FILE DOES
--------------------
Interactive web dashboard with SIX pages:

    1. Home              -> animated hero banner + live dataset stats
    2. Movie Recommender  -> content-based recommender using TF-IDF
    3. EDA Explorer       -> interactive charts from cleaned data
    4. Trend Forecast     -> trend analysis + forecast
    5. Type Predictor     -> Movie vs TV Show classifier (with leakage controls)
    6. Business Dashboard -> business insights + fair model results

FIXES APPLIED:
- Data leakage detection and fair mode toggle
- Full dataset usage in recommender (not 3000 sample)
- Better content features for recommendations
- Leakage-aware model training

PATHS
------
All paths are relative to THIS file's folder using os.path.dirname(__file__).
No hardcoded paths needed.
================================================================================
"""

import os
import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==============================================================================
# 1. PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Netflix Analytics & ML Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. PATHS - All relative to this file
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "netflix_cleaned.csv")
RAW_CSV = os.path.join(DATA_DIR, "netflix_dataset.csv")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TASK2_DIR = os.path.join(OUTPUTS_DIR, "task2")
TASK3_DIR = os.path.join(OUTPUTS_DIR, "task3")
TASK4_DIR = os.path.join(OUTPUTS_DIR, "task4")
TASK5_DIR = os.path.join(OUTPUTS_DIR, "task5")
TASK6_DIR = os.path.join(OUTPUTS_DIR, "task6")

MODELS_DIR = os.path.join(BASE_DIR, "models")


def read_text_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return "(report file not found yet -- run the matching task script first)"


def show_image_if_exists(path: str, caption: str = ""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Chart not generated yet: `{os.path.relpath(path, BASE_DIR)}`. "
                f"Run `python3 main.py` once to generate it.")


# ==============================================================================
# 3. GLOBAL CSS - Netflix dark theme
# ==============================================================================
CUSTOM_CSS = """
<style>
@keyframes bgShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    background: linear-gradient(120deg, #0b0b0f, #14100f, #1a0a0a, #0b0b0f);
    background-size: 300% 300%;
    animation: bgShift 18s ease infinite;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #140000 0%, #0b0b0f 100%);
    border-right: 1px solid rgba(229,9,20,0.25);
}

.hero-wrap {
    position: relative;
    padding: 46px 34px;
    border-radius: 22px;
    overflow: hidden;
    background: radial-gradient(circle at 20% 20%, rgba(229,9,20,0.35), transparent 55%),
                radial-gradient(circle at 80% 80%, rgba(229,9,20,0.20), transparent 55%),
                #120303;
    border: 1px solid rgba(229,9,20,0.35);
    margin-bottom: 28px;
    animation: fadeInUp 0.9s ease both;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #e50914, #ffffff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
    margin: 0;
}
.hero-sub {
    color: #d0d0d0;
    font-size: 1.1rem;
    margin-top: 6px;
}
@keyframes shine {
    to { background-position: 200% center; }
}

.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(40px);
    opacity: 0.55;
    animation: float 7s ease-in-out infinite;
}
.blob1 { width: 160px; height: 160px; background: #e50914; top: -40px; right: 60px; animation-delay: 0s; }
.blob2 { width: 110px; height: 110px; background: #ff5c5c; bottom: -30px; right: 220px; animation-delay: 1.5s; }
.blob3 { width: 90px;  height: 90px;  background: #8b0000; top: 30px; right: 380px; animation-delay: 3s; }
@keyframes float {
    0%, 100% { transform: translateY(0px) translateX(0px); }
    50%      { transform: translateY(-22px) translateX(10px); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(229,9,20,0.3);
    border-radius: 16px;
    padding: 18px 10px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeInUp 0.7s ease both;
}
.stat-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 14px 30px rgba(229,9,20,0.35);
    border-color: #e50914;
}
.stat-number {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
}
.stat-label {
    color: #b5b5b5;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 2px;
}

.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: rgba(229,9,20,0.08);
    border-top: 1px solid rgba(229,9,20,0.3);
    border-bottom: 1px solid rgba(229,9,20,0.3);
    padding: 10px 0;
    margin: 8px 0 26px 0;
    border-radius: 8px;
}
.ticker-move {
    display: inline-block;
    white-space: nowrap;
    padding-left: 100%;
    animation: ticker 32s linear infinite;
    color: #f2f2f2;
    font-weight: 600;
}
.ticker-move span { margin: 0 28px; color: #ff4d4d; }
@keyframes ticker {
    0%   { transform: translate3d(0, 0, 0); }
    100% { transform: translate3d(-100%, 0, 0); }
}

.movie-card {
    position: relative;
    border-radius: 16px;
    padding: 18px 16px 16px 16px;
    height: 210px;
    color: white;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.6s ease both;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.movie-card:hover {
    transform: translateY(-10px) scale(1.035);
    box-shadow: 0 18px 34px rgba(0,0,0,0.55);
    z-index: 5;
}
.movie-card .badge {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(0,0,0,0.45);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    backdrop-filter: blur(4px);
}
.movie-card h4 {
    margin: 0 0 4px 0;
    font-size: 1.02rem;
    line-height: 1.25;
    text-shadow: 0 2px 6px rgba(0,0,0,0.6);
}
.movie-card .genre-line {
    font-size: 0.78rem;
    opacity: 0.9;
}
.movie-card .sim-bar-bg {
    margin-top: 10px;
    height: 6px;
    background: rgba(255,255,255,0.2);
    border-radius: 4px;
    overflow: hidden;
}
.movie-card .sim-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #ffffff, #ff4d4d);
    border-radius: 4px;
    animation: growBar 1.1s ease both;
}
@keyframes growBar {
    from { width: 0%; }
}

.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    border-left: 5px solid #e50914;
    padding-left: 12px;
    margin: 6px 0 16px 0;
    animation: fadeInUp 0.6s ease both;
}

.pred-banner {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 800;
    color: white;
    animation: popIn 0.5s cubic-bezier(.26,1.4,.4,1) both;
    border: 1px solid rgba(255,255,255,0.15);
}
@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}
.glow-pulse {
    animation: popIn 0.5s cubic-bezier(.26,1.4,.4,1) both, glowPulse 2.2s ease-in-out 0.5s infinite;
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0px rgba(229,9,20,0.0); }
    50%      { box-shadow: 0 0 34px rgba(229,9,20,0.55); }
}

.spark-field {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 6;
}
.spark {
    position: absolute;
    top: 50%;
    left: 50%;
    color: #ff5c5c;
    font-size: 1.1rem;
    text-shadow: 0 0 8px rgba(229,9,20,0.9);
    transform: translate(-50%, -50%) rotate(var(--angle)) translate(0) scale(0);
    animation: sparkOut 0.9s ease-out forwards;
}
@keyframes sparkOut {
    0%   { transform: translate(-50%, -50%) rotate(var(--angle)) translate(0) scale(0);   opacity: 1; }
    70%  { opacity: 1; }
    100% { transform: translate(-50%, -50%) rotate(var(--angle)) translate(90px) scale(1); opacity: 0; }
}

.particle-field {
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
}
.particle {
    position: absolute;
    bottom: -20px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #e50914;
    box-shadow: 0 0 8px 2px rgba(229,9,20,0.7);
    opacity: 0;
    animation: rise linear infinite;
}
@keyframes rise {
    0%   { transform: translateY(0) translateX(0); opacity: 0; }
    10%  { opacity: 0.85; }
    90%  { opacity: 0.5; }
    100% { transform: translateY(-105vh) translateX(20px); opacity: 0; }
}

.typewriter {
    overflow: hidden;
    white-space: nowrap;
    border-right: 2px solid #e50914;
    width: 0;
    animation: typing 3.2s steps(60, end) 0.4s forwards, blinkCaret 0.8s step-end infinite;
}
@keyframes typing {
    from { width: 0; }
    to   { width: 100%; }
}
@keyframes blinkCaret {
    from, to { border-color: transparent; }
    50%      { border-color: #e50914; }
}

.movie-card { transform-style: preserve-3d; }
.movie-card:hover { transform: translateY(-10px) scale(1.035) rotateX(2deg) rotateY(-2deg); }

section[data-testid="stSidebar"] label:has(input:checked) {
    background: rgba(229,9,20,0.15);
    border-radius: 8px;
    box-shadow: inset 3px 0 0 #e50914;
}

.stButton > button {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 20px rgba(229,9,20,0.35);
}

/* Leakage warning banner */
.leakage-warning {
    background: linear-gradient(135deg, #8b0000, #e50914);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    border: 1px solid rgba(255,100,100,0.4);
    animation: fadeInUp 0.5s ease both;
}
.leakage-warning h4 {
    margin: 0 0 8px 0;
    color: #fff;
    font-size: 1.1rem;
}
.leakage-warning p {
    margin: 0;
    color: #ffcccc;
    font-size: 0.9rem;
}

/* Fair mode badge */
.fair-badge {
    display: inline-block;
    background: linear-gradient(135deg, #006400, #228b22);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 8px;
}

div.block-container { padding-top: 1.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Floating ember particles
import random as _random
_rng = _random.Random(42)
_particles_html = "<div class='particle-field'>"
for _i in range(28):
    left = _rng.uniform(0, 100)
    duration = _rng.uniform(6, 16)
    delay = _rng.uniform(0, 12)
    size = _rng.uniform(3, 7)
    _particles_html += (
        f"<div class='particle' style='left:{left:.1f}%; width:{size:.1f}px; height:{size:.1f}px; "
        f"animation-duration:{duration:.1f}s; animation-delay:{delay:.1f}s;'></div>"
    )
_particles_html += "</div>"
st.markdown(_particles_html, unsafe_allow_html=True)


# ==============================================================================
# 4. DATA LOADING
# ==============================================================================
@st.cache_data(show_spinner="Loading Netflix dataset...")
def load_data() -> pd.DataFrame:
    if os.path.exists(CLEAN_CSV):
        df = pd.read_csv(CLEAN_CSV)
    else:
        df = pd.read_csv(RAW_CSV)

    df["director"] = df["director"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("")
    df["rating"] = df["rating"].fillna("Not Rated")
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# 5. RECOMMENDATION ENGINE (FIXED: Full dataset + better features)
# ==============================================================================
@st.cache_resource(show_spinner="Building recommendation engine (TF-IDF)...")
def build_recommender(df: pd.DataFrame):
    work = df.copy().reset_index(drop=True)

    # FIXED: Richer content features including cast and description
    work["director"] = work["director"].fillna("Unknown")
    work["cast"] = work.get("cast", pd.Series([""] * len(work))).fillna("")
    work["description"] = work.get("description", pd.Series([""] * len(work))).fillna("")
    
    work["content_features"] = (
        work["listed_in"].astype(str) + " " +
        work["director"].astype(str) + " " +
        work["cast"].astype(str) + " " +
        work["country"].astype(str) + " " +
        work["description"].astype(str).str[:200]
    )
    work["content_features_clean"] = work["content_features"].apply(clean_text)

    # FIXED: Full dataset, no sampling
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(work["content_features_clean"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(work.index, index=work["title"]).drop_duplicates()

    return work, cosine_sim, indices


def get_recommendations(title: str, work: pd.DataFrame, cosine_sim, indices, top_n: int = 8):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    rec_idx = [i for i, _ in sim_scores]
    scores = [round(s, 3) for _, s in sim_scores]
    result = work.iloc[rec_idx][["title", "type", "listed_in", "release_year", "country", "director"]].copy()
    result["similarity"] = scores
    return result.reset_index(drop=True)


# ==============================================================================
# 6. TYPE CLASSIFIER (FIXED: Leakage-aware with toggles)
# ==============================================================================
@st.cache_resource(show_spinner="Training classifier...")
def train_classifier(df: pd.DataFrame, use_duration: bool = True, use_director_known: bool = True):
    work = df.copy()
    work["primary_genre"] = work["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
    work["primary_country"] = work["country"].apply(lambda x: str(x).split(",")[0].strip())
    work["director_known"] = (work["director"] != "Unknown").astype(int)

    if "duration_value" not in work.columns:
        work["duration_value"] = pd.to_numeric(
            work["duration"].str.extract(r"(\d+)")[0], errors="coerce"
        )
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

    encoders = {}
    for col in feature_cols_cat:
        le = LabelEncoder()
        work[col + "_enc"] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    work["type_enc"] = target_le.fit_transform(work["type"])

    feature_cols = [c + "_enc" for c in feature_cols_cat] + feature_cols_num
    X = work[feature_cols]
    y = work["type_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    meta = {
        "encoders": encoders,
        "target_le": target_le,
        "feature_cols": feature_cols,
        "feature_cols_cat": feature_cols_cat,
        "feature_cols_num": feature_cols_num,
        "use_duration": use_duration,
        "use_director_known": use_director_known,
        "test_accuracy": test_acc,
        "importances": importances,
        "genre_options": sorted(work["primary_genre"].unique().tolist()),
        "country_options": sorted(work["primary_country"].unique().tolist()),
        "rating_options": sorted(work["rating"].astype(str).unique().tolist()),
    }
    return model, meta


def safe_encode(le: LabelEncoder, value: str) -> int:
    if value in le.classes_:
        return int(le.transform([value])[0])
    return int(le.transform([le.classes_[0]])[0])


# ==============================================================================
# 7. VISUAL HELPERS
# ==============================================================================
PALETTE = [
    "linear-gradient(135deg,#8b0000,#e50914)",
    "linear-gradient(135deg,#141414,#e50914)",
    "linear-gradient(135deg,#4a0000,#ff5c5c)",
    "linear-gradient(135deg,#0d0d0d,#b0060f)",
    "linear-gradient(135deg,#2b0a0a,#ff2e2e)",
    "linear-gradient(135deg,#3a0000,#ff7a7a)",
]


def card_gradient(seed: str) -> str:
    return PALETTE[abs(hash(seed)) % len(PALETTE)]


def render_movie_card(row, delay_index: int = 0):
    genres = str(row["listed_in"]).split(",")
    genre_line = ", ".join(g.strip() for g in genres[:2])
    sim_pct = int(round(float(row.get("similarity", 0)) * 100))
    grad = card_gradient(row["title"])
    st.markdown(
        f"""
        <div class="movie-card" style="background:{grad}; animation-delay:{delay_index*0.06}s;">
            <span class="badge">{row['type']}</span>
            <h4>{row['title']}</h4>
            <div class="genre-line">{genre_line} &middot; {int(row['release_year'])}</div>
            <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{sim_pct}%;"></div></div>
            <div class="genre-line" style="margin-top:4px;">{sim_pct}% match</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 8. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown(
    "<h2 style='color:#e50914; text-shadow:0 0 12px rgba(229,9,20,0.6);'>🎬 NETFLIX DASH</h2>",
    unsafe_allow_html=True,
)
PAGE = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🎥 Movie Recommender",
        "📊 EDA Explorer",
        "📈 Trend Forecast",
        "🤖 Type Predictor",
        "💼 Business Dashboard",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data Science Internship -- Auspify Technologies\n\n"
    "Built by Talha Akbar. v2.0 - Data Leakage Fixed"
)

# Load data once
df = load_data()


# ==============================================================================
# 9. PAGE: HOME
# ==============================================================================
if PAGE == "🏠 Home":
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="blob blob1"></div>
            <div class="blob blob2"></div>
            <div class="blob blob3"></div>
            <p class="hero-title">Netflix Analytics &amp; ML Dashboard</p>
            <p class="hero-sub typewriter">
                Explore {len(df):,} titles &middot; get recommendations &middot; predict with FAIR ML
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Leakage warning banner
    st.markdown("""
    <div class="leakage-warning">
        <h4>🔒 Data Leakage Fixed in v2.0</h4>
        <p>This dashboard now detects and controls for data leakage in the classification model. 
        Use the <b>Type Predictor</b> page to see the difference between full and fair modes.</p>
    </div>
    """, unsafe_allow_html=True)

    ticker_titles = df["title"].dropna().sample(min(25, len(df)), random_state=7).tolist()
    ticker_html = "".join(f"<span>🎬 {t}</span>" for t in ticker_titles)
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-move">{ticker_html}</div></div>',
        unsafe_allow_html=True,
    )

    n_movies = int((df["type"] == "Movie").sum())
    n_shows = int((df["type"] == "TV Show").sum())
    n_countries = df["country"].apply(lambda x: str(x).split(",")[0].strip()).nunique()
    year_span = f"{int(df['release_year'].min())}-{int(df['release_year'].max())}"

    stat_cols = st.columns(5)
    stats = [
        ("🎞️", f"{len(df):,}", "Total Titles"),
        ("🎬", f"{n_movies:,}", "Movies"),
        ("📺", f"{n_shows:,}", "TV Shows"),
        ("🌍", f"{n_countries:,}", "Countries"),
        ("📅", year_span, "Year Range"),
    ]
    for col, (icon, number, label) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div class="stat-number">{number}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown('<div class="section-title">Content mix by type</div>', unsafe_allow_html=True)
    type_counts = df["type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig = px.pie(
        type_counts, names="type", values="count", hole=0.55,
        color="type", color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"},
    )
    fig.update_traces(textfont_size=14, pull=[0.03, 0.03])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", legend=dict(orientation="h"), margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "👈 Use the sidebar to explore. The **Type Predictor** now shows "
        "data leakage effects with a Fair Mode toggle!"
    )


# ==============================================================================
# 10. PAGE: MOVIE RECOMMENDER
# ==============================================================================
elif PAGE == "🎥 Movie Recommender":
    st.markdown('<div class="section-title">🎥 Find something similar to watch</div>', unsafe_allow_html=True)
    st.write(
        "Pick a title you like and the engine will find the closest matches using TF-IDF "
        "text similarity. **FIXED**: Now uses the FULL dataset and richer content features "
        "(genre, director, cast, country, description)."
    )

    work, cosine_sim, indices = build_recommender(df)
    all_titles = sorted(indices.index.tolist())

    if "title_selectbox" not in st.session_state:
        st.session_state["title_selectbox"] = (
            "Dick Johnson Is Dead" if "Dick Johnson Is Dead" in all_titles else all_titles[0]
        )

    col_search, col_n, col_dice = st.columns([3, 1, 1])
    with col_n:
        top_n = st.slider("How many recommendations?", 4, 12, 8)
    with col_dice:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise me", use_container_width=True):
            st.session_state["title_selectbox"] = _random.choice(all_titles)
            st.rerun()
    with col_search:
        selected_title = st.selectbox(
            "🔎 Search a title (type to filter)", all_titles, key="title_selectbox",
        )

    if selected_title:
        seed_row = work.loc
