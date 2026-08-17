"""
================================================================================
 NETFLIX ANALYTICS & ML DASHBOARD -- STREAMLIT UI
 Auspify Technologies - Data Science Internship (Talha Akbar)
================================================================================

WHAT THIS FILE DOES
--------------------
Interactive web dashboard with SIX pages:

    1. Home               -> animated hero banner + live dataset stats
    2. Movie Recommender  -> content-based recommender using TF-IDF
    3. EDA Explorer       -> interactive charts from cleaned data
    4. Trend Forecast     -> trend analysis + forecast
    5. Type Predictor     -> Movie vs TV Show classifier (leakage-free, balanced)
    6. Business Dashboard -> business insights

v3.0 CHANGES
------------
- DATA LEAKAGE FULLY REMOVED: `duration_value` and `director_known` are no
  longer used as model features anywhere in this file (not even behind a
  toggle). They were data-collection artifacts, not real signal:
  Netflix's own catalog simply forgets to list directors for ~91% of TV
  Shows, and duration units differ by type (minutes vs seasons) -- neither
  fact would be knowable in advance for a title that doesn't exist yet.
- CLASS IMBALANCE FIXED: Movies (6,126) outnumber TV Shows (2,664) about
  2.3:1 in the raw data, which used to bias every prediction toward
  "Movie". The training set is now rebalanced (minority-class oversampling
  + class_weight="balanced") so both classes get a fair vote.
- Honest accuracy is computed live and shown everywhere a number is shown.
- Small animated fairy guide ("Nova") docked in the sidebar: speaks an
  English welcome + a guided walkthrough of whichever page is open, using
  the browser's built-in speech engine.
- More ambient motion: drifting emoji layer, glow rings, animated arrow.

PATHS
------
All paths are relative to THIS file's folder using os.path.dirname(__file__).
No hardcoded paths needed.

SCOPE OF THIS EDIT: only this file (app.py) was modified. No other project
file was touched.
================================================================================
"""

import os
import re
import random as _random
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


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
        st.info(
            f"Chart not generated yet: `{os.path.relpath(path, BASE_DIR)}`. "
            f"Run `python3 main.py` once to generate it."
        )


# ==============================================================================
# 3. GLOBAL CSS - Netflix dark theme + motion
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
    transform-style: preserve-3d;
}
.movie-card:hover {
    transform: translateY(-10px) scale(1.035) rotateX(2deg) rotateY(-2deg);
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

/* Drifting emoji layer -- extra ambient motion requested by user */
.emoji-field {
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
}
.emoji-drift {
    position: absolute;
    bottom: -60px;
    opacity: 0;
    animation: driftUp linear infinite;
    filter: drop-shadow(0 0 6px rgba(229,9,20,0.35));
}
@keyframes driftUp {
    0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    8%   { opacity: 0.35; }
    50%  { transform: translateY(-55vh) translateX(30px) rotate(180deg); }
    92%  { opacity: 0.3; }
    100% { transform: translateY(-115vh) translateX(-20px) rotate(360deg); opacity: 0; }
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

/* Leakage warning / fixed banners */
.leakage-warning {
    background: linear-gradient(135deg, #8b0000, #e50914);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    border: 1px solid rgba(255,100,100,0.4);
    animation: fadeInUp 0.5s ease both;
}
.leakage-warning h4 { margin: 0 0 8px 0; color: #fff; font-size: 1.1rem; }
.leakage-warning p { margin: 0; color: #ffcccc; font-size: 0.9rem; }

.fixed-banner {
    background: linear-gradient(135deg, #063d1e, #0c7a3e);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    border: 1px solid rgba(120,255,170,0.4);
    animation: fadeInUp 0.5s ease both;
}
.fixed-banner h4 { margin: 0 0 8px 0; color: #fff; font-size: 1.1rem; }
.fixed-banner p { margin: 0; color: #d6ffe6; font-size: 0.9rem; }

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

# Drifting emoji layer (movie/tv themed) -- more moving objects across the app
_emoji_pool = ["🎬", "🍿", "📺", "🎞️", "⭐", "🎥"]
_emoji_html = "<div class='emoji-field'>"
for _i in range(14):
    left = _rng.uniform(0, 100)
    duration = _rng.uniform(14, 26)
    delay = _rng.uniform(0, 20)
    size = _rng.uniform(14, 26)
    emoji = _rng.choice(_emoji_pool)
    _emoji_html += (
        f"<div class='emoji-drift' style='left:{left:.1f}%; font-size:{size:.0f}px; "
        f"animation-duration:{duration:.1f}s; animation-delay:{delay:.1f}s;'>{emoji}</div>"
    )
_emoji_html += "</div>"
st.markdown(_emoji_html, unsafe_allow_html=True)


# ==============================================================================
# 3B. NOVA -- the animated fairy tour-guide (voice + pointing, English)
# ==============================================================================
def render_nova_guide(current_page: str):
    """
    Small girl-fairy mascot docked in the sidebar. Uses the browser's built-in
    Speech Synthesis engine (no external API / no extra dependency) to:
      1) speak a one-time English welcome + how-to-use explanation
      2) on demand, explain + "point at" whichever page is currently open
    Runs inside its own sandboxed component iframe -- entirely self-contained,
    does not touch any other project file.
    """
    tour_scripts = {
        "🏠 Home": (
            "You're on the Home page. Up here, those glowing cards show live totals: "
            "how many movies, how many TV shows, how many countries. "
            "Use the menu above me, on the left, to jump to any other page."
        ),
        "🎥 Movie Recommender": (
            "This is the Movie Recommender. Pick any title from the search box at the top, "
            "and I'll find titles with similar genre, director, cast and country, "
            "using text similarity. Try the Surprise Me dice button for a random pick!"
        ),
        "📊 EDA Explorer": (
            "Welcome to the EDA Explorer. These interactive charts let you slice the catalog "
            "by genre, country, rating and year. Hover over any chart to see exact numbers."
        ),
        "📈 Trend Forecast": (
            "This is the Trend Forecast page. The line chart shows how many titles were added "
            "each year, split by movies and TV shows, plus a short forecast for the next few years."
        ),
        "🤖 Type Predictor": (
            "Here's the Type Predictor, my favorite! Fill in a genre, country, rating and year "
            "in the form on the right, then press Predict. I only use fair, honest features now, "
            "the data leakage from before has been completely removed, and the training data "
            "was rebalanced so movies don't unfairly dominate the prediction."
        ),
        "💼 Business Dashboard": (
            "This is the Business Dashboard. It summarizes the big picture: top genres, top markets, "
            "and content strategy recommendations, all in one place."
        ),
    }
    welcome_line = (
        "Hi! I'm Nova, your guide for this dashboard. "
        "Pick a page from the menu above, then press the Guide Me button below and "
        "I'll explain what's on it, out loud."
    )
    page_line = tour_scripts.get(current_page, welcome_line)

    html = f"""
    <div id="nova-wrap" style="font-family: 'Segoe UI', sans-serif;">
      <style>
        #nova-wrap {{ position: relative; text-align:center; padding: 6px 4px 2px 4px; }}
        #nova-bubble {{
            background: rgba(229,9,20,0.12);
            border: 1px solid rgba(229,9,20,0.4);
            border-radius: 12px;
            padding: 8px 10px;
            font-size: 11.5px;
            line-height: 1.35;
            color: #f5e6e6;
            margin-bottom: 6px;
            min-height: 34px;
            text-align: left;
        }}
        @keyframes novaFloat {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-7px); }}
        }}
        @keyframes wingFlap {{
            0%, 100% {{ transform: rotate(-18deg) scaleY(1); }}
            50% {{ transform: rotate(-38deg) scaleY(0.85); }}
        }}
        @keyframes wingFlapR {{
            0%, 100% {{ transform: rotate(18deg) scaleY(1); }}
            50% {{ transform: rotate(38deg) scaleY(0.85); }}
        }}
        @keyframes sparkleTwinkle {{
            0%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
            50%      {{ opacity: 1;   transform: scale(1.15); }}
        }}
        #nova-sprite {{ animation: novaFloat 3s ease-in-out infinite; }}
        #nova-wing-l {{ transform-origin: 60% 50%; animation: wingFlap 1.1s ease-in-out infinite; }}
        #nova-wing-r {{ transform-origin: 40% 50%; animation: wingFlapR 1.1s ease-in-out infinite; }}
        .nova-sparkle {{ animation: sparkleTwinkle 1.6s ease-in-out infinite; }}
        #nova-btn-row {{ display:flex; gap:6px; justify-content:center; margin-top:4px; }}
        .nova-btn {{
            background: linear-gradient(135deg,#8b0000,#e50914);
            color:white; border:none; border-radius:20px;
            padding:5px 10px; font-size:11px; font-weight:700; cursor:pointer;
            box-shadow: 0 3px 10px rgba(229,9,20,0.4);
        }}
        .nova-btn:hover {{ filter: brightness(1.15); }}
        .nova-btn.stop {{ background: linear-gradient(135deg,#3a3a3a,#5a5a5a); }}
      </style>

      <div id="nova-bubble">✨ Hi!</div>

      <svg id="nova-sprite" width="86" height="96" viewBox="0 0 86 96">
        <ellipse class="nova-sparkle" cx="10" cy="20" rx="2.2" ry="2.2" fill="#ffe9b0"/>
        <ellipse class="nova-sparkle" cx="76" cy="30" rx="1.6" ry="1.6" fill="#ffe9b0" style="animation-delay:.4s"/>
        <ellipse class="nova-sparkle" cx="14" cy="70" rx="1.6" ry="1.6" fill="#ffe9b0" style="animation-delay:.9s"/>

        <!-- wings -->
        <ellipse id="nova-wing-l" cx="26" cy="46" rx="16" ry="9" fill="rgba(255,255,255,0.55)" stroke="#ffd7d7" stroke-width="1"/>
        <ellipse id="nova-wing-r" cx="60" cy="46" rx="16" ry="9" fill="rgba(255,255,255,0.55)" stroke="#ffd7d7" stroke-width="1"/>

        <!-- dress / body -->
        <path d="M43 50 L58 84 Q43 92 28 84 Z" fill="#e50914"/>
        <path d="M43 50 L58 84 Q43 92 28 84 Z" fill="url(#dressGrad)"/>
        <defs>
          <linearGradient id="dressGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#ff4d4d"/>
            <stop offset="100%" stop-color="#8b0000"/>
          </linearGradient>
        </defs>

        <!-- arms -->
        <line x1="34" y1="56" x2="20" y2="66" stroke="#ffd9b3" stroke-width="4" stroke-linecap="round"/>
        <line x1="52" y1="56" x2="66" y2="50" stroke="#ffd9b3" stroke-width="4" stroke-linecap="round"/>
        <!-- little wand -->
        <line x1="66" y1="50" x2="76" y2="40" stroke="#ffd700" stroke-width="2" stroke-linecap="round"/>
        <circle cx="77" cy="38" r="2.6" fill="#fff3b0"/>

        <!-- head -->
        <circle cx="43" cy="34" r="16" fill="#ffd9b3"/>
        <!-- hair pigtails -->
        <circle cx="24" cy="32" r="7" fill="#3a0d0d"/>
        <circle cx="62" cy="32" r="7" fill="#3a0d0d"/>
        <path d="M27 22 Q43 12 59 22 Q56 30 43 27 Q30 30 27 22 Z" fill="#3a0d0d"/>
        <!-- face -->
        <circle cx="37" cy="35" r="1.6" fill="#2a2a2a"/>
        <circle cx="49" cy="35" r="1.6" fill="#2a2a2a"/>
        <path d="M38 41 Q43 44 48 41" stroke="#a33" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        <circle cx="33" cy="39" r="2.4" fill="#ff8f8f" opacity="0.6"/>
        <circle cx="53" cy="39" r="2.4" fill="#ff8f8f" opacity="0.6"/>

        <!-- legs / feet (per user request: "jis k per ho" -- she has feet/legs) -->
        <line x1="37" y1="84" x2="34" y2="93" stroke="#ffd9b3" stroke-width="4" stroke-linecap="round"/>
        <line x1="49" y1="84" x2="52" y2="93" stroke="#ffd9b3" stroke-width="4" stroke-linecap="round"/>
        <ellipse cx="33" cy="94" rx="4" ry="2" fill="#3a0d0d"/>
        <ellipse cx="53" cy="94" rx="4" ry="2" fill="#3a0d0d"/>
      </svg>

      <div id="nova-btn-row">
        <button class="nova-btn" onclick="novaGuide()">🔊 Guide me here</button>
        <button class="nova-btn stop" onclick="novaStop()">🔇</button>
      </div>
    </div>

    <script>
    (function() {{
        const bubble = document.getElementById('nova-bubble');
        const pageLine = {page_line!r};
        const welcomeLine = {welcome_line!r};

        function pickVoice() {{
            const voices = window.speechSynthesis.getVoices();
            const prefer = [
                "Google UK English Female", "Google US English",
                "Microsoft Zira", "Samantha", "Female"
            ];
            for (const name of prefer) {{
                const v = voices.find(v => v.name.includes(name) && v.lang.startsWith("en"));
                if (v) return v;
            }}
            return voices.find(v => v.lang.startsWith("en")) || voices[0];
        }}

        function speak(text) {{
            try {{
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(text);
                const v = pickVoice();
                if (v) utter.voice = v;
                utter.lang = "en-US";
                utter.rate = 0.97;
                utter.pitch = 1.15;
                utter.volume = 1.0;
                window.speechSynthesis.speak(utter);
            }} catch (e) {{ /* speech not supported -- fail silently */ }}
            bubble.innerText = "✨ " + text;
        }}

        window.novaGuide = function() {{ speak(pageLine); }};
        window.novaStop = function() {{
            try {{ window.speechSynthesis.cancel(); }} catch (e) {{}}
            bubble.innerText = "✨ Okay, staying quiet. Press Guide me here anytime!";
        }};

        // one-time auto welcome per browser session
        bubble.innerText = "✨ " + welcomeLine;
        if (!sessionStorage.getItem('nova_welcomed')) {{
            sessionStorage.setItem('nova_welcomed', '1');
            setTimeout(function() {{ speak(welcomeLine); }}, 500);
        }}

        // best-effort: point a small glowing arrow at the main content area
        try {{
            const parentDoc = window.parent.document;
            let arrow = parentDoc.getElementById('nova-pointer-arrow');
            if (!arrow) {{
                arrow = parentDoc.createElement('div');
                arrow.id = 'nova-pointer-arrow';
                arrow.innerText = '👉';
                arrow.style.position = 'fixed';
                arrow.style.top = '78px';
                arrow.style.left = '340px';
                arrow.style.fontSize = '26px';
                arrow.style.zIndex = 999999;
                arrow.style.pointerEvents = 'none';
                arrow.style.transition = 'opacity 0.4s ease';
                arrow.style.filter = 'drop-shadow(0 0 6px rgba(229,9,20,0.8))';
                arrow.style.animation = 'novaArrowBob 1s ease-in-out infinite';
                const styleTag = parentDoc.createElement('style');
                styleTag.innerText = '@keyframes novaArrowBob {{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(8px)}}}}';
                parentDoc.head.appendChild(styleTag);
                parentDoc.body.appendChild(arrow);
            }}
            arrow.style.opacity = '0.85';
            clearTimeout(window.__novaArrowTimer);
            window.__novaArrowTimer = setTimeout(function() {{ arrow.style.opacity = '0'; }}, 4000);
        }} catch (e) {{ /* cross-origin or sandboxed -- ignore, mascot still fully works */ }}
    }})();
    </script>
    """
    components.html(html, height=270)


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

    if "year_added" not in df.columns:
        df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year
    df["year_added"] = df["year_added"].fillna(df["year_added"].median())

    if "duration_value" not in df.columns:
        df["duration_value"] = pd.to_numeric(
            df["duration"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    df["duration_value"] = df["duration_value"].fillna(df["duration_value"].median())

    df["primary_genre"] = df["listed_in"].apply(lambda x: str(x).split(",")[0].strip() if str(x).strip() else "Unknown")
    df["primary_country"] = df["country"].apply(lambda x: str(x).split(",")[0].strip())
    df["director_known"] = (df["director"] != "Unknown").astype(int)
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# 5. RECOMMENDATION ENGINE (full dataset, content-based TF-IDF)
# ==============================================================================
@st.cache_resource(show_spinner="Building recommendation engine (TF-IDF)...")
def build_recommender(df: pd.DataFrame):
    work = df.copy().reset_index(drop=True)

    work["director"] = work["director"].fillna("Unknown")
    work["cast"] = work["cast"] if "cast" in work.columns else pd.Series([""] * len(work))
    work["description"] = work["description"] if "description" in work.columns else pd.Series([""] * len(work))
    work["cast"] = work["cast"].fillna("")
    work["description"] = work["description"].fillna("")

    work["content_features"] = (
        work["listed_in"].astype(str) + " " +
        work["director"].astype(str) + " " +
        work["cast"].astype(str) + " " +
        work["country"].astype(str) + " " +
        work["description"].astype(str).str[:200]
    )
    work["content_features_clean"] = work["content_features"].apply(clean_text)

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
# 6. TYPE CLASSIFIER -- data-leakage FREE, class-balanced
# ==============================================================================
# WHAT CHANGED FROM v2.0
# -----------------------
# `duration_value` and `director_known` are gone completely (not a toggle --
# physically removed from the feature list). They accounted for ~86% of the
# old model's decision power and were both artifacts of how Netflix's own
# catalog happens to be recorded, not genuine signal about the title itself:
#   - director_known: 97.2% of Movies list a director vs only 9.3% of TV
#     Shows -- purely a metadata-collection habit, unrelated to content.
#   - duration_value: literally measured in different units per class
#     (minutes for Movies, seasons for TV Shows) -- the model was just
#     decoding the unit, not learning anything about the title.
#
# Remaining features are genuine, pre-release metadata a cataloguer would
# always have: genre(s), country, content rating, release year, year added.
#
# CLASS IMBALANCE FIX
# --------------------
# Movies (6,126) outnumber TV Shows (2,664) ~2.3:1 in the raw data. Left
# alone, that imbalance alone pushes a model toward predicting "Movie" by
# default. Two things are done about it:
#   1. The *training* split is rebalanced by oversampling the minority
#      class (TV Shows) up to match the majority count -- i.e. TV Show
#      examples are duplicated (with replacement) so the model sees roughly
#      equal amounts of both classes while learning.
#   2. RandomForestClassifier(class_weight="balanced") on top of that, so
#      misclassifying the minority class is penalised more heavily.
# The *test* split is left untouched (real-world proportions) so the
# reported accuracy is honest, not inflated by the balancing step.
GENRE_HINT_COLS = ["primary_genre_enc", "secondary_genre_enc", "num_genres"]


@st.cache_resource(show_spinner="Training the leakage-free classifier...")
def train_classifier(df: pd.DataFrame):
    work = df.copy()
    work["rating"] = work["rating"].fillna("Not Rated")

    genre_lists = work["listed_in"].apply(lambda x: [g.strip() for g in str(x).split(",") if g.strip()])
    work["secondary_genre"] = genre_lists.apply(lambda g: g[1] if len(g) > 1 else "None")
    work["num_genres"] = genre_lists.apply(lambda g: max(len(g), 1))
    work["num_countries"] = work["country"].apply(lambda x: len(str(x).split(",")))

    feature_cols_cat = ["primary_genre", "secondary_genre", "primary_country", "rating"]
    feature_cols_num = ["release_year", "year_added", "num_genres", "num_countries"]

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

    # --- class-balance fix: oversample minority class in TRAIN split only ---
    train_bal = X_train.copy()
    train_bal["__y"] = y_train.values
    counts = train_bal["__y"].value_counts()
    max_n = counts.max()
    parts = [train_bal]
    for cls, n in counts.items():
        if n < max_n:
            extra = train_bal[train_bal["__y"] == cls].sample(
                max_n - n, replace=True, random_state=42
            )
            parts.append(extra)
    train_bal = pd.concat(parts, ignore_index=True)
    X_train_bal = train_bal[feature_cols]
    y_train_bal = train_bal["__y"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_bal, y_train_bal)
    preds = model.predict(X_test)

    test_acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=target_le.classes_, output_dict=True
    )

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Secondary, "genre-blind" model for transparency: how much signal is
    # there WITHOUT any genre info at all (country + rating + year only)?
    blind_cols = [c for c in feature_cols if c not in GENRE_HINT_COLS]
    Xb_train, Xb_test = X_train_bal[blind_cols], X_test[blind_cols]
    blind_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    blind_model.fit(Xb_train, y_train_bal)
    blind_acc = accuracy_score(y_test, blind_model.predict(Xb_test))

    class_counts_before = y_train.value_counts().rename(index=dict(enumerate(target_le.classes_)))
    class_counts_after = y_train_bal.value_counts().rename(index=dict(enumerate(target_le.classes_)))

    meta = {
        "encoders": encoders,
        "target_le": target_le,
        "feature_cols": feature_cols,
        "feature_cols_cat": feature_cols_cat,
        "feature_cols_num": feature_cols_num,
        "test_accuracy": test_acc,
        "balanced_accuracy": bal_acc,
        "genre_blind_accuracy": blind_acc,
        "confusion_matrix": cm,
        "report": report,
        "importances": importances,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "genre_options": sorted(work["primary_genre"].unique().tolist()),
        "secondary_genre_options": sorted(work["secondary_genre"].unique().tolist()),
        "country_options": sorted(work["primary_country"].unique().tolist()),
        "rating_options": sorted(work["rating"].astype(str).unique().tolist()),
        "n_train": len(X_train),
        "n_train_balanced": len(X_train_bal),
        "n_test": len(X_test),
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


def render_spark_banner(text: str, bg: str):
    n_sparks = 10
    sparks = "".join(
        f"<span class='spark' style='--angle:{int(360*i/n_sparks)}deg; animation-delay:{i*0.03:.2f}s;'>✦</span>"
        for i in range(n_sparks)
    )
    st.markdown(
        f"""
        <div class="pred-banner glow-pulse" style="background:{bg}; position:relative;">
            <div class="spark-field">{sparks}</div>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 8. SIDEBAR NAVIGATION + NOVA GUIDE
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
render_nova_guide(PAGE)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data Science Internship -- Auspify Technologies\n\n"
    "Built by Talha Akbar. v3.0 - Leakage removed, classes balanced, meet Nova 🧚"
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

    st.markdown("""
    <div class="fixed-banner">
        <h4>✅ Data Leakage Removed + Classes Balanced in v3.0</h4>
        <p>The old model leaned on <b>duration</b> and <b>director-listed</b> -- two data-collection
        artifacts, not real signal -- for 86% of its decision power, and the training data itself
        was skewed ~2.3:1 toward Movies. Both are fixed now: those two features are gone for good,
        and the training set is rebalanced so TV Shows get a fair vote too.
        See the <b>Type Predictor</b> page for the full before/after.</p>
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
        "👈 Use the sidebar to explore -- and say hi to **Nova**, your guide, right below the menu! "
        "Ask her to Guide me here on any page."
    )


# ==============================================================================
# 10. PAGE: MOVIE RECOMMENDER
# ==============================================================================
elif PAGE == "🎥 Movie Recommender":
    st.markdown('<div class="section-title">🎥 Find something similar to watch</div>', unsafe_allow_html=True)
    st.write(
        "Pick a title you like and the engine will find the closest matches using TF-IDF "
        "text similarity across genre, director, cast, country and description."
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
        seed_idx = indices[selected_title]
        seed_row = work.iloc[seed_idx]

        st.markdown(
            f"""
            <div class="movie-card" style="background:{card_gradient(selected_title)}; height:150px;">
                <span class="badge">{seed_row['type']}</span>
                <h4 style="font-size:1.3rem;">🎯 {seed_row['title']}</h4>
                <div class="genre-line">{str(seed_row['listed_in'])[:70]} &middot; {int(seed_row['release_year'])}</div>
                <div class="genre-line">📍 {seed_row['country']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="section-title">Because you liked that...</div>', unsafe_allow_html=True)

        recs = get_recommendations(selected_title, work, cosine_sim, indices, top_n)
        if recs.empty:
            st.warning("No recommendations found for this title -- try another one.")
        else:
            n_cols = 4
            rows = [recs.iloc[i:i + n_cols] for i in range(0, len(recs), n_cols)]
            counter = 0
            for row_chunk in rows:
                cols = st.columns(n_cols)
                for col, (_, rec_row) in zip(cols, row_chunk.iterrows()):
                    with col:
                        render_movie_card(rec_row, delay_index=counter)
                    counter += 1


# ==============================================================================
# 11. PAGE: EDA EXPLORER
# ==============================================================================
elif PAGE == "📊 EDA Explorer":
    st.markdown('<div class="section-title">📊 Explore the catalog</div>', unsafe_allow_html=True)
    st.write("Interactive charts computed live from the cleaned dataset.")

    tab_genre, tab_geo, tab_rating, tab_time, tab_duration = st.tabs(
        ["🎭 Genres", "🌍 Countries", "🔞 Ratings", "🗓️ Added over time", "⏱️ Duration"]
    )

    with tab_genre:
        top_n_genre = st.slider("How many top genres?", 5, 20, 10, key="genre_slider")
        genre_series = df["primary_genre"].value_counts().head(top_n_genre).reset_index()
        genre_series.columns = ["genre", "count"]
        fig = px.bar(
            genre_series.sort_values("count"), x="count", y="genre", orientation="h",
            color="count", color_continuous_scale=["#4a0000", "#e50914"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_geo:
        top_n_country = st.slider("How many top countries?", 5, 20, 10, key="country_slider")
        country_series = df["primary_country"].value_counts().head(top_n_country).reset_index()
        country_series.columns = ["country", "count"]
        fig = px.bar(
            country_series.sort_values("count"), x="count", y="country", orientation="h",
            color="count", color_continuous_scale=["#141414", "#e50914"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        fig_map = px.choropleth(
            country_series, locations="country", locationmode="country names",
            color="count", color_continuous_scale=["#2b0a0a", "#e50914"],
        )
        fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               geo=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10))
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_rating:
        rating_by_type = df.groupby(["rating", "type"]).size().reset_index(name="count")
        fig = px.bar(
            rating_by_type, x="rating", y="count", color="type", barmode="group",
            color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Notice MPAA ratings (G, PG, PG-13, R, NC-17) are almost exclusively Movies, while "
            "TV-* ratings split across both -- that's a genuine, legitimate signal (Netflix's own "
            "rating system differs by content type), not a data artifact."
        )

    with tab_time:
        added_by_year = df.groupby(["year_added", "type"]).size().reset_index(name="count")
        fig = px.area(
            added_by_year, x="year_added", y="count", color="type",
            color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_duration:
        col_a, col_b = st.columns(2)
        with col_a:
            movie_dur = df[df["type"] == "Movie"]["duration_value"]
            fig = px.histogram(movie_dur, nbins=40, color_discrete_sequence=["#e50914"])
            fig.update_layout(title="Movie duration (minutes)", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            show_dur = df[df["type"] == "TV Show"]["duration_value"]
            fig = px.histogram(show_dur, nbins=15, color_discrete_sequence=["#ffffff"])
            fig.update_layout(title="TV Show duration (seasons)", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This is exactly why duration was removed from the Type Predictor: it isn't comparable "
            "across classes (minutes vs seasons), so a model 'using' it is really just decoding units."
        )

    with st.expander("📄 Original EDA summary (from task2)"):
        st.text(read_text_file(os.path.join(TASK2_DIR, "eda_summary.txt")))


# ==============================================================================
# 12. PAGE: TREND FORECAST
# ==============================================================================
elif PAGE == "📈 Trend Forecast":
    st.markdown('<div class="section-title">📈 Content growth &amp; forecast</div>', unsafe_allow_html=True)
    st.write("How many titles were released each year, and a simple trend projection forward.")

    yearly = df.groupby(["release_year", "type"]).size().reset_index(name="count")
    yearly = yearly[yearly["release_year"] >= 2000]

    fig = px.line(
        yearly, x="release_year", y="count", color="type", markers=True,
        color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Next 3 years (simple linear projection)</div>', unsafe_allow_html=True)
    total_by_year = df[df["release_year"] >= 2000].groupby("release_year").size().reset_index(name="count")
    x = total_by_year["release_year"].values
    y = total_by_year["count"].values
    if len(x) >= 3:
        coeffs = np.polyfit(x, y, 1)
        future_years = np.arange(x.max() + 1, x.max() + 4)
        future_pred = np.clip(np.polyval(coeffs, future_years), 0, None).round().astype(int)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Historical", line=dict(color="#e50914")))
        fig2.add_trace(go.Scatter(
            x=future_years, y=future_pred, mode="lines+markers", name="Forecast",
            line=dict(color="#ffffff", dash="dash"),
        ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="white", margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

        forecast_cols = st.columns(len(future_years))
        for col, yr, pred in zip(forecast_cols, future_years, future_pred):
            with col:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-number">{pred:,}</div>
                    <div class="stat-label">{int(yr)} (proj.)</div></div>""",
                    unsafe_allow_html=True,
                )
        st.caption(
            "A plain linear trend -- treat this as a rough baseline, not a guarantee. Real catalog "
            "growth depends on licensing deals and regional strategy that a straight line can't capture."
        )

    with st.expander("📄 Original forecast report (from task4)"):
        st.text(read_text_file(os.path.join(TASK4_DIR, "trend_forecast_report.txt")))


# ==============================================================================
# 13. PAGE: TYPE PREDICTOR (leakage-free + class-balanced)
# ==============================================================================
elif PAGE == "🤖 Type Predictor":
    st.markdown('<div class="section-title">🤖 Movie or TV Show?</div>', unsafe_allow_html=True)
    st.write(
        "Fill in a few honest, pre-release details and the model predicts the content type. "
        "No duration, no director-listed flag -- those were data leakage and are gone for good."
    )

    model, meta = train_classifier(df)

    acc_col1, acc_col2, acc_col3 = st.columns(3)
    with acc_col1:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-number">{meta['test_accuracy']*100:.1f}%</div>
            <div class="stat-label">Honest test accuracy</div></div>""", unsafe_allow_html=True,
        )
    with acc_col2:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-number">{meta['balanced_accuracy']*100:.1f}%</div>
            <div class="stat-label">Balanced accuracy</div></div>""", unsafe_allow_html=True,
        )
    with acc_col3:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-number">{meta['genre_blind_accuracy']*100:.1f}%</div>
            <div class="stat-label">Accuracy w/o genre (country+rating+year only)</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("🕵️ See the data-leakage evidence (before vs after)"):
        director_stats = df.groupby("type")["director_known"].mean().mul(100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                director_stats.reset_index(), x="type", y="director_known",
                color="type", color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"},
                labels={"director_known": "% with a director listed"},
                title="Old leaky feature: 'director listed'",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"{director_stats.get('Movie', 0):.1f}% of Movies list a director vs only "
                f"{director_stats.get('TV Show', 0):.1f}% of TV Shows -- purely how Netflix "
                f"records its catalog, unrelated to the actual content."
            )
        with c2:
            before = meta["class_counts_before"]
            after = meta["class_counts_after"]
            bal_df = pd.DataFrame({
                "type": list(before.index) + list(after.index),
                "count": list(before.values) + list(after.values),
                "stage": ["Before balancing"] * len(before) + ["After balancing"] * len(after),
            })
            fig2 = px.bar(
                bal_df, x="type", y="count", color="stage", barmode="group",
                color_discrete_map={"Before balancing": "#8b0000", "After balancing": "#22c55e"},
                title="Class balance fix (training set only)",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="white", margin=dict(t=40))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Training set was {meta['n_train']:,} rows (Movie-heavy). After oversampling the "
                f"minority class it's {meta['n_train_balanced']:,} rows, split evenly. "
                f"The {meta['n_test']:,}-row test set is left untouched for honest scoring."
            )

    st.markdown('<div class="section-title">Feature importance (current, fair model)</div>', unsafe_allow_html=True)
    imp_df = meta["importances"].rename("importance").rename_axis("feature").reset_index()
    fig_imp = px.bar(
        imp_df,
        x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#141414", "#e50914"],
    )
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.write("")
    st.markdown('<div class="section-title">Try a prediction</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        f1, f2 = st.columns(2)
        with f1:
            genre = st.selectbox("Primary genre", meta["genre_options"])
            secondary_genre = st.selectbox("Secondary genre (optional)", meta["secondary_genre_options"])
            country = st.selectbox("Primary country", meta["country_options"])
        with f2:
            rating = st.selectbox("Content rating", meta["rating_options"])
            release_year = st.slider("Release year", 1940, datetime.now().year, 2020)
            year_added = st.slider("Year added to Netflix", 2008, datetime.now().year, datetime.now().year)
        submitted = st.form_submit_button("🎬 Predict", use_container_width=True)

    if submitted:
        enc = meta["encoders"]
        row = {
            "primary_genre_enc": safe_encode(enc["primary_genre"], genre),
            "secondary_genre_enc": safe_encode(enc["secondary_genre"], secondary_genre),
            "primary_country_enc": safe_encode(enc["primary_country"], country),
            "rating_enc": safe_encode(enc["rating"], rating),
            "release_year": release_year,
            "year_added": year_added,
            "num_genres": 2 if secondary_genre != "None" else 1,
            "num_countries": 1,
        }
        X_input = pd.DataFrame([row])[meta["feature_cols"]]
        pred_enc = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        pred_label = meta["target_le"].inverse_transform([pred_enc])[0]
        confidence = float(proba[pred_enc])

        bg = "linear-gradient(135deg,#8b0000,#e50914)" if pred_label == "Movie" else "linear-gradient(135deg,#141414,#3a3a3a)"
        icon = "🎬" if pred_label == "Movie" else "📺"
        render_spark_banner(f"{icon} Predicted: <b>{pred_label}</b> &middot; {confidence*100:.1f}% confidence", bg)

        prob_df = pd.DataFrame({
            "type": meta["target_le"].classes_,
            "probability": proba,
        })
        fig_p = px.bar(
            prob_df, x="type", y="probability", color="type",
            color_discrete_map={"Movie": "#e50914", "TV Show": "#ffffff"}, range_y=[0, 1],
        )
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="white", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_p, use_container_width=True)

        if rating in ("G", "PG", "PG-13", "R", "NC-17", "UR") and secondary_genre != "None" and "TV" in secondary_genre:
            st.info(
                "Heads up: that rating is almost exclusively used for Movies in this catalog, while "
                "the genre you picked looks TV-flavoured -- that combination is genuinely rare/ambiguous "
                "in the real data, so don't be surprised by a lower-confidence call here."
            )

    st.write("")
    with st.expander("📊 Confusion matrix & classification report"):
        cm = meta["confusion_matrix"]
        labels = meta["target_le"].classes_
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale=["#141414", "#e50914"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=10))
        st.plotly_chart(fig_cm, use_container_width=True)

        report_df = pd.DataFrame(meta["report"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ==============================================================================
# 14. PAGE: BUSINESS DASHBOARD
# ==============================================================================
elif PAGE == "💼 Business Dashboard":
    st.markdown('<div class="section-title">💼 Business insights</div>', unsafe_allow_html=True)

    n_movies = int((df["type"] == "Movie").sum())
    n_shows = int((df["type"] == "TV Show").sum())
    movie_pct = n_movies / len(df) * 100

    kpi_cols = st.columns(4)
    top_country = df["primary_country"].value_counts().idxmax()
    top_rating = df["rating"].value_counts().idxmax()
    kpis = [
        ("🎞️", f"{len(df):,}", "Total titles"),
        ("⚖️", f"{movie_pct:.0f}% / {100-movie_pct:.0f}%", "Movie / TV split"),
        ("🌍", top_country, "Top market"),
        ("🔞", top_rating, "Most common rating"),
    ]
    for col, (icon, number, label) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"""<div class="stat-card"><div style="font-size:1.4rem;">{icon}</div>
                <div class="stat-number" style="font-size:1.5rem;">{number}</div>
                <div class="stat-label">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="section-title">Top 5 markets</div>', unsafe_allow_html=True)
        top5_country = df["primary_country"].value_counts().head(5).reset_index()
        top5_country.columns = ["country", "count"]
        fig = px.bar(top5_country, x="country", y="count", color="count",
                      color_continuous_scale=["#141414", "#e50914"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        st.markdown('<div class="section-title">Top 5 genres</div>', unsafe_allow_html=True)
        top5_genre = df["primary_genre"].value_counts().head(5).reset_index()
        top5_genre.columns = ["genre", "count"]
        fig = px.bar(top5_genre, x="genre", y="count", color="count",
                      color_continuous_scale=["#4a0000", "#e50914"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
    missing_director_pct = (df["director"] == "Unknown").mean() * 100
    st.markdown(f"""
    - **Content mix**: Movies are {movie_pct:.0f}% of the catalog. TV Shows tend to drive longer
      subscriber engagement industry-wide, so continued TV Show investment is worth prioritising.
    - **Market focus**: **{top_country}** leads content volume -- underrepresented but populous
      markets could be good targets for regional originals to diversify the catalog.
    - **Ratings strategy**: **{top_rating}** is the most common rating, indicating a mature-audience
      skew; family-friendly content is comparatively under-represented.
    - **Data quality**: {missing_director_pct:.0f}% of titles are missing director metadata --
      improving this would help recommendation quality (note: this is exactly why the ML model in
      this dashboard no longer relies on that field to predict content type).
    """)

    with st.expander("📄 Original business insights report (from task6)"):
        st.text(read_text_file(os.path.join(TASK6_DIR, "business_insights_report.txt")))
    with st.expander("🖼️ Original static dashboard image (from task6)"):
        show_image_if_exists(os.path.join(TASK6_DIR, "business_dashboard.png"))
