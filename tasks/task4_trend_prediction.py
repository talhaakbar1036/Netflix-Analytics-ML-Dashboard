"""
Task 4 (Medium) - Trend Prediction Analysis
Auspify Technologies - Data Science Internship

Goal: Analyze historical content trends and forecast future content growth.

Workflow:
 Step 1: Prepare release-year data.
 Step 2: Analyze yearly content trends.
 Step 3: Build forecasting models.
 Step 4: Visualize future predictions.
 Step 5: Interpret results.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task4")

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
    print("TASK 4: TREND PREDICTION ANALYSIS")
    print("=" * 70)

    df = load_data()

    # ------------------------------------------------------------------
    # Step 1: Prepare release-year data
    # ------------------------------------------------------------------
    yearly = df.groupby("release_year").size().reset_index(name="title_count")
    yearly = yearly[(yearly["release_year"] >= 2000) & (yearly["release_year"] <= 2021)]
    yearly = yearly.sort_values("release_year").reset_index(drop=True)
    print(f"\n[Step 1] Yearly release counts prepared (2000-2021): {len(yearly)} data points")
    print(yearly.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 2: Analyze yearly content trends
    # ------------------------------------------------------------------
    yearly["pct_change"] = yearly["title_count"].pct_change() * 100
    growth_years = yearly[yearly["pct_change"] > 0]
    print(f"\n[Step 2] Average year-over-year growth: {yearly['pct_change'].mean():.1f}%")
    print(f"Peak release year: {yearly.loc[yearly['title_count'].idxmax(), 'release_year']:.0f} "
          f"with {yearly['title_count'].max()} titles")

    # Also split trend by content type
    yearly_by_type = (
        df[(df["release_year"] >= 2000) & (df["release_year"] <= 2021)]
        .groupby(["release_year", "type"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nTrend by content type (last 5 years):")
    print(yearly_by_type.tail(5).to_string())

    # ------------------------------------------------------------------
    # Step 3: Build forecasting models
    # ------------------------------------------------------------------
    X = yearly["release_year"].values.reshape(-1, 1)
    y = yearly["title_count"].values

    # Train/test split by time (last 3 years as holdout)
    split = -3
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"\n[Step 3] Linear Regression model trained.")
    print(f"Holdout (last 3 years) evaluation -> MAE: {mae:.1f}, R2: {r2:.3f}")

    # Refit on full data for future forecasting
    model_full = LinearRegression()
    model_full.fit(X, y)

    future_years = np.array(range(2022, 2027)).reshape(-1, 1)
    future_preds = model_full.predict(future_years)
    future_preds = np.clip(future_preds, 0, None)  # no negative counts

    forecast_df = pd.DataFrame({
        "release_year": future_years.flatten(),
        "predicted_title_count": future_preds.round().astype(int)
    })
    print("\nForecast for 2022-2026:")
    print(forecast_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 4: Visualize future predictions
    # ------------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plt.plot(yearly["release_year"], yearly["title_count"], marker="o",
              label="Historical (actual)", color="#221f1f")
    plt.plot(forecast_df["release_year"], forecast_df["predicted_title_count"],
              marker="o", linestyle="--", label="Forecast (2022-2026)", color="#E50914")
    plt.axvline(x=2021.5, color="gray", linestyle=":", alpha=0.7)
    plt.title("Netflix Content Release Trend & Forecast")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "trend_forecast.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Step 4] Forecast chart saved -> {chart_path}")

    # ------------------------------------------------------------------
    # Step 5: Interpret results
    # ------------------------------------------------------------------
    trend_direction = "increasing" if model_full.coef_[0] > 0 else "decreasing"
    interpretation = (
        f"The linear trend model shows content releases have been {trend_direction} "
        f"by roughly {abs(model_full.coef_[0]):.0f} titles/year on average. "
        f"Model fit on the 2019-2021 holdout produced R2={r2:.2f}, meaning a simple "
        f"linear trend {'explains most of' if r2 > 0.5 else 'only weakly captures'} "
        f"the year-to-year variation (real-world catalogue growth is influenced by "
        f"licensing deals, originals strategy, and regional expansion, which a purely "
        f"linear model cannot capture). The 2020-2021 dip likely reflects production "
        f"slowdowns during the COVID-19 pandemic rather than a genuine long-term decline."
    )
    print("\n[Step 5] Interpretation:")
    print(interpretation)

    report_path = os.path.join(OUTPUT_DIR, "trend_forecast_report.txt")
    with open(report_path, "w") as f:
        f.write("TASK 4 - TREND PREDICTION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write("Historical yearly counts:\n")
        f.write(yearly.to_string(index=False) + "\n\n")
        f.write("Forecast (2022-2026):\n")
        f.write(forecast_df.to_string(index=False) + "\n\n")
        f.write(f"Model MAE: {mae:.1f}, R2: {r2:.3f}\n\n")
        f.write("Interpretation:\n" + interpretation)
    print(f"\nReport saved -> {report_path}")

    return forecast_df


if __name__ == "__main__":
    run()
