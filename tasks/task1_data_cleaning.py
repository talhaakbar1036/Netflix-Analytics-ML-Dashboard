"""
Task 1 (Easy) - Data Cleaning & Preprocessing
Auspify Technologies - Data Science Internship

Goal: Prepare the Netflix dataset for analysis by cleaning, transforming,
and organizing raw data.

Workflow:
 Step 1: Import the dataset using Pandas.
 Step 2: Identify missing and duplicate records.
 Step 3: Handle null values and inconsistent formats.
 Step 4: Transform categorical and date-related columns.
 Step 5: Create a clean dataset for further analysis.
"""

import os
import pandas as pd

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task1")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_cleaned.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(save_report=True):
    print("=" * 70)
    print("TASK 1: DATA CLEANING & PREPROCESSING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Import the dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"\n[Step 1] Dataset loaded -> shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head(3).to_string())

    # ------------------------------------------------------------------
    # Step 2: Identify missing and duplicate records
    # ------------------------------------------------------------------
    print("\n[Step 2] Missing values per column (raw '' / NaN):")
    missing_raw = df.isnull().sum()
    print(missing_raw.to_string())

    # This dataset uses the literal string "Not Given" as a placeholder
    # for missing country values instead of NaN -> treat it as missing too.
    placeholder_missing = (df == "Not Given").sum()
    placeholder_missing = placeholder_missing[placeholder_missing > 0]
    print("\nPlaceholder missing values ('Not Given'):")
    print(placeholder_missing.to_string() if len(placeholder_missing) else "None found")

    dup_rows = df.duplicated().sum()
    dup_titles = df.duplicated(subset=["title"]).sum()
    print(f"\nFully duplicated rows: {dup_rows}")
    print(f"Duplicate titles (same title appears more than once): {dup_titles}")

    # ------------------------------------------------------------------
    # Step 3: Handle null values and inconsistent formats
    # ------------------------------------------------------------------
    df_clean = df.copy()

    # Replace "Not Given" placeholder with proper NaN, then fill sensibly
    df_clean["country"] = df_clean["country"].replace("Not Given", pd.NA)
    df_clean["country"] = df_clean["country"].fillna("Unknown")

    # director sometimes uses similar placeholders / blanks -> normalize
    df_clean["director"] = df_clean["director"].replace(
        ["Not Given", "", " "], pd.NA
    )
    df_clean["director"] = df_clean["director"].fillna("Unknown")

    # Drop exact duplicate rows
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    print(f"\n[Step 3] Dropped {before - after} exact duplicate rows")

    # Strip whitespace from all string/object columns
    obj_cols = df_clean.select_dtypes(include="object").columns
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # ------------------------------------------------------------------
    # Step 4: Transform categorical and date-related columns
    # ------------------------------------------------------------------
    # Parse date_added into a real datetime column
    df_clean["date_added"] = pd.to_datetime(df_clean["date_added"], errors="coerce")
    df_clean["year_added"] = df_clean["date_added"].dt.year
    df_clean["month_added"] = df_clean["date_added"].dt.month_name()

    # Categorical typing for memory efficiency + consistency
    df_clean["type"] = df_clean["type"].astype("category")
    df_clean["rating"] = df_clean["rating"].astype("category")

    # Split "duration" into a numeric value + unit (min / Season(s))
    dur_split = df_clean["duration"].str.extract(r"(\d+)\s*(\w+)")
    df_clean["duration_value"] = pd.to_numeric(dur_split[0], errors="coerce")
    df_clean["duration_unit"] = dur_split[1]

    # Split multi-value "listed_in" genre string into a list column
    df_clean["genres"] = df_clean["listed_in"].apply(
        lambda x: [g.strip() for g in str(x).split(",")]
    )

    print("\n[Step 4] New / transformed columns created:")
    print(" - date_added (datetime), year_added, month_added")
    print(" - type, rating -> category dtype")
    print(" - duration_value (numeric), duration_unit")
    print(" - genres (list of individual genres)")

    # ------------------------------------------------------------------
    # Step 5: Create a clean dataset for further analysis
    # ------------------------------------------------------------------
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n[Step 5] Clean dataset saved -> {CLEAN_DATA_PATH}")
    print(f"Final shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")

    if save_report:
        report_path = os.path.join(OUTPUT_DIR, "cleaning_report.txt")
        with open(report_path, "w") as f:
            f.write("TASK 1 - DATA CLEANING REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Raw shape: {df.shape}\n")
            f.write(f"Clean shape: {df_clean.shape}\n")
            f.write(f"Duplicate rows removed: {before - after}\n")
            f.write(f"'Not Given' countries replaced: {placeholder_missing.get('country', 0)}\n")
            f.write(f"Duplicate titles found: {dup_titles}\n")
        print(f"Report saved -> {report_path}")

    return df_clean


if __name__ == "__main__":
    run()
