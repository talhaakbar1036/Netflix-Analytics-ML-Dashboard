"""
MAIN RUNNER
Auspify Technologies - Data Science Internship (Netflix Dataset)

Runs all 6 tasks end-to-end, in order, and prints a final summary of
where every output (charts, reports, cleaned data) was saved.

Usage:
    python3 main.py                # run all 6 tasks
    python3 main.py --tasks 1 2 5  # run only specific tasks
"""

import os
import sys
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"))

TASK_MODULES = {
    1: ("task1_data_cleaning", "Data Cleaning & Preprocessing"),
    2: ("task2_eda", "Exploratory Data Analysis (EDA)"),
    3: ("task3_recommendation_system", "Recommendation System Analysis"),
    4: ("task4_trend_prediction", "Trend Prediction Analysis"),
    5: ("task5_classification_model", "ML Classification Model"),
    6: ("task6_business_dashboard", "Business Insights Dashboard"),
}


def main():
    parser = argparse.ArgumentParser(description="Run Netflix Data Science internship tasks")
    parser.add_argument(
        "--tasks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        help="Which task numbers to run (default: all 6)"
    )
    args = parser.parse_args()

    print("#" * 70)
    print("# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP")
    print("# Netflix Dataset - Full Task Pipeline")
    print("#" * 70)

    results = {}
    errors = {}

    for task_num in args.tasks:
        if task_num not in TASK_MODULES:
            print(f"\n[!] Skipping unknown task number: {task_num}")
            continue

        module_name, task_title = TASK_MODULES[task_num]
        print(f"\n\n>>> RUNNING TASK {task_num}: {task_title} ...\n")
        try:
            module = __import__(module_name)
            result = module.run()
            results[task_num] = result
        except Exception as e:
            print(f"[ERROR] Task {task_num} failed: {e}")
            traceback.print_exc()
            errors[task_num] = str(e)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# PIPELINE COMPLETE - SUMMARY")
    print("#" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    for task_num in args.tasks:
        if task_num not in TASK_MODULES:
            continue
        _, task_title = TASK_MODULES[task_num]
        status = "FAILED" if task_num in errors else "SUCCESS"
        out_dir = os.path.join(base_dir, "outputs", f"task{task_num}")
        print(f"Task {task_num} [{status}] - {task_title}")
        if status == "SUCCESS" and os.path.isdir(out_dir):
            for fname in sorted(os.listdir(out_dir)):
                print(f"    -> outputs/task{task_num}/{fname}")

    if os.path.exists(os.path.join(base_dir, "data", "netflix_cleaned.csv")):
        print(f"\nCleaned dataset available at: data/netflix_cleaned.csv")

    if errors:
        print(f"\n{len(errors)} task(s) failed: {list(errors.keys())}")
    else:
        print("\nAll requested tasks completed successfully.")


if __name__ == "__main__":
    main()
