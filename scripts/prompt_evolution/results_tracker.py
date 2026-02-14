#!/usr/bin/env python3
"""
Analyze prompt evolution results and identify winning prompts.

Usage:
    # Show best prompts per dataset
    python results_tracker.py

    # Show top N for specific dataset
    python results_tracker.py --dataset AILACasedocs --top 10

    # Export best prompts as Python dict
    python results_tracker.py --export

    # Show improvement stats
    python results_tracker.py --stats
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
RESULTS_CSV = SCRIPT_DIR / "results.csv"

# Baseline scores (without prompts)
BASELINES = {
    "AILACasedocs": 0.4749,
    "AILAStatutes": 0.5435,
    "FreshStackRetrieval": 0.5079,
    "DS1000Retrieval": 0.7119,
    "LegalQuAD": 0.7463,
    "HC3FinanceRetrieval": 0.7683,
    "ChatDoctorRetrieval": 0.7720,
    "LegalSummarization": 0.7809,
    "FinQARetrieval": 0.8899,
}


def load_results() -> pd.DataFrame:
    """Load results from CSV file."""
    if not RESULTS_CSV.exists():
        print(f"No results file found at {RESULTS_CSV}")
        print("Run some tests first with: python run_prompt_test.py --dataset <name>")
        return pd.DataFrame()
    return pd.read_csv(RESULTS_CSV)


def analyze_results(df: pd.DataFrame, show_all: bool = False):
    """Analyze results and show best prompts per dataset."""
    if df.empty:
        return

    # Filter out errors
    df = df[df["score"].notna()]

    if df.empty:
        print("No successful results found.")
        return

    # Group by dataset, find best prompt
    best = df.loc[df.groupby("dataset")["score"].idxmax()]

    print("=" * 80)
    print("BEST PROMPTS BY DATASET")
    print("=" * 80)

    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = BASELINES.get(dataset, 0)
        improvement = row["score"] - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0

        print(f"\n{dataset}:")
        print(f"  Best Score: {row['score']:.4f}")
        print(f"  Baseline:   {baseline:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
        print(f"  Prompt ID:  {row['prompt_id']}")
        prompt_display = (
            row["prompt"][:70] + "..."
            if len(str(row["prompt"])) > 70
            else row["prompt"]
        )
        prompt_display = prompt_display or "(empty baseline)"
        print(f"  Prompt:     {prompt_display}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Dataset':<25} {'Baseline':>10} {'Best':>10} {'Improve':>10} {'% Improve':>10}"
    )
    print("-" * 65)

    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = BASELINES.get(dataset, 0)
        improvement = row["score"] - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0
        print(
            f"{dataset:<25} {baseline:>10.4f} {row['score']:>10.4f} "
            f"{improvement:>+10.4f} {pct_improvement:>+9.1f}%"
        )


def get_top_n(df: pd.DataFrame, dataset: str, n: int = 5) -> pd.DataFrame:
    """Get top N prompts for a dataset."""
    if df.empty:
        return pd.DataFrame()

    df_dataset = df[(df["dataset"] == dataset) & (df["score"].notna())]
    return df_dataset.nlargest(n, "score")


def show_dataset_details(df: pd.DataFrame, dataset: str, top_n: int = 10):
    """Show detailed results for a specific dataset."""
    if df.empty:
        return

    df_dataset = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df_dataset.empty:
        print(f"No results found for {dataset}")
        return

    baseline = BASELINES.get(dataset, 0)
    top = df_dataset.nlargest(top_n, "score")

    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} PROMPTS FOR {dataset}")
    print(f"Baseline: {baseline:.4f}")
    print(f"{'=' * 80}")

    for i, (_, row) in enumerate(top.iterrows(), 1):
        improvement = row["score"] - baseline
        prompt_display = (
            row["prompt"][:60] + "..."
            if len(str(row["prompt"])) > 60
            else row["prompt"]
        )
        prompt_display = prompt_display or "(empty)"
        print(f"\n{i}. Score: {row['score']:.4f} ({improvement:+.4f})")
        print(f"   ID: {row['prompt_id']}")
        print(f"   Prompt: {prompt_display}")


def export_best_prompts(df: pd.DataFrame) -> dict:
    """Export best prompts as a Python dictionary for voyage_models.py."""
    if df.empty:
        return {}

    df = df[df["score"].notna()]
    best = df.loc[df.groupby("dataset")["score"].idxmax()]

    result = {}
    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = BASELINES.get(dataset, 0)

        # Only include if it improves over baseline
        if row["score"] > baseline:
            result[dataset] = row["prompt"]
        else:
            result[dataset] = ""  # Use empty string (no prompt) if prompts hurt

    print("# Optimized prompts for OPTIMIZED_PROMPTS dict in voyage_models.py")
    print("EVOLVED_PROMPTS = {")
    for dataset, prompt in sorted(result.items()):
        if prompt:
            print(f'    "{dataset}": "{prompt}",')
        else:
            print(f'    "{dataset}": "",  # baseline is better')
    print("}")

    return result


def show_stats(df: pd.DataFrame):
    """Show overall statistics."""
    if df.empty:
        return

    df = df[df["score"].notna()]

    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 80}")

    print(f"\nTotal tests run: {len(df)}")
    print(f"Datasets tested: {df['dataset'].nunique()}")
    print(f"Unique prompts: {df['prompt'].nunique()}")

    # Calculate improvements
    improvements = []
    for dataset in df["dataset"].unique():
        baseline = BASELINES.get(dataset, 0)
        best = df[df["dataset"] == dataset]["score"].max()
        if baseline > 0:
            improvements.append((best - baseline) / baseline * 100)

    if improvements:
        print(f"\nAverage improvement: {sum(improvements) / len(improvements):+.1f}%")
        print(f"Max improvement: {max(improvements):+.1f}%")
        print(f"Min improvement: {min(improvements):+.1f}%")

        # Count datasets improved
        improved = sum(1 for i in improvements if i > 0)
        print(f"\nDatasets improved: {improved}/{len(improvements)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt evolution results")
    parser.add_argument("--dataset", help="Show details for specific dataset")
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top prompts to show"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export best prompts as Python dict"
    )
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")
    parser.add_argument("--all", action="store_true", help="Show all results")
    args = parser.parse_args()

    df = load_results()

    if df.empty:
        return

    if args.export:
        export_best_prompts(df)
    elif args.dataset:
        show_dataset_details(df, args.dataset, args.top)
    elif args.stats:
        show_stats(df)
    else:
        analyze_results(df, args.all)
        print("\n" + "-" * 80)
        print("Use --dataset <name> to see detailed results for a specific dataset")
        print("Use --export to generate Python code for voyage_models.py")
        print("Use --stats for overall statistics")


if __name__ == "__main__":
    main()
