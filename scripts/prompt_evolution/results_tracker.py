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
RESULTS_DIR = SCRIPT_DIR / "results"


def model_short_name(model: str) -> str:
    """Extract short model name from full model identifier."""
    name = model.split("/")[-1]
    name = name.replace("-prompt-test", "")
    return name


def get_results_csv(model: str) -> Path:
    """Get model-specific results CSV path."""
    short = model_short_name(model)
    return RESULTS_DIR / short / "results.csv"

# Fallback baseline scores (voyage-4-large production, used when no measured baseline exists)
FALLBACK_BASELINES = {
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


def get_measured_baselines(df: pd.DataFrame) -> dict[str, float]:
    """Extract measured baselines from results: the score of {text}+{text} combo.

    The baseline is the score when both query and corpus prompts are just "{text}"
    (i.e., passthrough / no prompt). This is model-specific and measured, not hardcoded.
    """
    baselines = {}
    if df.empty:
        return baselines
    for dataset in df["dataset"].unique():
        df_ds = df[
            (df["dataset"] == dataset)
            & (df["score"].notna())
            & (df["prompt"] == "{text}")
            & (df["corpus_prompt"] == "{text}")
        ]
        if not df_ds.empty:
            baselines[dataset] = df_ds["score"].iloc[0]
    return baselines


def _safe_str(val) -> str:
    """Convert value to string, handling NaN/None as empty string."""
    if pd.isna(val):
        return ""
    return str(val)


def load_results(model: str = "voyageai/voyage-4-large-prompt-test") -> pd.DataFrame:
    """Load results from model-specific CSV file."""
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        print(f"No results file found at {results_csv}")
        print("Run some tests first with: python run_prompt_test.py --dataset <name>")
        return pd.DataFrame()
    df = pd.read_csv(results_csv)
    # Backward compat: fill missing corpus columns
    if "corpus_prompt" not in df.columns:
        df["corpus_prompt"] = ""
    if "corpus_prompt_id" not in df.columns:
        df["corpus_prompt_id"] = ""
    df["corpus_prompt"] = df["corpus_prompt"].fillna("")
    df["corpus_prompt_id"] = df["corpus_prompt_id"].fillna("")
    df["prompt"] = df["prompt"].fillna("")
    return df


def analyze_results(df: pd.DataFrame, show_all: bool = False):
    """Analyze results and show best prompt combinations per dataset."""
    if df.empty:
        return

    # Filter out errors
    df = df[df["score"].notna()]

    if df.empty:
        print("No successful results found.")
        return

    # Use measured baselines (from {text}+{text} combo), fall back to hardcoded
    measured = get_measured_baselines(df)

    # Group by dataset, find best combo
    best = df.loc[df.groupby("dataset")["score"].idxmax()]

    print("=" * 90)
    print("BEST PROMPT COMBINATIONS BY DATASET")
    print("=" * 90)

    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0))
        improvement = row["score"] - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0

        print(f"\n{dataset}:")
        print(f"  Best Score: {row['score']:.4f}")
        print(f"  Baseline:   {baseline:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
        print(f"  Query Prompt ID:  {row['prompt_id']}")
        qp = _safe_str(row["prompt"])
        qp_display = qp[:70] + "..." if len(qp) > 70 else qp
        print(f"  Query Prompt:     {qp_display or '(empty baseline)'}")
        cp = _safe_str(row["corpus_prompt"])
        cp_display = cp[:70] + "..." if len(cp) > 70 else cp
        print(f"  Corpus Prompt ID: {row['corpus_prompt_id'] or 'N/A'}")
        print(f"  Corpus Prompt:    {cp_display or '(empty baseline)'}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(
        f"{'Dataset':<25} {'Baseline':>10} {'Best':>10} {'Improve':>10} {'% Improve':>10}"
    )
    print("-" * 65)

    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0))
        improvement = row["score"] - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0
        print(
            f"{dataset:<25} {baseline:>10.4f} {row['score']:>10.4f} "
            f"{improvement:>+10.4f} {pct_improvement:>+9.1f}%"
        )


def get_top_n(df: pd.DataFrame, dataset: str, n: int = 5) -> pd.DataFrame:
    """Get top N prompt combos for a dataset."""
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

    measured = get_measured_baselines(df)
    baseline = measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0))
    top = df_dataset.nlargest(top_n, "score")

    print(f"\n{'=' * 90}")
    print(f"TOP {top_n} PROMPT COMBINATIONS FOR {dataset}")
    print(f"Baseline: {baseline:.4f}")
    print(f"{'=' * 90}")

    for i, (_, row) in enumerate(top.iterrows(), 1):
        improvement = row["score"] - baseline
        qp = _safe_str(row["prompt"])
        qp_display = qp[:50] + "..." if len(qp) > 50 else qp
        cp = _safe_str(row["corpus_prompt"])
        cp_display = cp[:50] + "..." if len(cp) > 50 else cp
        print(f"\n{i}. Score: {row['score']:.4f} ({improvement:+.4f})")
        print(f"   Q: {row['prompt_id']} = {qp_display or '(empty)'}")
        print(f"   C: {row['corpus_prompt_id'] or 'N/A'} = {cp_display or '(empty)'}")


def export_best_prompts(df: pd.DataFrame) -> dict:
    """Export best prompt combinations as a Python dictionary for voyage_models.py."""
    if df.empty:
        return {}

    df = df[df["score"].notna()]
    measured = get_measured_baselines(df)
    best = df.loc[df.groupby("dataset")["score"].idxmax()]

    result = {}
    for _, row in best.iterrows():
        dataset = row["dataset"]
        baseline = measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0))

        # Only include if it improves over baseline
        if row["score"] > baseline:
            result[dataset] = {
                "query": _safe_str(row["prompt"]),
                "corpus": _safe_str(row["corpus_prompt"]),
            }
        else:
            result[dataset] = {"query": "", "corpus": ""}

    print("# Optimized prompts for voyage_models.py")
    print("EVOLVED_BEST_PROMPTS = {")
    for dataset, prompts in sorted(result.items()):
        if prompts["query"] or prompts["corpus"]:
            print(f'    "{dataset}": "{prompts["query"]}",')
        else:
            print(f'    "{dataset}": "",  # baseline is better')
    print("}")
    print()
    print("EVOLVED_BEST_CORPUS_PROMPTS = {")
    for dataset, prompts in sorted(result.items()):
        if prompts["corpus"]:
            print(f'    "{dataset}": "{prompts["corpus"]}",')
        else:
            print(f'    "{dataset}": "",  # no corpus prompt needed')
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
    print(f"Unique query prompts: {df['prompt'].nunique()}")
    print(f"Unique corpus prompts: {df['corpus_prompt'].nunique()}")
    n_with_corpus = len(df[df["corpus_prompt"] != ""])
    print(f"Tests with corpus prompt: {n_with_corpus}/{len(df)}")

    # Calculate improvements
    measured = get_measured_baselines(df)
    improvements = []
    for dataset in df["dataset"].unique():
        baseline = measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0))
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
    parser.add_argument(
        "--model",
        type=str,
        default="voyageai/voyage-4-large-prompt-test",
        help="Model to load results for (default: voyageai/voyage-4-large-prompt-test)",
    )
    args = parser.parse_args()

    df = load_results(args.model)

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
