#!/usr/bin/env python3
"""
Compare prompt performance across Voyage models.

Loads gen0 results from all model-specific CSVs, joins on (query_prompt, corpus_prompt),
and produces a comparison table showing how each prompt combo performs on each model.

Usage:
    python compare_models.py --dataset AILAStatutes
    python compare_models.py --dataset AILAStatutes --top 20
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_prompt_test import RESULTS_DIR, model_short_name

MODELS = ["voyage-4-nano", "voyage-4-lite", "voyage-4", "voyage-4-large"]


def load_model_results(dataset: str, model_short: str) -> dict[tuple[str, str], float]:
    """Load results for a model, keyed by (query_prompt, corpus_prompt) -> best score."""
    results_csv = RESULTS_DIR / model_short / "results.csv"
    if not results_csv.exists():
        return {}

    scores = {}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != dataset:
                continue
            score = row.get("score")
            if not score:
                continue
            key = (row.get("prompt", ""), row.get("corpus_prompt", ""))
            val = float(score)
            # Keep best score if tested multiple times
            if key not in scores or val > scores[key]:
                scores[key] = val
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare prompt performance across Voyage models")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--top", type=int, default=10, help="Show top N combos per model (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: results/comparison_{dataset}.csv)")
    args = parser.parse_args()

    # Load results for all models
    all_results = {}
    for model in MODELS:
        all_results[model] = load_model_results(args.dataset, model)

    # Check which models have data
    available = {m: r for m, r in all_results.items() if r}
    if not available:
        print(f"No results found for {args.dataset}. Run run_cross_model.py first.")
        return

    print(f"Dataset: {args.dataset}")
    print(f"Models with results: {', '.join(available.keys())}")
    print()

    # Per-model summary
    print(f"{'Model':<20} {'Count':>6} {'Baseline':>10} {'Best':>10} {'Avg':>10} {'Improvement':>12}")
    print("-" * 70)
    for model in MODELS:
        scores = all_results[model]
        if not scores:
            print(f"{model:<20} {'(no data)':>6}")
            continue

        vals = list(scores.values())
        baseline_key = ("{text}", "{text}")
        baseline = scores.get(baseline_key)
        best_score = max(vals)
        avg_score = sum(vals) / len(vals)
        improvement = ""
        if baseline is not None:
            diff = best_score - baseline
            pct = (diff / baseline * 100) if baseline > 0 else 0
            improvement = f"+{diff:.4f} ({pct:+.1f}%)"

        print(
            f"{model:<20} {len(vals):>6} "
            f"{baseline if baseline else 'N/A':>10.4f} "
            f"{best_score:>10.4f} {avg_score:>10.4f} {improvement:>12}"
        )

    print()

    # Collect all prompt keys across models
    all_keys = set()
    for scores in all_results.values():
        all_keys.update(scores.keys())

    # Build comparison table
    rows = []
    for key in sorted(all_keys):
        qp, cp = key
        row = {"query_prompt": qp, "corpus_prompt": cp}
        for model in MODELS:
            row[model] = all_results[model].get(key)
        rows.append(row)

    # Show top combos for each model
    for model in MODELS:
        scores = all_results[model]
        if not scores:
            continue

        print(f"Top {args.top} prompts for {model}:")
        sorted_combos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, ((qp, cp), score) in enumerate(sorted_combos[: args.top]):
            qd = (qp[:40] + "...") if len(qp) > 40 else qp
            cd = (cp[:30] + "...") if len(cp) > 30 else cp
            # Show scores on other models for this combo
            others = []
            for other_model in MODELS:
                if other_model == model:
                    continue
                other_score = all_results[other_model].get((qp, cp))
                if other_score is not None:
                    others.append(f"{other_model}: {other_score:.4f}")
            other_str = " | ".join(others) if others else ""
            print(f"  {i + 1:>3}. {score:.4f}  Q: {qd}")
            print(f"              C: {cd}")
            if other_str:
                print(f"              [{other_str}]")
        print()

    # Cross-model correlation: find prompts that work well/poorly across models
    if len(available) >= 2:
        print("Cross-model analysis:")
        # Find combos tested on all available models
        common_keys = set.intersection(*(set(r.keys()) for r in available.values()))
        if common_keys:
            print(f"  Prompts tested on all {len(available)} models: {len(common_keys)}")

            # Find universally good prompts (above average on all models)
            model_avgs = {m: sum(r.values()) / len(r) for m, r in available.items()}
            universal_good = []
            for key in common_keys:
                if all(available[m][key] > model_avgs[m] for m in available):
                    avg_rank_score = sum(available[m][key] for m in available) / len(available)
                    universal_good.append((key, avg_rank_score))

            if universal_good:
                universal_good.sort(key=lambda x: x[1], reverse=True)
                print(f"\n  Universally above-average prompts ({len(universal_good)}):")
                for (qp, cp), avg_s in universal_good[:5]:
                    qd = (qp[:50] + "...") if len(qp) > 50 else qp
                    cd = (cp[:40] + "...") if len(cp) > 40 else cp
                    parts = [f"{m}: {available[m][(qp, cp)]:.4f}" for m in available]
                    print(f"    Q: {qd}")
                    print(f"    C: {cd}")
                    print(f"    Scores: {' | '.join(parts)}")
            else:
                print("  No universally above-average prompts found.")
        else:
            print("  No prompts tested on all models yet.")
        print()

    # Save comparison CSV
    output_path = args.output or str(RESULTS_DIR / f"comparison_{args.dataset}.csv")
    fieldnames = ["query_prompt", "corpus_prompt"] + MODELS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Comparison CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
