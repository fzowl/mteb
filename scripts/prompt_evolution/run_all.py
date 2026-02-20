#!/usr/bin/env python3
"""
Master orchestration script for prompt evolution.

Runs the full iterative pipeline:
1. Generate initial prompts (if not exists)
2. Test all prompts
3. Analyze results
4. Evolve prompts based on winners
5. Repeat for specified generations

Usage:
    # Run full pipeline for all datasets
    python run_all.py

    # Run for specific datasets
    python run_all.py --datasets AILACasedocs FreshStackRetrieval

    # Run multiple generations
    python run_all.py --generations 3

    # Skip prompt generation (use existing)
    python run_all.py --skip-generate

    # Dry run (show what would be done)
    python run_all.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROMPTS_DIR = SCRIPT_DIR / "prompts"
RESULTS_DIR = SCRIPT_DIR / "results"


def model_short_name(model: str) -> str:
    """Extract short model name from full model identifier."""
    name = model.split("/")[-1]
    name = name.replace("-prompt-test", "")
    return name


def get_results_csv(model: str) -> Path:
    """Get model-specific results CSV path."""
    short = model_short_name(model)
    results_dir = RESULTS_DIR / short
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "results.csv"


def get_results_md(model: str) -> Path:
    """Get model-specific results MD path."""
    short = model_short_name(model)
    results_dir = RESULTS_DIR / short
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "RESULTS.md"

# All target datasets ordered by priority (largest gaps first)
ALL_DATASETS = [
    "AILACasedocs",  # 47.49% - Legal
    "FreshStackRetrieval",  # 50.79% - Code
    "AILAStatutes",  # 54.35% - Legal
    "DS1000Retrieval",  # 71.19% - Code
    "LegalQuAD",  # 74.63% - Legal
    "HC3FinanceRetrieval",  # 76.83% - Finance
    "ChatDoctorRetrieval",  # 77.20% - Medical
    "LegalSummarization",  # 78.09% - Legal
    "FinQARetrieval",  # 88.99% - Finance
    "FinanceBenchRetrieval",  # 93.16% - Finance
    "WikiSQLRetrieval",  # 96.70% - Code
    "HumanEvalRetrieval",  # 99.36% - Code
]

# Fallback baselines (voyage-4-large production). Used when no measured baseline exists.
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
    "FinanceBenchRetrieval": 0.9316,
    "WikiSQLRetrieval": 0.9670,
    "HumanEvalRetrieval": 0.9936,
}


def get_measured_baseline(df, dataset: str) -> float:
    """Get measured baseline from the {text}+{text} combo in results."""
    baseline_rows = df[
        (df["dataset"] == dataset)
        & (df["score"].notna())
        & (df["prompt"] == "{text}")
        & (df["corpus_prompt"] == "{text}")
    ]
    if not baseline_rows.empty:
        return baseline_rows["score"].iloc[0]
    return FALLBACK_BASELINES.get(dataset, 0)


def get_llm_backend(preferred: str = None):
    """Determine which LLM backend to use based on available API keys."""
    if preferred == "openai" and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if preferred == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    # Auto-detect
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def check_api_key(preferred: str = None):
    """Check that an LLM API key is set."""
    backend = get_llm_backend(preferred if preferred != "auto" else None)
    if not backend:
        print("Error: No LLM API key found")
        print("Set one of:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  export OPENAI_API_KEY=your_key")
        return None
    print(f"Using LLM backend: {backend}")
    return backend


def generate_prompts(datasets: list[str], backend: str, dry_run: bool = False) -> bool:
    """Generate initial prompts for datasets that don't have them."""
    from prompt_generator import generate_prompts as gen_prompts, DATASETS, get_client

    seed_dir = PROMPTS_DIR / "seed"
    missing = []
    for ds in datasets:
        prompts_file = seed_dir / f"{ds}.json"
        if not prompts_file.exists():
            missing.append(ds)

    if not missing:
        print("All datasets already have prompts generated.")
        return True

    print(f"\n{'=' * 60}")
    print(f"PHASE: Generate Initial Prompts")
    print(f"Datasets: {missing}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would generate prompts for:", missing)
        return True

    client = get_client(backend)
    seed_dir.mkdir(parents=True, exist_ok=True)

    for ds in missing:
        if ds not in DATASETS:
            print(f"  Warning: {ds} not in DATASETS config, skipping")
            continue

        print(f"\nGenerating prompts for {ds}...")
        prompts = gen_prompts(ds, client, backend)
        output_file = seed_dir / f"{ds}.json"
        output_file.write_text(json.dumps(prompts, indent=2))
        n_q = len(prompts.get("query_prompts", []))
        n_c = len(prompts.get("corpus_prompts", []))
        print(f"  Generated {n_q} query + {n_c} corpus prompts -> {output_file}")

    return True


def test_prompts(
    datasets: list[str],
    generation: int = 0,
    workers: int = 3,
    dry_run: bool = False,
    max_prompts: int = None,
    model: str = "voyageai/voyage-4-large-prompt-test",
    num_gpus: int | None = None,
    batch_size: int = 1000,
) -> bool:
    """Test prompts for all datasets."""
    from run_prompt_test import load_prompts, build_prompt_matrix, filter_already_tested, run_tests_parallel, run_tests_sequential

    print(f"\n{'=' * 60}")
    print(f"PHASE: Test Prompts (Generation {generation})")
    print(f"Datasets: {datasets}")
    print(f"Workers: {workers}")
    if max_prompts:
        print(f"Max prompts per dimension: {max_prompts}")
    print(f"{'=' * 60}")

    for ds in datasets:
        try:
            prompt_data = load_prompts(ds, generation, model)
        except FileNotFoundError:
            print(f"  Skipping {ds}: no prompts file for generation {generation}")
            continue

        query_prompts = prompt_data["query_prompts"]
        corpus_prompts = prompt_data["corpus_prompts"]

        # Limit each dimension if specified
        if max_prompts:
            query_prompts = query_prompts[:max_prompts]
            corpus_prompts = corpus_prompts[:max_prompts]

        # Build cross-product matrix
        prompt_tuples = build_prompt_matrix(query_prompts, corpus_prompts, generation)
        n_total = len(prompt_tuples)
        print(f"\nTesting {ds} ({len(query_prompts)}q x {len(corpus_prompts)}c = {n_total} combinations)...")

        # Skip combos already tested with valid scores
        prompt_tuples = filter_already_tested(ds, prompt_tuples, model)

        if not prompt_tuples:
            print(f"  All {n_total} combinations already tested, skipping")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would test {len(prompt_tuples)} combinations")
            continue

        if workers > 1:
            run_tests_parallel(ds, prompt_tuples, batch_size=batch_size, workers=workers, model_name=model, num_gpus=num_gpus)
        else:
            run_tests_sequential(ds, prompt_tuples, batch_size=batch_size, model_name=model)

    return True


def analyze_results(datasets: list[str], model: str = "voyageai/voyage-4-large-prompt-test") -> dict:
    """Analyze results and return summary."""
    import pandas as pd

    results_csv = get_results_csv(model)
    if not results_csv.exists():
        print("No results file found.")
        return {}

    df = pd.read_csv(results_csv)
    df = df[df["score"].notna()]
    # Fill NaN for backward compat
    df["corpus_prompt"] = df.get("corpus_prompt", pd.Series("", index=df.index)).fillna("")

    summary = {}
    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        if df_ds.empty:
            continue

        best_row = df_ds.loc[df_ds["score"].idxmax()]
        baseline = get_measured_baseline(df, ds)

        summary[ds] = {
            "baseline": baseline,
            "best_score": best_row["score"],
            "improvement": best_row["score"] - baseline,
            "best_prompt": best_row["prompt"] if pd.notna(best_row["prompt"]) else "",
            "best_corpus_prompt": best_row.get("corpus_prompt", ""),
            "best_prompt_id": best_row["prompt_id"],
            "best_corpus_prompt_id": best_row.get("corpus_prompt_id", ""),
        }

    return summary


def evolve_prompts(
    datasets: list[str],
    generation: int,
    backend: str,
    dry_run: bool = False,
    model: str = "voyage-4-large",
    full_model: str = "voyageai/voyage-4-large-prompt-test",
) -> bool:
    """Generate evolved prompts based on top performers."""
    from evolve_prompts import (
        evolve_prompts as evolve,
        save_evolved_prompts,
        get_client,
        get_top_combos,
    )

    print(f"\n{'=' * 60}")
    print(f"PHASE: Evolve Prompts -> Generation {generation}")
    print(f"Datasets: {datasets}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would evolve prompts for:", datasets)
        return True

    client = get_client(backend)

    for ds in datasets:
        top = get_top_combos(ds, n=3, model=full_model)
        if not top:
            print(f"  Skipping {ds}: no results found")
            continue

        print(f"\nEvolving {ds} (using full history)...")
        evolved = evolve(ds, client, backend=backend, model=full_model)
        if evolved:
            save_evolved_prompts(ds, evolved, generation, model)
        else:
            print(f"  Failed to generate evolved prompts for {ds}")

    return True


def update_results_doc(summary: dict, generation: int, model: str = "voyageai/voyage-4-large-prompt-test"):
    """Update RESULTS.md with latest findings."""
    results_md = get_results_md(model)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Read existing content or create new
    if results_md.exists():
        content = results_md.read_text()
    else:
        content = """# Prompt Evolution Results

This document tracks the results of prompt optimization experiments for voyage-4-large.

## Target Datasets

| Dataset | Baseline | Domain |
|---------|----------|--------|
| AILACasedocs | 47.49% | Legal |
| FreshStackRetrieval | 50.79% | Code |
| AILAStatutes | 54.35% | Legal |
| DS1000Retrieval | 71.19% | Code |
| LegalQuAD | 74.63% | Legal |
| HC3FinanceRetrieval | 76.83% | Finance |
| ChatDoctorRetrieval | 77.20% | Medical |
| LegalSummarization | 78.09% | Legal |
| FinQARetrieval | 88.99% | Finance |

---

"""

    # Add new results section
    new_section = f"\n## Generation {generation} Results ({timestamp})\n\n"
    new_section += "| Dataset | Baseline | Best Score | Improvement | Best Prompt |\n"
    new_section += "|---------|----------|------------|-------------|-------------|\n"

    for ds, data in sorted(summary.items()):
        prompt_preview = (
            data["best_prompt"][:40] + "..."
            if len(str(data["best_prompt"])) > 40
            else data["best_prompt"]
        )
        prompt_preview = prompt_preview or "(empty)"
        improvement_pct = (
            data["improvement"] / data["baseline"] * 100 if data["baseline"] > 0 else 0
        )
        new_section += (
            f"| {ds} | {data['baseline']:.2%} | {data['best_score']:.2%} | "
            f"{data['improvement']:+.2%} ({improvement_pct:+.1f}%) | {prompt_preview} |\n"
        )

    # Add learnings section
    new_section += "\n### Key Learnings\n\n"

    improved = [ds for ds, d in summary.items() if d["improvement"] > 0]
    hurt = [ds for ds, d in summary.items() if d["improvement"] < 0]

    if improved:
        new_section += f"**Improved:** {', '.join(improved)}\n\n"
    if hurt:
        new_section += (
            f"**Hurt by prompts:** {', '.join(hurt)} (baseline is better)\n\n"
        )

    # Add winning prompts
    new_section += "### Winning Prompts\n\n"
    new_section += "```python\nWINNING_PROMPTS = {\n"
    for ds, data in sorted(summary.items()):
        if data["improvement"] > 0:
            qp = data["best_prompt"]
            cp = data.get("best_corpus_prompt", "")
            new_section += f'    "{ds}": {{"query": "{qp}", "corpus": "{cp}"}},\n'
        else:
            new_section += f'    "{ds}": {{"query": "", "corpus": ""}},  # baseline is better\n'
    new_section += "}\n```\n"

    content += new_section
    results_md.write_text(content)
    print(f"\nResults documented in: {results_md}")


def print_summary(summary: dict):
    """Print summary to console."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<25} {'Baseline':>10} {'Best':>10} {'Improve':>12}")
    print("-" * 57)

    for ds, data in sorted(summary.items()):
        pct = (
            data["improvement"] / data["baseline"] * 100 if data["baseline"] > 0 else 0
        )
        print(
            f"{ds:<25} {data['baseline']:>10.2%} {data['best_score']:>10.2%} "
            f"{data['improvement']:>+10.2%} ({pct:>+.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Run full prompt evolution pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="Number of evolution generations to run (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip initial prompt generation",
    )
    parser.add_argument(
        "--start-generation",
        type=int,
        default=0,
        help="Start from specific generation (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: 1 dataset, 3 prompts, 1 generation",
    )
    parser.add_argument(
        "--prompts-per-gen",
        type=int,
        default=None,
        help="Limit prompts tested per generation (for quick testing)",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "openai", "auto"],
        default="auto",
        help="LLM backend for prompt generation/evolution (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for encoding (default: 1)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to distribute workers across (round-robin assignment)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voyageai/voyage-4-large-prompt-test",
        help="Model to test (default: voyageai/voyage-4-large-prompt-test)",
    )
    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.datasets = [args.datasets[0]] if args.datasets else ["FinQARetrieval"]
        args.generations = 1
        args.prompts_per_gen = 3
        print("🧪 QUICK TEST MODE: 1 dataset, 3 prompts, 1 generation")

    print("=" * 60)
    print("PROMPT EVOLUTION PIPELINE")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Generations: {args.generations}")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    # Check API key and get backend
    backend = None
    if not args.dry_run:
        backend = check_api_key(args.backend if args.backend != "auto" else None)
        if not backend:
            return

    # Phase 1: Generate initial prompts
    if not args.skip_generate and args.start_generation == 0:
        generate_prompts(args.datasets, backend, args.dry_run)

    # Auto-detect start generation: find the next available gen number
    # so we don't waste a cycle re-testing already-completed seed prompts
    start_gen = args.start_generation
    if start_gen == 0:
        from evolve_prompts import determine_next_generation
        model_short = args.model.split("/")[-1].replace("-prompt-test", "")
        # Check if seed (gen 0) has already been tested — if so, start from next gen
        results_csv = get_results_csv(args.model)
        if results_csv.exists():
            import pandas as pd
            df_check = pd.read_csv(results_csv)
            for ds in args.datasets:
                ds_results = df_check[(df_check["dataset"] == ds) & (df_check["score"].notna())]
                if len(ds_results) > 0:
                    # Results exist — auto-advance to evolve new prompts
                    next_gen = determine_next_generation(ds, model_short)
                    start_gen = max(start_gen, next_gen)
                    print(f"Auto-detected: {ds} has results, starting evolution from gen {next_gen}")

    # Run generations: evolve first, then test
    for i in range(args.generations):
        gen = start_gen + i
        print(f"\n{'#' * 60}")
        print(f"# GENERATION {gen}")
        print(f"{'#' * 60}")

        # If this is a new generation (not seed), evolve prompts first
        if gen > 0:
            model_short = args.model.split("/")[-1].replace("-prompt-test", "")
            evolve_prompts(args.datasets, gen, backend, args.dry_run, model_short, args.model)

        # Test current generation
        test_prompts(
            args.datasets, gen, args.workers, args.dry_run, args.prompts_per_gen, args.model, args.gpus, args.batch_size
        )

        # Analyze results
        if not args.dry_run:
            summary = analyze_results(args.datasets, args.model)
            print_summary(summary)
            update_results_doc(summary, gen, args.model)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"Results: {get_results_csv(args.model)}")
    print(f"Documentation: {get_results_md(args.model)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
