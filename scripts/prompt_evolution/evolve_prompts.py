#!/usr/bin/env python3
"""
Generate evolved prompts based on top performers using LLM-assisted mutation/crossover.

This is an iterative learning process:
- Analyzes what worked (top performers) and what didn't (worst performers)
- Generates evolved prompts combining successful patterns
- Avoids patterns that led to poor performance

Usage:
    # Evolve prompts for a dataset based on current best performers
    python evolve_prompts.py --dataset AILACasedocs

    # Generate a specific generation
    python evolve_prompts.py --dataset AILACasedocs --generation 2

    # Evolve prompts for all datasets
    python evolve_prompts.py --all

    # Include analysis of what didn't work
    python evolve_prompts.py --dataset AILACasedocs --learn-from-failures
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
RESULTS_CSV = SCRIPT_DIR / "results.csv"


def get_client(backend: str):
    """Get LLM client for the specified backend."""
    if backend == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    elif backend == "openai":
        import openai

        return openai.OpenAI()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Baseline scores for comparison
BASELINES = {
    "AILACasedocs": 0.4749,
    "AILAStatutes": 0.4978,
    "FreshStackRetrieval": 0.5079,
    "DS1000Retrieval": 0.7119,
    "LegalQuAD": 0.7460,
    "HC3FinanceRetrieval": 0.7749,
    "ChatDoctorRetrieval": 0.7720,
    "LegalSummarization": 0.7809,
    "FinQARetrieval": 0.8899,
    "FinanceBenchRetrieval": 0.9307,
    "WikiSQLRetrieval": 0.9670,
    "HumanEvalRetrieval": 0.9936,
    # Local datasets
    "CUREv1RetrievalLocal": 0.8228,  # Aggregated across 30 medical subsets (en/es)
    "MIRACLRetrievalHardNegativesLocal": 0.8678,  # Aggregated across 18 language subsets
}

# Domain info for context
DATASET_INFO = {
    "AILACasedocs": {
        "domain": "Legal",
        "description": "Legal case document retrieval",
        "corpus_type": "Court case documents, legal precedents",
    },
    "AILAStatutes": {
        "domain": "Legal",
        "description": "Statute retrieval",
        "corpus_type": "Legislative statutes and provisions",
    },
    "FreshStackRetrieval": {
        "domain": "Code",
        "description": "Stack Overflow retrieval",
        "corpus_type": "Stack Overflow questions and answers",
    },
    "DS1000Retrieval": {
        "domain": "Code",
        "description": "Data science code retrieval",
        "corpus_type": "Python data science code snippets",
    },
    "LegalQuAD": {
        "domain": "Legal",
        "description": "German legal QA",
        "corpus_type": "German legal documents",
    },
    "HC3FinanceRetrieval": {
        "domain": "Finance",
        "description": "Finance QA retrieval",
        "corpus_type": "Financial Q&A documents",
    },
    "ChatDoctorRetrieval": {
        "domain": "Medical",
        "description": "Medical Q&A retrieval",
        "corpus_type": "Medical Q&A documents",
    },
    "LegalSummarization": {
        "domain": "Legal",
        "description": "Legal summarization retrieval",
        "corpus_type": "Legal documents for summarization",
    },
    "FinQARetrieval": {
        "domain": "Finance",
        "description": "Financial QA with tables",
        "corpus_type": "Financial reports and tables",
    },
    "FinanceBenchRetrieval": {
        "domain": "Finance",
        "description": "Finance benchmark retrieval",
        "corpus_type": "Financial filings and SEC documents",
    },
    "WikiSQLRetrieval": {
        "domain": "Code",
        "description": "SQL query retrieval for natural language questions",
        "corpus_type": "SQL queries answering database questions",
    },
    "HumanEvalRetrieval": {
        "domain": "Code",
        "description": "Python function retrieval for programming tasks",
        "corpus_type": "Python function implementations",
    },
    "CUREv1RetrievalLocal": {
        "domain": "Medical",
        "description": "Clinical information retrieval",
        "corpus_type": "Clinical documents and medical records",
    },
    "MIRACLRetrievalHardNegativesLocal": {
        "domain": "Multilingual",
        "description": "Multilingual information retrieval with hard negatives",
        "corpus_type": "Multilingual web documents and Wikipedia passages",
    },
}

# Dataset-specific learnings from previous generations
DATASET_LEARNINGS = {
    "AILACasedocs": """## CRITICAL LEARNING FOR THIS DATASET
VERY SHORT prompts (2-5 words) work BEST for this dataset!
- "Law query:" scored 0.4753 (BEATS baseline 0.4749)
- "Court case:" scored 0.4734
- Long prompts consistently HURT performance
Generate at least 15 prompts that are UNDER 15 characters (2-3 words only)!
Examples: "Legal:", "Case:", "Law:", "Find case:", "Court:", "Retrieve:"
""",
    "HumanEvalRetrieval": """## KEY LEARNING FOR THIS DATASET
The word "easiest" was a breakthrough!
- "Return the easiest Python function solving the problem" achieved 100%!
- Simple, direct language works better than complex instructions
Focus on simplicity signals and direct action verbs.""",
    "WikiSQLRetrieval": """## KEY LEARNING FOR THIS DATASET
- Best prompt: "For the database inquiry, surface the SQL query that matches"
- Action verbs like "surface", "reveal", "uncover" work well
- Keep prompts under 15 words for best results.""",
    "LegalQuAD": """## KEY LEARNING FOR THIS DATASET
- Shorter is better: "Surface directly applicable legal content" (5 words) won
- Empty baseline was competitive - don't overcomplicate
- "directly applicable" is a powerful phrase.""",
}

EVOLUTION_PROMPT = """You are optimizing retrieval prompts for an embedding model. Based on experimental results, generate improved prompts.

## Dataset: {dataset_name}
- Domain: {domain}
- Task: {description}
- Corpus: {corpus_type}
- Baseline score (no prompt): {baseline:.2%}

## Top Performing Prompts (LEARN FROM THESE)
These prompts achieved the best scores. Analyze what makes them effective:

{top_prompts_json}

## Worst Performing Prompts (AVOID THESE PATTERNS)
These prompts performed poorly. Avoid their patterns:

{worst_prompts_json}

## Pattern Analysis
Based on the results:
- Best improvement over baseline: {best_improvement:+.2%}
- Worst degradation from baseline: {worst_degradation:+.2%}
{pattern_insights}

## Your Task
Generate 25 NEW evolved prompts that:
1. Combine successful patterns from top performers
2. Avoid patterns that appeared in worst performers
3. Explore new variations that might work even better

The model uses format: "Instruct: {{instruction}}\\nQuery: " + user_query

{dataset_specific_guidance}

## Evolution Strategies (use all)

1. **CROSSOVER (5 prompts)**: Combine elements from different top performers
   - Mix action verbs from one with structure from another
   - Blend domain terms from multiple winners

2. **MUTATION - Refine Winners (5 prompts)**: Slightly modify top performers
   - Small tweaks to the best performing prompts
   - Test variations of what already works

3. **MUTATION - Avoid Failures (3 prompts)**: Opposite of what failed
   - If long prompts failed, try shorter ones
   - If generic prompts failed, try more specific ones

4. **EXTENSION (4 prompts)**: Add qualifiers to winners
   - "most relevant", "directly applicable", "comprehensive"

5. **COMPRESSION (4 prompts)**: Shorter versions of winners
   - Distill to essential elements

6. **EXPERIMENTAL (4 prompts)**: Creative new approaches
   - Try framings not yet tested
   - Consider what makes this task unique

## Requirements
- Each prompt must be meaningfully different
- Keep prompts under 50 words
- Include variety in length
- Learn from both successes and failures

Return ONLY a JSON array of 25 strings. No explanations.
"""


def get_top_prompts(dataset: str, n: int = 5) -> list[dict]:
    """Get top N performing prompts for a dataset from results."""
    if not RESULTS_CSV.exists():
        return []

    df = pd.read_csv(RESULTS_CSV)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return []

    top = df.nlargest(n, "score")
    return [
        {"prompt": row["prompt"], "score": row["score"], "prompt_id": row["prompt_id"]}
        for _, row in top.iterrows()
    ]


def get_worst_prompts(dataset: str, n: int = 5) -> list[dict]:
    """Get worst N performing prompts for a dataset from results."""
    if not RESULTS_CSV.exists():
        return []

    df = pd.read_csv(RESULTS_CSV)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return []

    # Exclude empty baseline from worst (it's expected to be different)
    df_non_empty = df[df["prompt"].notna() & (df["prompt"] != "")]
    if len(df_non_empty) < n:
        worst = df.nsmallest(n, "score")
    else:
        worst = df_non_empty.nsmallest(n, "score")

    return [
        {"prompt": row["prompt"], "score": row["score"], "prompt_id": row["prompt_id"]}
        for _, row in worst.iterrows()
    ]


def analyze_patterns(top: list[dict], worst: list[dict], baseline: float) -> str:
    """Analyze patterns in successful vs unsuccessful prompts."""
    insights = []

    if not top or not worst:
        return ""

    # Analyze length patterns
    top_lengths = [len(str(p["prompt"])) for p in top if p["prompt"]]
    worst_lengths = [len(str(p["prompt"])) for p in worst if p["prompt"]]

    if top_lengths and worst_lengths:
        avg_top_len = sum(top_lengths) / len(top_lengths)
        avg_worst_len = sum(worst_lengths) / len(worst_lengths)
        if avg_top_len < avg_worst_len * 0.7:
            insights.append("- Shorter prompts tend to perform better")
        elif avg_top_len > avg_worst_len * 1.3:
            insights.append("- Longer, more specific prompts tend to perform better")

    # Check if empty baseline beats prompts
    baseline_in_top = any(not p["prompt"] for p in top[:3])
    if baseline_in_top:
        insights.append(
            "- Empty baseline (no prompt) is competitive - be careful not to overcomplicate"
        )

    # Analyze common words in top vs worst
    top_words = set()
    worst_words = set()
    for p in top:
        if p["prompt"]:
            top_words.update(str(p["prompt"]).lower().split())
    for p in worst:
        if p["prompt"]:
            worst_words.update(str(p["prompt"]).lower().split())

    good_words = top_words - worst_words
    bad_words = worst_words - top_words

    if good_words:
        sample = list(good_words)[:5]
        insights.append(f"- Words appearing only in winners: {', '.join(sample)}")
    if bad_words:
        sample = list(bad_words)[:5]
        insights.append(
            f"- Words appearing only in losers (avoid): {', '.join(sample)}"
        )

    return "\n".join(insights) if insights else "- No clear patterns detected yet"


def evolve_prompts(
    dataset: str,
    top_prompts: list[dict],
    client,
    worst_prompts: list[dict] = None,
    backend: str = "anthropic",
) -> list[str]:
    """Generate evolved prompts from top performers using LLM."""
    info = DATASET_INFO.get(
        dataset,
        {"domain": "General", "description": dataset, "corpus_type": "documents"},
    )
    baseline = BASELINES.get(dataset, 0.5)

    # Get worst prompts if not provided
    if worst_prompts is None:
        worst_prompts = get_worst_prompts(dataset, n=5)

    # Calculate improvements/degradations
    best_score = max(p["score"] for p in top_prompts) if top_prompts else baseline
    worst_score = min(p["score"] for p in worst_prompts) if worst_prompts else baseline
    best_improvement = best_score - baseline
    worst_degradation = worst_score - baseline

    # Analyze patterns
    pattern_insights = analyze_patterns(top_prompts, worst_prompts, baseline)

    # Get dataset-specific learnings
    dataset_guidance = DATASET_LEARNINGS.get(dataset, "")

    prompt_content = EVOLUTION_PROMPT.format(
        dataset_name=dataset,
        domain=info["domain"],
        description=info["description"],
        corpus_type=info["corpus_type"],
        baseline=baseline,
        top_prompts_json=json.dumps(top_prompts, indent=2),
        worst_prompts_json=json.dumps(worst_prompts, indent=2),
        best_improvement=best_improvement,
        worst_degradation=worst_degradation,
        pattern_insights=pattern_insights,
        dataset_specific_guidance=dataset_guidance,
    )

    if backend == "anthropic":
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt_content}],
        )
        text = response.content[0].text
    else:  # openai
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt_content}],
        )
        text = response.choices[0].message.content

    # Parse JSON from response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON: {e}")
            return []
    return []


def save_evolved_prompts(dataset: str, prompts: list[str], generation: int):
    """Save evolved prompts to file."""
    output_file = PROMPTS_DIR / f"{dataset}_gen{generation}.json"
    output_file.write_text(json.dumps(prompts, indent=2))
    print(f"  Saved {len(prompts)} prompts to {output_file}")


def determine_next_generation(dataset: str) -> int:
    """Determine the next generation number based on existing files."""
    gen = 1
    while (PROMPTS_DIR / f"{dataset}_gen{gen}.json").exists():
        gen += 1
    return gen


def check_convergence(dataset: str, threshold: float = 0.001) -> bool:
    """Check if optimization has converged (improvement is minimal)."""
    if not RESULTS_CSV.exists():
        return False

    df = pd.read_csv(RESULTS_CSV)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if len(df) < 10:
        return False

    # Check if recent results are all similar (converged)
    recent = df.nlargest(10, "score")["score"]
    score_range = recent.max() - recent.min()

    if score_range < threshold:
        print(f"  Convergence detected: top 10 scores within {score_range:.4f}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Evolve prompts based on top performers"
    )
    parser.add_argument("--dataset", help="Dataset to evolve prompts for")
    parser.add_argument(
        "--all", action="store_true", help="Evolve for all datasets with results"
    )
    parser.add_argument(
        "--generation", type=int, help="Specific generation number to create"
    )
    parser.add_argument(
        "--top-n", type=int, default=5, help="Number of top prompts to use as parents"
    )
    parser.add_argument(
        "--learn-from-failures",
        action="store_true",
        default=True,
        help="Include worst performers in analysis (default: True)",
    )
    parser.add_argument(
        "--check-convergence",
        action="store_true",
        help="Skip datasets that have converged",
    )
    args = parser.parse_args()

    # Determine which backend to use
    backend = None
    client = None

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            backend = "anthropic"
            print("Using Anthropic API (Claude)")
        except ImportError:
            pass

    if not backend and os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            client = openai.OpenAI()
            backend = "openai"
            print("Using OpenAI API (GPT-4)")
        except ImportError:
            pass

    if not backend:
        print("Error: No LLM API available.")
        print("Set one of: ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return

    if args.all:
        # Evolve for all datasets with results
        if not RESULTS_CSV.exists():
            print("No results found. Run tests first.")
            return

        df = pd.read_csv(RESULTS_CSV)
        datasets = df["dataset"].unique().tolist()
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Specify --dataset <name> or --all")
        return

    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"Evolving prompts for {dataset}")
        print(f"{'=' * 50}")

        # Check convergence
        if args.check_convergence and check_convergence(dataset):
            print(f"  Skipping {dataset}: already converged")
            continue

        # Get top performers
        top = get_top_prompts(dataset, args.top_n)
        if not top:
            print(f"  No results found for {dataset}, skipping")
            continue

        # Get worst performers for learning
        worst = (
            get_worst_prompts(dataset, args.top_n) if args.learn_from_failures else []
        )

        baseline = BASELINES.get(dataset, 0.5)
        best_score = max(p["score"] for p in top)
        improvement = best_score - baseline

        print(f"  Baseline: {baseline:.2%}")
        print(f"  Best so far: {best_score:.2%} ({improvement:+.2%})")
        print(f"  Using top {len(top)} and worst {len(worst)} as training signal")

        print(f"\n  Top performers:")
        for p in top[:3]:
            prompt_display = (
                p["prompt"][:40] + "..." if len(str(p["prompt"])) > 40 else p["prompt"]
            )
            prompt_display = prompt_display or "(empty)"
            print(f"    {p['score']:.4f}: {prompt_display}")

        if worst:
            print(f"\n  Worst performers (to avoid):")
            for p in worst[:3]:
                prompt_display = (
                    p["prompt"][:40] + "..."
                    if len(str(p["prompt"])) > 40
                    else p["prompt"]
                )
                prompt_display = prompt_display or "(empty)"
                print(f"    {p['score']:.4f}: {prompt_display}")

        # Determine generation
        generation = args.generation or determine_next_generation(dataset)
        print(f"\n  Creating generation {generation}...")

        # Generate evolved prompts
        evolved = evolve_prompts(dataset, top, client, worst, backend)
        if evolved:
            save_evolved_prompts(dataset, evolved, generation)
            print(f"  Success: {len(evolved)} new prompts ready for testing")
        else:
            print(f"  Failed to generate evolved prompts")


if __name__ == "__main__":
    main()
