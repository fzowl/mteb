#!/usr/bin/env python3
"""
Generate evolved prompts based on top performers using LLM-assisted mutation/crossover.

This is an iterative learning process:
- Analyzes what worked (top performers) and what didn't (worst performers)
- Computes marginal scores for every unique query and corpus prompt across full history
- Generates evolved query + corpus prompts combining successful patterns
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
    "CUREv1RetrievalLocal": 0.8228,
    "MIRACLRetrievalHardNegativesLocal": 0.8678,
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
Generate at least 5 query prompts that are UNDER 15 characters (2-3 words only)!
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

EVOLUTION_PROMPT = """You are optimizing retrieval prompts for an embedding model. We test BOTH query prompts (applied to queries) and corpus prompts (applied to documents). Based on experimental results, generate improved prompts.

## Dataset: {dataset_name}
- Domain: {domain}
- Task: {description}
- Corpus: {corpus_type}
- Baseline score (no prompts): {baseline:.2%}

## COMPLETE TEST HISTORY
Every (query_prompt, corpus_prompt) combination ever tested, sorted by score (best first):

{full_combo_history_json}

## Pattern Analysis
- Best improvement over baseline: {best_improvement:+.2%}
- Worst degradation from baseline: {worst_degradation:+.2%}
{pattern_insights}

{dataset_specific_guidance}

## Prompt Format
Every prompt MUST contain {{text}} as placeholder for the actual input sentence.
The model replaces {{text}} with the query or document text before encoding.
- "Instruct: Find relevant statutes\\nQuery: {{text}}" — structured template
- "Legal question: {{text}} — find matching precedents" — text in the middle
- "Find relevant documents: {{text}}" — simple prefix
- "{{text}}" — passthrough (baseline)

## Your Task
Generate 10 NEW query prompts and 10 NEW corpus prompts.

CRITICAL: Do NOT duplicate any prompt from the history above. Every prompt you generate must be different from all previously tested prompts. Duplicates are automatically detected and skipped, wasting a slot. Note: varying the wording within the same format IS allowed and encouraged (e.g. "Instruct: Find statutes\\nQuery: {{text}}" and "Instruct: Locate provisions\\nQuery: {{text}}" are different prompts even though they share the same format).

STRATEGY — Study the complete history above carefully:
- If a clear winning FORMAT or PATTERN emerges (e.g. a specific prompt structure consistently scores higher), generate ~8 out of 10 prompts as NEW VARIATIONS within that winning pattern. Vary the wording, instructions, and specifics while keeping the format.
- Generate ~2 out of 10 prompts as EXPLORATORY: completely different formats or approaches not yet tried. This ensures we don't miss a potentially better pattern.
- If no clear pattern has emerged yet, be more exploratory (50/50 split).

For QUERY prompts: focus on what instructions help the model understand the search INTENT.
For CORPUS prompts: focus on what labels/descriptions help the model understand the document TYPE.

Return ONLY a JSON object. No explanations.
IMPORTANT: Every prompt string MUST contain {{text}} exactly once. The baseline (passthrough) is just "{{text}}".
{{"query_prompts": [10 strings], "corpus_prompts": [10 strings]}}
"""


def _safe_str(val) -> str:
    """Convert a value to string, handling NaN/None as empty string."""
    if pd.isna(val):
        return ""
    return str(val)


def get_top_combos(dataset: str, n: int = 10, model: str = "voyageai/voyage-4-large-prompt-test") -> list[dict]:
    """Get top N performing prompt combinations for a dataset."""
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return []

    df = pd.read_csv(results_csv)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return []

    top = df.nlargest(n, "score")
    return [
        {
            "query_prompt": _safe_str(row.get("prompt", "")),
            "corpus_prompt": _safe_str(row.get("corpus_prompt", "")),
            "score": row["score"],
        }
        for _, row in top.iterrows()
    ]


def get_worst_combos(dataset: str, n: int = 5, model: str = "voyageai/voyage-4-large-prompt-test") -> list[dict]:
    """Get worst N performing prompt combinations for a dataset."""
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return []

    df = pd.read_csv(results_csv)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return []

    worst = df.nsmallest(n, "score")
    return [
        {
            "query_prompt": _safe_str(row.get("prompt", "")),
            "corpus_prompt": _safe_str(row.get("corpus_prompt", "")),
            "score": row["score"],
        }
        for _, row in worst.iterrows()
    ]


def get_full_combo_history(dataset: str, model: str = "voyageai/voyage-4-large-prompt-test") -> list[dict]:
    """Get every tested (query_prompt, corpus_prompt) combo with its score, sorted best-first."""
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return []

    df = pd.read_csv(results_csv)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return []

    df["prompt"] = df["prompt"].fillna("")
    df["corpus_prompt"] = df.get("corpus_prompt", pd.Series("", index=df.index)).fillna("")

    df = df.sort_values("score", ascending=False)
    return [
        {
            "query_prompt": row["prompt"],
            "corpus_prompt": row["corpus_prompt"],
            "score": round(row["score"], 4),
        }
        for _, row in df.iterrows()
    ]


def get_all_prompt_history(dataset: str, model: str = "voyageai/voyage-4-large-prompt-test") -> tuple[list[dict], list[dict]]:
    """Compute marginal scores for every unique query and corpus prompt.

    Returns (query_history, corpus_history) where each is a sorted list of
    {"prompt": str, "avg_score": float, "n_tests": int}.
    """
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return [], []

    df = pd.read_csv(results_csv)
    df = df[(df["dataset"] == dataset) & (df["score"].notna())]

    if df.empty:
        return [], []

    # Fill NaN for backward compat with old CSVs
    df["prompt"] = df["prompt"].fillna("")
    df["corpus_prompt"] = df.get("corpus_prompt", pd.Series("", index=df.index)).fillna("")

    # Marginal scores for query prompts
    query_groups = df.groupby("prompt")["score"].agg(["mean", "count"]).reset_index()
    query_history = sorted(
        [
            {"prompt": row["prompt"], "avg_score": round(row["mean"], 4), "n_tests": int(row["count"])}
            for _, row in query_groups.iterrows()
        ],
        key=lambda x: x["avg_score"],
        reverse=True,
    )

    # Marginal scores for corpus prompts
    corpus_groups = df.groupby("corpus_prompt")["score"].agg(["mean", "count"]).reset_index()
    corpus_history = sorted(
        [
            {"prompt": row["corpus_prompt"], "avg_score": round(row["mean"], 4), "n_tests": int(row["count"])}
            for _, row in corpus_groups.iterrows()
        ],
        key=lambda x: x["avg_score"],
        reverse=True,
    )

    return query_history, corpus_history


def analyze_patterns(top: list[dict], worst: list[dict], baseline: float) -> str:
    """Analyze patterns in successful vs unsuccessful prompt combinations."""
    insights = []

    if not top or not worst:
        return ""

    # Analyze query prompt length patterns
    top_q_lengths = [len(str(p.get("query_prompt", ""))) for p in top if p.get("query_prompt")]
    worst_q_lengths = [len(str(p.get("query_prompt", ""))) for p in worst if p.get("query_prompt")]

    if top_q_lengths and worst_q_lengths:
        avg_top_len = sum(top_q_lengths) / len(top_q_lengths)
        avg_worst_len = sum(worst_q_lengths) / len(worst_q_lengths)
        if avg_top_len < avg_worst_len * 0.7:
            insights.append("- Shorter query prompts tend to perform better")
        elif avg_top_len > avg_worst_len * 1.3:
            insights.append("- Longer, more specific query prompts tend to perform better")

    # Check if empty baseline beats prompts
    baseline_in_top = any(
        not p.get("query_prompt") and not p.get("corpus_prompt") for p in top[:3]
    )
    if baseline_in_top:
        insights.append(
            "- Empty baseline (no prompts) is competitive - be careful not to overcomplicate"
        )

    # Check if corpus prompts help
    top_has_corpus = [p for p in top if p.get("corpus_prompt")]
    if top_has_corpus:
        insights.append(f"- {len(top_has_corpus)}/{len(top)} top combos use a corpus prompt")
    else:
        insights.append("- No corpus prompts in top combos yet - room for experimentation")

    return "\n".join(insights) if insights else "- No clear patterns detected yet"


FIX_PROMPT = """The following prompts are INVALID because they don't contain {{text}} as a placeholder.
Every prompt MUST contain {{text}} exactly once — the model replaces it with the actual input.

Invalid prompts:
{invalid_json}

Please rewrite ONLY these prompts so each contains {{text}}. Return a JSON array of the fixed prompts (same order, same count).

Example fixes:
- "Find relevant documents" -> "Find relevant documents: {{text}}"
- "Legal:" -> "Legal: {{text}}"
- "Instruct: Search\\nQuery: " -> "Instruct: Search\\nQuery: {{text}}"
"""


def _fix_invalid_prompts(invalid: list[str], client, backend: str) -> list[str]:
    """Ask LLM to fix prompts that are missing {text}."""
    content = FIX_PROMPT.format(invalid_json=json.dumps(invalid, indent=2))

    if backend == "anthropic":
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text
    else:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = response.choices[0].message.content

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _validate_and_fix_prompts(
    prompts: list[str], label: str, client, backend: str
) -> list[str]:
    """Validate all prompts contain {text}, ask LLM to fix any that don't."""
    invalid = [p for p in prompts if "{text}" not in p]
    if not invalid:
        return prompts

    print(f"  {len(invalid)} {label} missing {{text}}, asking LLM to fix...")
    fixed = _fix_invalid_prompts(invalid, client, backend)

    if len(fixed) == len(invalid):
        fix_idx = 0
        result = []
        for p in prompts:
            if "{text}" not in p and fix_idx < len(fixed):
                new_p = fixed[fix_idx]
                fix_idx += 1
                if "{text}" in new_p:
                    result.append(new_p)
                    print(f"    Fixed: {p!r} -> {new_p!r}")
                else:
                    print(f"    Dropped (still invalid): {p!r}")
            else:
                result.append(p)
        return result
    else:
        print(f"  Fix count mismatch, dropping {len(invalid)} invalid prompts")
        return [p for p in prompts if "{text}" in p]


def _parse_llm_response(text: str) -> dict[str, list[str]] | None:
    """Parse LLM response into prompt dict. Returns None on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "query_prompts" in data:
                return {
                    "query_prompts": data["query_prompts"],
                    "corpus_prompts": data.get("corpus_prompts", ["{text}"]),
                }
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            prompts = json.loads(match.group())
            return {"query_prompts": prompts, "corpus_prompts": ["{text}"]}
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON: {e}")

    return None


def evolve_prompts(
    dataset: str,
    client,
    backend: str = "anthropic",
    model: str = "voyageai/voyage-4-large-prompt-test",
    carry_forward: int = 3,
) -> dict[str, list[str]] | None:
    """Generate evolved query + corpus prompts using full history.

    Args:
        carry_forward: Number of top-performing prompts from history to inject
            into the new generation (ensures scores never regress).
    """
    info = DATASET_INFO.get(
        dataset,
        {"domain": "General", "description": dataset, "corpus_type": "documents"},
    )
    baseline = BASELINES.get(dataset, 0.5)

    # Get full combo history (every tested pair with score)
    full_combo_history = get_full_combo_history(dataset, model=model)
    if not full_combo_history:
        print(f"  No results found for {dataset}")
        return None

    # Also get top/worst for carry-forward and pattern analysis
    top_combos = full_combo_history[:10]
    worst_combos = full_combo_history[-5:]

    # Get marginal history (still needed for carry-forward logic below)
    query_history, corpus_history = get_all_prompt_history(dataset, model=model)

    # Calculate improvements/degradations
    best_score = full_combo_history[0]["score"]
    worst_score = full_combo_history[-1]["score"]
    best_improvement = best_score - baseline
    worst_degradation = worst_score - baseline

    # Analyze patterns
    pattern_insights = analyze_patterns(top_combos, worst_combos, baseline)

    # Get dataset-specific learnings
    dataset_guidance = DATASET_LEARNINGS.get(dataset, "")

    prompt_content = EVOLUTION_PROMPT.format(
        dataset_name=dataset,
        domain=info["domain"],
        description=info["description"],
        corpus_type=info["corpus_type"],
        baseline=baseline,
        full_combo_history_json=json.dumps(full_combo_history, indent=2),
        best_improvement=best_improvement,
        worst_degradation=worst_degradation,
        pattern_insights=pattern_insights,
        dataset_specific_guidance=dataset_guidance,
    )

    if backend == "anthropic":
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt_content}],
        )
        text = response.content[0].text
    else:  # openai
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt_content}],
        )
        text = response.choices[0].message.content

    result = _parse_llm_response(text)
    if not result:
        return None

    # Validate and fix prompts missing {text}
    result["query_prompts"] = _validate_and_fix_prompts(
        result["query_prompts"], "query prompts", client, backend
    )
    result["corpus_prompts"] = _validate_and_fix_prompts(
        result["corpus_prompts"], "corpus prompts", client, backend
    )

    # Carry forward top-performing prompts from history to prevent regression.
    # Already-tested combos will be skipped by filter_already_tested() in the
    # test runner, so this doesn't waste compute.
    if carry_forward > 0:
        carry_query = set()
        carry_corpus = set()

        # From top combos: the exact prompts that produced the best scores
        for combo in top_combos[:carry_forward]:
            qp = combo.get("query_prompt", "")
            cp = combo.get("corpus_prompt", "")
            if qp and "{text}" in qp:
                carry_query.add(qp)
            if cp and "{text}" in cp:
                carry_corpus.add(cp)

        # From marginal history: best average performers
        if query_history:
            for h in query_history[:carry_forward]:
                if h["prompt"] and "{text}" in h["prompt"]:
                    carry_query.add(h["prompt"])
        if corpus_history:
            for h in corpus_history[:carry_forward]:
                if h["prompt"] and "{text}" in h["prompt"]:
                    carry_corpus.add(h["prompt"])

        # Add top performers that aren't already in the new set
        for qp in carry_query:
            if qp not in result["query_prompts"]:
                result["query_prompts"].append(qp)
                print(f"  Carried forward query prompt: {qp[:60]}")
        for cp in carry_corpus:
            if cp not in result["corpus_prompts"]:
                result["corpus_prompts"].append(cp)
                print(f"  Carried forward corpus prompt: {cp[:60]}")

    return result


def save_evolved_prompts(dataset: str, prompts: dict[str, list[str]], generation: int, model: str = "voyage-4-large"):
    """Save evolved prompts to model-specific directory."""
    model_dir = PROMPTS_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_dir / f"{dataset}_gen{generation}.json"
    output_file.write_text(json.dumps(prompts, indent=2))
    n_q = len(prompts.get("query_prompts", []))
    n_c = len(prompts.get("corpus_prompts", []))
    print(f"  Saved {n_q} query + {n_c} corpus prompts to {output_file}")


def determine_next_generation(dataset: str, model: str = "voyage-4-large") -> int:
    """Determine the next generation number based on existing files."""
    model_dir = PROMPTS_DIR / model
    gen = 1
    while (model_dir / f"{dataset}_gen{gen}.json").exists():
        gen += 1
    return gen


def check_convergence(dataset: str, threshold: float = 0.001, model: str = "voyageai/voyage-4-large-prompt-test") -> bool:
    """Check if optimization has converged (improvement is minimal)."""
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return False

    df = pd.read_csv(results_csv)
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


# Keep these for backward compat with run_all.py imports
def get_top_prompts(dataset: str, n: int = 5, model: str = "voyageai/voyage-4-large-prompt-test") -> list[dict]:
    """Get top N performing prompt combos (backward compat wrapper)."""
    return get_top_combos(dataset, n, model)


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
    parser.add_argument(
        "--model",
        type=str,
        default="voyage-4-large",
        help="Model short name for prompt directory (default: voyage-4-large)",
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
        results_csv = get_results_csv(args.model)
        if not results_csv.exists():
            print("No results found. Run tests first.")
            return

        df = pd.read_csv(results_csv)
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
        if args.check_convergence and check_convergence(dataset, model=args.model):
            print(f"  Skipping {dataset}: already converged")
            continue

        baseline = BASELINES.get(dataset, 0.5)

        # Show current state
        top = get_top_combos(dataset, n=3, model=args.model)
        if not top:
            print(f"  No results found for {dataset}, skipping")
            continue

        best_score = top[0]["score"]
        improvement = best_score - baseline
        print(f"  Baseline: {baseline:.2%}")
        print(f"  Best so far: {best_score:.2%} ({improvement:+.2%})")

        print(f"\n  Top combinations:")
        for p in top:
            qp = p["query_prompt"][:30] + "..." if len(str(p["query_prompt"])) > 30 else p["query_prompt"]
            cp = p["corpus_prompt"][:30] + "..." if len(str(p["corpus_prompt"])) > 30 else p["corpus_prompt"]
            print(f"    {p['score']:.4f}: Q={qp or '(empty)'} | C={cp or '(empty)'}")

        # Determine generation
        generation = args.generation or determine_next_generation(dataset, args.model)
        print(f"\n  Creating generation {generation} for {args.model}...")

        # Generate evolved prompts
        evolved = evolve_prompts(dataset, client, backend, model=args.model)
        if evolved:
            save_evolved_prompts(dataset, evolved, generation, args.model)
            n_q = len(evolved.get("query_prompts", []))
            n_c = len(evolved.get("corpus_prompts", []))
            print(f"  Success: {n_q} query + {n_c} corpus prompts ready for testing")
        else:
            print(f"  Failed to generate evolved prompts")


if __name__ == "__main__":
    main()
