#!/usr/bin/env python3
"""
LLM-assisted prompt generation for voyage-4-large optimization.
Uses Claude API (default) or OpenAI API as fallback.
"""

import json
import os
import re
from pathlib import Path

# Try Anthropic first, fall back to OpenAI
LLM_BACKEND = None
try:
    import anthropic

    if os.environ.get("ANTHROPIC_API_KEY"):
        LLM_BACKEND = "anthropic"
except ImportError:
    pass

if not LLM_BACKEND:
    try:
        import openai

        if os.environ.get("OPENAI_API_KEY"):
            LLM_BACKEND = "openai"
    except ImportError:
        pass

DATASETS = {
    "AILACasedocs": {
        "domain": "Legal",
        "description": "Legal case document retrieval - given a case scenario, find relevant case documents",
        "corpus_type": "Court case documents, legal precedents",
        "current_score": 0.4749,
    },
    "AILAStatutes": {
        "domain": "Legal",
        "description": "Statute retrieval - given a legal scenario, find relevant statutes",
        "corpus_type": "Legislative statutes and provisions",
        "current_score": 0.5435,
    },
    "FreshStackRetrieval": {
        "domain": "Code",
        "description": "Stack Overflow retrieval - given a programming question, find relevant posts",
        "corpus_type": "Stack Overflow questions and answers",
        "current_score": 0.5079,
    },
    "DS1000Retrieval": {
        "domain": "Code",
        "description": "Data science code retrieval - given a problem, find relevant code",
        "corpus_type": "Python data science code snippets (pandas, numpy, scipy, etc.)",
        "current_score": 0.7119,
    },
    "LegalQuAD": {
        "domain": "Legal",
        "description": "German legal QA - given a legal question, find relevant passages",
        "corpus_type": "German legal documents",
        "current_score": 0.7463,
    },
    "HC3FinanceRetrieval": {
        "domain": "Finance",
        "description": "Finance QA retrieval - given a finance question, find relevant docs",
        "corpus_type": "Financial Q&A documents, ChatGPT vs human comparison",
        "current_score": 0.7683,
    },
    "ChatDoctorRetrieval": {
        "domain": "Medical",
        "description": "Medical Q&A retrieval - given a patient question, find relevant info",
        "corpus_type": "Medical Q&A, patient advice documents",
        "current_score": 0.7720,
    },
    "LegalSummarization": {
        "domain": "Legal",
        "description": "Legal summarization retrieval - find documents for summarization",
        "corpus_type": "Legal documents for summarization",
        "current_score": 0.7809,
    },
    "FinQARetrieval": {
        "domain": "Finance",
        "description": "Financial QA with tables - given a question, find relevant financial data",
        "corpus_type": "Financial reports, tables, earnings data",
        "current_score": 0.8899,
    },
    "FinanceBenchRetrieval": {
        "domain": "Finance",
        "description": "Financial retrieval - given a financial question, find relevant financial information",
        "corpus_type": "Financial document excerpts, annual reports, SEC filings",
        "current_score": 0.9316,
    },
    "HumanEvalRetrieval": {
        "domain": "Code",
        "description": "Code retrieval - given a programming problem description, find the Python implementation",
        "corpus_type": "Python function implementations",
        "current_score": 0.9936,
    },
    "WikiSQLRetrieval": {
        "domain": "Code",
        "description": "SQL retrieval - given a natural language question, find the matching SQL query",
        "corpus_type": "SQL SELECT statements for database queries",
        "current_score": 0.9670,
    },
}

GENERATION_PROMPT = """Generate diverse retrieval prompts for the following dataset. We need TWO types of prompts:
1. **Query prompts** - applied to the user's search query before encoding
2. **Corpus prompts** - applied to each document in the corpus before encoding

Dataset: {dataset_name}
Domain: {domain}
Task: {description}
Corpus Type: {corpus_type}
Current Score: {current_score:.1%}

## Prompt Format
Every non-empty prompt MUST contain {{text}} as placeholder for the actual input sentence.
The model replaces {{text}} with the query or document text before encoding.
- "Instruct: Find relevant statutes\\nQuery: {{text}}" — structured template
- "Legal question: {{text}} — find matching precedents" — text in the middle
- "Find relevant documents: {{text}}" — simple prefix
- "" — empty/passthrough (baseline)

## Requirements for Query Prompts (10 prompts):
1. Each prompt is a template with {{text}} where the user's query goes
2. Vary across: action verbs, framing, specificity, length (short to long)
3. Include both generic and highly domain-specific prompts
4. Include experimental/creative approaches
5. Try different {{text}} placements: beginning, middle, end

## Requirements for Corpus Prompts (10 prompts):
1. Each prompt is a template with {{text}} where the document text goes
2. Types to try:
   - Labels: "Legal document: {{text}}"
   - Structured: "Document type: statute\\nContent: {{text}}"
   - Descriptive: "This is a court case document about {{text}}"
   - Contextual: "The following is a financial report excerpt: {{text}}"
   - Classification: "Category: {domain}\\n{{text}}"
3. Corpus prompts should help the model understand what TYPE of document it's encoding
4. Include variety in approach and length

Return ONLY a JSON object with two arrays.
IMPORTANT: Every prompt string MUST contain {{text}} exactly once. The baseline (passthrough) is just "{{text}}". Include it as the first option in BOTH arrays.

Example format:
{{
  "query_prompts": [
    "{{text}}",
    "Given a legal scenario, retrieve relevant case documents for: {{text}}",
    "Instruct: Find matching statutes\\nQuery: {{text}}",
    ...
  ],
  "corpus_prompts": [
    "{{text}}",
    "Legal document: {{text}}",
    "Document type: statute\\nContent: {{text}}",
    "This is a legislative provision: {{text}}",
    ...
  ]
}}
"""


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


def generate_prompts_anthropic(dataset_name: str, client) -> list[str]:
    """Generate diverse prompts using Claude API."""
    info = DATASETS[dataset_name]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    dataset_name=dataset_name,
                    domain=info["domain"],
                    description=info["description"],
                    corpus_type=info["corpus_type"],
                    current_score=info["current_score"],
                ),
            }
        ],
    )
    return response.content[0].text


def generate_prompts_openai(dataset_name: str, client) -> list[str]:
    """Generate diverse prompts using OpenAI API."""
    info = DATASETS[dataset_name]
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    dataset_name=dataset_name,
                    domain=info["domain"],
                    description=info["description"],
                    corpus_type=info["corpus_type"],
                    current_score=info["current_score"],
                ),
            }
        ],
    )
    return response.choices[0].message.content


def _ensure_baseline_first(prompts: list[str]) -> list[str]:
    """Ensure {text} baseline is first in the list."""
    baseline = "{text}"
    if baseline not in prompts:
        prompts.insert(0, baseline)
    elif prompts[0] != baseline:
        prompts.remove(baseline)
        prompts.insert(0, baseline)
    return prompts


def _find_invalid_prompts(prompts: list[str]) -> list[str]:
    """Return prompts that don't contain {text}."""
    return [p for p in prompts if "{text}" not in p]


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
    invalid = _find_invalid_prompts(prompts)
    if not invalid:
        return prompts

    print(f"  {len(invalid)} {label} missing {{text}}, asking LLM to fix...")
    fixed = _fix_invalid_prompts(invalid, client, backend)

    if len(fixed) == len(invalid):
        # Replace invalid prompts with fixed versions
        invalid_set = {id(p): i for i, p in enumerate(invalid)}
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
                    # Still broken, drop it
                    print(f"    Dropped (still invalid): {p!r}")
            else:
                result.append(p)
        return result
    else:
        # Fix count mismatch — just drop invalid prompts
        print(f"  Fix count mismatch, dropping {len(invalid)} invalid prompts")
        return [p for p in prompts if "{text}" in p]


def _parse_llm_response(text: str) -> dict[str, list[str]] | None:
    """Parse LLM response into prompt dict. Returns None on failure."""
    # Try JSON object with query_prompts and corpus_prompts
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

    # Fallback: flat JSON array (backward compat)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            prompts = json.loads(match.group())
            return {"query_prompts": prompts, "corpus_prompts": ["{text}"]}
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON: {e}")

    return None


def generate_prompts(
    dataset_name: str, client, backend: str = "anthropic"
) -> dict[str, list[str]]:
    """Generate diverse query and corpus prompts for a dataset using LLM."""
    if backend == "anthropic":
        text = generate_prompts_anthropic(dataset_name, client)
    else:
        text = generate_prompts_openai(dataset_name, client)

    result = _parse_llm_response(text)
    if not result:
        return {"query_prompts": ["{text}"], "corpus_prompts": ["{text}"]}

    # Validate and fix prompts missing {text}
    result["query_prompts"] = _validate_and_fix_prompts(
        result["query_prompts"], "query prompts", client, backend
    )
    result["corpus_prompts"] = _validate_and_fix_prompts(
        result["corpus_prompts"], "corpus prompts", client, backend
    )

    # Ensure baseline is first
    result["query_prompts"] = _ensure_baseline_first(result["query_prompts"])
    result["corpus_prompts"] = _ensure_baseline_first(result["corpus_prompts"])

    return result


def main():
    """Generate prompts for all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate prompts for datasets")
    parser.add_argument("datasets", nargs="*", help="Datasets to generate prompts for")
    parser.add_argument(
        "--backend",
        choices=["anthropic", "openai", "auto"],
        default="auto",
        help="LLM backend to use (default: auto)",
    )
    args = parser.parse_args()

    # Determine which backend to use
    backend = None
    client = None

    if args.backend in ("anthropic", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            backend = "anthropic"
            print("Using Anthropic API (Claude)")
        except ImportError:
            if args.backend == "anthropic":
                print("Error: anthropic package not installed")
                return

    if (
        not backend
        and args.backend in ("openai", "auto")
        and os.environ.get("OPENAI_API_KEY")
    ):
        try:
            import openai

            client = openai.OpenAI()
            backend = "openai"
            print("Using OpenAI API (GPT-4)")
        except ImportError:
            if args.backend == "openai":
                print("Error: openai package not installed")
                return

    if not backend:
        print("Error: No LLM API available.")
        print("Set one of:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  export OPENAI_API_KEY=your_key")
        print("Or use --backend to specify which to use")
        return

    output_dir = Path(__file__).parent / "prompts" / "seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get datasets
    if args.datasets:
        datasets = [d for d in args.datasets if d in DATASETS]
        if not datasets:
            print(f"Unknown datasets. Available: {list(DATASETS.keys())}")
            return
    else:
        datasets = list(DATASETS.keys())

    for dataset in datasets:
        print(f"Generating prompts for {dataset}...")
        prompts = generate_prompts(dataset, client, backend)
        output_file = output_dir / f"{dataset}.json"
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=2)
        n_query = len(prompts["query_prompts"])
        n_corpus = len(prompts["corpus_prompts"])
        print(f"  Generated {n_query} query + {n_corpus} corpus prompts -> {output_file}")


if __name__ == "__main__":
    main()
