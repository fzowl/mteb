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

GENERATION_PROMPT = """Generate 25 diverse retrieval instruction prompts for the following dataset:

Dataset: {dataset_name}
Domain: {domain}
Task: {description}
Corpus Type: {corpus_type}
Current Score: {current_score:.1%}

The model uses this format: "Instruct: {{instruction}}\\nQuery: " + user_query

Requirements for the 25 prompts:
1. Each prompt should be a single instruction that will be prepended to the user's query
2. The prompt replaces {{instruction}} in the template above
3. Vary across these dimensions:
   - Action verbs: retrieve, find, search, locate, identify, match, discover, fetch, obtain, surface
   - Query framing: "Given a...", "For the...", "Based on...", "Using...", "Search for..."
   - Specificity: generic vs domain-specific language
   - Output description: what to return and why
   - Length: short (5-10 words), medium (10-20 words), long (20-30 words)
4. Include both generic and highly domain-specific prompts
5. Include prompts that emphasize different aspects:
   - Relevance/similarity focus
   - Answering capability focus
   - Precision vs recall tradeoff
   - Semantic matching
   - Exact matching
6. Include some experimental/creative prompts that might work well

Return ONLY a JSON array of 25 strings. Include "" (empty string) as the first option for baseline comparison.

Example format:
[
  "",
  "Given a legal scenario, retrieve relevant case documents",
  "Find case precedents that match the legal issue described",
  ...
]
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


def generate_prompts(
    dataset_name: str, client, backend: str = "anthropic"
) -> list[str]:
    """Generate diverse prompts for a dataset using LLM."""
    if backend == "anthropic":
        text = generate_prompts_anthropic(dataset_name, client)
    else:
        text = generate_prompts_openai(dataset_name, client)

    # Extract JSON array from response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            prompts = json.loads(match.group())
            # Ensure empty string baseline is first
            if "" not in prompts:
                prompts.insert(0, "")
            elif prompts[0] != "":
                prompts.remove("")
                prompts.insert(0, "")
            return prompts
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON: {e}")
            return [""]
    return [""]


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

    output_dir = Path(__file__).parent / "prompts"
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
        print(f"  Generated {len(prompts)} prompts -> {output_file}")


if __name__ == "__main__":
    main()
