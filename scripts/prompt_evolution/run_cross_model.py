#!/usr/bin/env python3
"""
Cross-model prompt comparison: run the same gen0 seed prompts on all 4 Voyage models.

Tests voyage-4-nano, voyage-4-lite, voyage-4, and voyage-4-large with identical
prompt combinations to understand model-specific prompt sensitivity.

Usage:
    # Run all API models (nano already done)
    python run_cross_model.py --dataset AILAStatutes

    # Run specific models only
    python run_cross_model.py --dataset AILAStatutes --models voyage-4-lite voyage-4

    # Dry run: show what would be tested
    python run_cross_model.py --dataset AILAStatutes --dry-run

    # Run nano (local SentenceTransformer, needs GPU)
    python run_cross_model.py --dataset AILAStatutes --models voyage-4-nano --workers 8
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_prompt_test import (
    build_prompt_matrix,
    filter_already_tested,
    load_prompts,
    log_result,
    run_tests_parallel,
    run_tests_sequential,
)

# Model name -> mteb model identifier
MODEL_MAP = {
    "voyage-4-nano": "voyageai/voyage-4-nano-prompt-test",
    "voyage-4-lite": "voyageai/voyage-4-lite-prompt-test",
    "voyage-4": "voyageai/voyage-4-prompt-test",
    "voyage-4-large": "voyageai/voyage-4-large-prompt-test",
}

# Default: API models only (nano is local and typically already done)
DEFAULT_MODELS = ["voyage-4-lite", "voyage-4", "voyage-4-large"]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model prompt comparison for Voyage models"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. AILAStatutes)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_MAP.keys()),
        help=f"Models to test (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument("--generation", type=int, default=0, help="Prompt generation (default: 0)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for mteb")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (for local models)")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (for local models)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested without running")
    args = parser.parse_args()

    # Load prompts (same for all models at gen0)
    prompt_data = load_prompts(args.dataset, args.generation)
    query_prompts = prompt_data["query_prompts"]
    corpus_prompts = prompt_data["corpus_prompts"]
    all_combos = build_prompt_matrix(query_prompts, corpus_prompts, args.generation)

    print(f"Dataset: {args.dataset}")
    print(f"Generation: {args.generation}")
    print(f"Total prompt combos: {len(all_combos)} ({len(query_prompts)}q x {len(corpus_prompts)}c)")
    print(f"Models to test: {', '.join(args.models)}")
    print()

    for model_short in args.models:
        model_name = MODEL_MAP[model_short]
        print(f"{'=' * 60}")
        print(f"Model: {model_short} ({model_name})")
        print(f"{'=' * 60}")

        # Filter already-tested combos
        remaining = filter_already_tested(args.dataset, all_combos, model_name)

        if not remaining:
            print(f"  All {len(all_combos)} combos already tested. Skipping.\n")
            continue

        print(f"  {len(remaining)} combos to test")

        if args.dry_run:
            for qid, qp, cid, cp in remaining[:5]:
                qd = (qp[:35] + "...") if len(qp) > 35 else qp
                cd = (cp[:35] + "...") if len(cp) > 35 else cp
                print(f"    {qid}+{cid}: Q={qd} | C={cd}")
            if len(remaining) > 5:
                print(f"    ... and {len(remaining) - 5} more")
            print()
            continue

        # Run tests
        is_local = model_short == "voyage-4-nano"
        if is_local and args.workers > 1:
            results = run_tests_parallel(
                args.dataset, remaining, args.batch_size,
                args.workers, model_name, args.gpus,
            )
        else:
            results = run_tests_sequential(
                args.dataset, remaining, args.batch_size, model_name,
            )

        # Summary for this model
        successful = [r for r in results if r["score"] is not None]
        if successful:
            best = max(successful, key=lambda x: x["score"])
            scores = [r["score"] for r in successful]
            avg = sum(scores) / len(scores)
            print(f"\n  {model_short}: {len(successful)}/{len(results)} successful")
            print(f"  Best: {best['score']:.4f} ({best['prompt_id']}+{best['corpus_prompt_id']})")
            print(f"  Avg:  {avg:.4f}")
        else:
            print(f"\n  {model_short}: No successful results")
        print()

    print("Done! Run compare_models.py to analyze results across models.")


if __name__ == "__main__":
    main()
