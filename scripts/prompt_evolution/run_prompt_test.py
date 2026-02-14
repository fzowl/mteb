#!/usr/bin/env python3
"""
Script-based prompt testing for voyage-4-large.
Uses the MTEB Python API to test different prompts.

Usage:
    # Test all prompts for a dataset
    python run_prompt_test.py --dataset AILACasedocs

    # Test specific prompt range
    python run_prompt_test.py --dataset AILACasedocs --start 0 --count 5

    # Test a single prompt by index
    python run_prompt_test.py --dataset AILACasedocs --index 3

    # Test with custom prompt (not from file)
    python run_prompt_test.py --dataset AILACasedocs --prompt "Find relevant legal documents"

    # Parallel execution with multiple workers
    python run_prompt_test.py --dataset AILACasedocs --workers 3
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

# Add project root to path for mteb imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
PROMPTS_DIR = SCRIPT_DIR / "prompts"
RESULTS_CSV = SCRIPT_DIR / "results.csv"

# Environment variables required for specific datasets (local datasets)
DATASET_ENV_VARS = {
    "CUREv1RetrievalLocal": {
        "CUREV1_LOCAL_PATH": str(PROJECT_ROOT / "data" / "subsets" / "CUREv1_subset"),
    },
    "MIRACLRetrievalHardNegativesLocal": {
        "MIRACL_LOCAL_PATH": str(PROJECT_ROOT / "data" / "subsets" / "MIRACLRetrievalHardNegatives_subset"),
    },
}


def model_short_name(model: str) -> str:
    """Extract short model name from full model identifier.

    E.g. 'voyageai/voyage-4-nano-prompt-test' -> 'voyage-4-nano'
         'voyageai/voyage-4-large-prompt-test' -> 'voyage-4-large'
         'voyage-4-nano' -> 'voyage-4-nano'
    """
    name = model.split("/")[-1]  # strip org prefix
    name = name.replace("-prompt-test", "")
    return name


def get_prompts_file(dataset: str, generation: int = 0, model: str | None = None) -> Path:
    """Get the prompts file for a dataset, generation, and model.

    Gen0 (seed) prompts are model-agnostic: prompts/seed/{dataset}.json
    GenN prompts are model-specific: prompts/{model_short}/{dataset}_genN.json
    """
    if generation == 0:
        return PROMPTS_DIR / "seed" / f"{dataset}.json"
    short = model_short_name(model) if model else "voyage-4-large"
    return PROMPTS_DIR / short / f"{dataset}_gen{generation}.json"


def load_prompts(dataset: str, generation: int = 0, model: str | None = None) -> list[str]:
    """Load prompts from JSON file."""
    prompts_file = get_prompts_file(dataset, generation, model)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    return json.loads(prompts_file.read_text())


def _run_single_test_worker(args: tuple) -> dict:
    """Worker function for parallel execution.

    This runs in a separate process with its own environment.
    """
    dataset, prompt, prompt_id, batch_size, model_name = args

    # Set environment variables for this process
    os.environ["PROMPT_TEST_DATASET"] = dataset
    os.environ["PROMPT_TEST_PROMPT"] = prompt if prompt else ""

    # Add project root to path for this worker process
    import sys
    from pathlib import Path
    worker_script_dir = Path(__file__).parent
    worker_project_root = worker_script_dir.parent.parent
    if str(worker_project_root) not in sys.path:
        sys.path.insert(0, str(worker_project_root))

    # Set dataset-specific environment variables (for local datasets)
    if dataset in DATASET_ENV_VARS:
        for env_key, env_value in DATASET_ENV_VARS[dataset].items():
            os.environ[env_key] = env_value
            print(f"[{prompt_id}] Set {env_key}={env_value}")

    # Import mteb here to ensure fresh model loading with new env vars
    import mteb

    prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
    prompt_display = prompt_display or "(empty baseline)"
    print(f"[{prompt_id}] Testing: {prompt_display}")

    try:
        # Load model - this will pick up the env vars
        model = mteb.get_model(model_name)

        # Get the task
        tasks = mteb.get_tasks(tasks=[dataset])

        # Run evaluation (always overwrite to avoid caching issues with different prompts)
        results = mteb.evaluate(
            model,
            tasks=tasks,
            encode_kwargs={"batch_size": batch_size},
            show_progress_bar=False,  # Reduce noise in parallel execution
            overwrite_strategy="always",  # Force fresh run for each prompt
        )

        # Extract score from results
        score = None
        if results:
            for result in results:
                if hasattr(result, "scores") and result.scores:
                    # First try standard "test" key (most datasets)
                    test_scores = result.scores.get("test", [])
                    if test_scores:
                        score = test_scores[0].get("main_score")
                        break

                    # For multi-subset datasets (CUREv1, MIRACL), aggregate scores
                    # These have subset keys like "emergency_medicine", "ar", "en", etc.
                    all_main_scores = []
                    for subset_key, subset_data in result.scores.items():
                        if isinstance(subset_data, list) and subset_data:
                            # Each subset has a list of dicts with language/split results
                            for split_data in subset_data:
                                if isinstance(split_data, dict):
                                    # Check if this dict has main_score directly
                                    if "main_score" in split_data:
                                        all_main_scores.append(split_data["main_score"])
                                    else:
                                        # Check nested language keys (e.g., 'en', 'es')
                                        for lang_key, lang_data in split_data.items():
                                            if isinstance(lang_data, dict) and "main_score" in lang_data:
                                                all_main_scores.append(lang_data["main_score"])

                    if all_main_scores:
                        score = sum(all_main_scores) / len(all_main_scores)
                        print(f"[{prompt_id}] Aggregated {len(all_main_scores)} subset scores")
                        break

        print(f"[{prompt_id}] Score: {score:.4f}" if score else f"[{prompt_id}] FAILED")

        return {
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "score": score,
            "error": None,
        }

    except Exception as e:
        print(f"[{prompt_id}] Error: {e}")
        return {
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "score": None,
            "error": str(e)[:500],
        }


def run_single_test(
    dataset: str,
    prompt: str,
    prompt_id: str,
    batch_size: int = 1000,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
) -> dict:
    """Run a single mteb test with a specific prompt using Python API."""
    return _run_single_test_worker((dataset, prompt, prompt_id, batch_size, model_name))


def log_result(result: dict):
    """Log a result to the CSV file (thread-safe via file locking)."""
    import fcntl

    file_exists = RESULTS_CSV.exists()
    fieldnames = ["timestamp", "dataset", "prompt_id", "prompt", "score", "error"]

    with open(RESULTS_CSV, "a", newline="") as f:
        # Use file locking for thread safety
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists and f.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": result["dataset"],
                    "prompt_id": result["prompt_id"],
                    "prompt": result["prompt"],
                    "score": result["score"],
                    "error": result.get("error"),
                }
            )
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def run_tests_sequential(
    dataset: str,
    prompts: list[tuple[str, str]],
    batch_size: int = 1000,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
) -> list[dict]:
    """Run tests sequentially for a list of prompts."""
    results = []

    for i, (prompt_id, prompt) in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Testing {prompt_id}")
        result = run_single_test(dataset, prompt, prompt_id, batch_size, model_name)
        results.append(result)
        log_result(result)

    return results


def run_tests_parallel(
    dataset: str,
    prompts: list[tuple[str, str]],
    batch_size: int = 1000,
    workers: int = 3,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
) -> list[dict]:
    """Run tests in parallel using multiple worker processes."""
    results = []

    # Prepare work items
    work_items = [
        (dataset, prompt, prompt_id, batch_size, model_name) for prompt_id, prompt in prompts
    ]

    print(f"Starting {workers} workers for {len(work_items)} prompts...")

    # Use 'spawn' context for clean process isolation
    ctx = get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        # Submit all work
        future_to_prompt = {
            executor.submit(_run_single_test_worker, item): item[2]  # prompt_id
            for item in work_items
        }

        # Collect results as they complete
        for future in as_completed(future_to_prompt):
            prompt_id = future_to_prompt[future]
            try:
                result = future.result()
                results.append(result)
                log_result(result)
            except Exception as e:
                print(f"[{prompt_id}] Worker exception: {e}")
                results.append(
                    {
                        "dataset": dataset,
                        "prompt_id": prompt_id,
                        "prompt": "",
                        "score": None,
                        "error": str(e),
                    }
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test prompts for voyage-4-large")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index in prompts list"
    )
    parser.add_argument(
        "--count", type=int, default=None, help="Number of prompts to test"
    )
    parser.add_argument("--index", type=int, help="Test only this prompt index")
    parser.add_argument("--prompt", type=str, help="Test a custom prompt string")
    parser.add_argument(
        "--generation", type=int, default=0, help="Prompt generation (0=initial)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for mteb"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voyageai/voyage-4-large-prompt-test",
        help="Model to test (default: voyageai/voyage-4-large-prompt-test)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available prompts and exit"
    )
    args = parser.parse_args()

    # Handle --list
    if args.list:
        try:
            prompts = load_prompts(args.dataset, args.generation, args.model)
            print(f"Prompts for {args.dataset} (gen {args.generation}):")
            for i, p in enumerate(prompts):
                display = p[:60] + "..." if len(p) > 60 else p
                display = display or "(empty baseline)"
                print(f"  [{i:2d}] {display}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    # Handle custom prompt
    if args.prompt is not None:
        prompt_tuples = [("custom", args.prompt)]
    elif args.index is not None:
        prompts = load_prompts(args.dataset, args.generation, args.model)
        if args.index >= len(prompts):
            print(f"Error: Index {args.index} out of range (max {len(prompts) - 1})")
            return
        prompt_tuples = [
            (f"gen{args.generation}_{args.index:02d}", prompts[args.index])
        ]
    else:
        prompts = load_prompts(args.dataset, args.generation, args.model)
        # Slice prompts
        end = args.start + args.count if args.count else None
        prompts = prompts[args.start : end]
        # Create (id, prompt) tuples
        prompt_tuples = [
            (f"gen{args.generation}_{i:02d}", p)
            for i, p in enumerate(prompts, args.start)
        ]

    print(f"Testing {len(prompt_tuples)} prompts for {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Results will be logged to: {RESULTS_CSV}")

    if args.workers > 1:
        results = run_tests_parallel(
            args.dataset, prompt_tuples, args.batch_size, args.workers, args.model
        )
    else:
        results = run_tests_sequential(args.dataset, prompt_tuples, args.batch_size, args.model)

    # Summary
    successful = [r for r in results if r["score"] is not None]
    if successful:
        best = max(successful, key=lambda x: x["score"])
        print(f"\n{'=' * 60}")
        print(f"Summary: {len(successful)}/{len(results)} successful")
        print(f"Best score: {best['score']:.4f} (prompt: {best['prompt_id']})")
        print(f"{'=' * 60}")

    print(f"\nDone! Results logged to {RESULTS_CSV}")
    print("Run 'python results_tracker.py' to analyze results.")


if __name__ == "__main__":
    main()
