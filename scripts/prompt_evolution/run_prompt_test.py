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
RESULTS_DIR = SCRIPT_DIR / "results"


def get_results_csv(model: str) -> Path:
    """Get model-specific results CSV path."""
    short = model_short_name(model)
    results_dir = RESULTS_DIR / short
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "results.csv"

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


def load_prompts(dataset: str, generation: int = 0, model: str | None = None) -> dict[str, list[str]]:
    """Load prompts from JSON file. Returns dict with query_prompts and corpus_prompts.

    Backward compatible: if the file contains a flat list, wraps it as query_prompts
    with an empty-only corpus_prompts list.
    """
    prompts_file = get_prompts_file(dataset, generation, model)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    data = json.loads(prompts_file.read_text())
    if isinstance(data, list):
        # Old format: flat list of query prompts
        return {"query_prompts": data, "corpus_prompts": ["{text}"]}
    if isinstance(data, dict) and "query_prompts" in data:
        return data
    raise ValueError(f"Unexpected prompts format in {prompts_file}")


def build_prompt_matrix(
    query_prompts: list[str], corpus_prompts: list[str], generation: int
) -> list[tuple[str, str, str, str]]:
    """Build cross-product of query x corpus prompts.

    Returns list of (query_prompt_id, query_prompt, corpus_prompt_id, corpus_prompt).
    """
    matrix = []
    for qi, qp in enumerate(query_prompts):
        for ci, cp in enumerate(corpus_prompts):
            qid = f"gen{generation}_q{qi:02d}"
            cid = f"gen{generation}_c{ci:02d}"
            matrix.append((qid, qp, cid, cp))
    return matrix


def _run_single_test_worker(args: tuple) -> dict:
    """Worker function for parallel execution.

    This runs in a separate process with its own environment.
    Args tuple: (dataset, prompt, prompt_id, corpus_prompt, corpus_prompt_id, batch_size, model_name, gpu_id)
    """
    dataset, prompt, prompt_id, corpus_prompt, corpus_prompt_id, batch_size, model_name, gpu_id = args

    # Pin this worker to a specific GPU (if assigned)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set environment variables for this process
    os.environ["PROMPT_TEST_DATASET"] = dataset
    os.environ["PROMPT_TEST_PROMPT"] = prompt if prompt else "{text}"
    os.environ["PROMPT_TEST_CORPUS_PROMPT"] = corpus_prompt if corpus_prompt else "{text}"

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
            print(f"[{prompt_id}+{corpus_prompt_id}] Set {env_key}={env_value}")

    # Import mteb here to ensure fresh model loading with new env vars
    import mteb

    combo_id = f"{prompt_id}+{corpus_prompt_id}"
    prompt_display = prompt[:40] + "..." if len(prompt) > 40 else prompt
    prompt_display = prompt_display or "(empty)"
    corpus_display = corpus_prompt[:40] + "..." if len(corpus_prompt) > 40 else corpus_prompt
    corpus_display = corpus_display or "(empty)"
    print(f"[{combo_id}] Q: {prompt_display} | C: {corpus_display}")

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
            show_progress_bar=False,
            overwrite_strategy="always",
        )

        # Extract score from results
        score = None
        if results:
            for result in results:
                if hasattr(result, "scores") and result.scores:
                    test_scores = result.scores.get("test", [])
                    if test_scores:
                        score = test_scores[0].get("main_score")
                        break

                    # For multi-subset datasets, aggregate scores
                    all_main_scores = []
                    for subset_key, subset_data in result.scores.items():
                        if isinstance(subset_data, list) and subset_data:
                            for split_data in subset_data:
                                if isinstance(split_data, dict):
                                    if "main_score" in split_data:
                                        all_main_scores.append(split_data["main_score"])
                                    else:
                                        for lang_key, lang_data in split_data.items():
                                            if isinstance(lang_data, dict) and "main_score" in lang_data:
                                                all_main_scores.append(lang_data["main_score"])

                    if all_main_scores:
                        score = sum(all_main_scores) / len(all_main_scores)
                        print(f"[{combo_id}] Aggregated {len(all_main_scores)} subset scores")
                        break

        print(f"[{combo_id}] Score: {score:.4f}" if score else f"[{combo_id}] FAILED")

        return {
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "corpus_prompt_id": corpus_prompt_id,
            "corpus_prompt": corpus_prompt,
            "score": score,
            "error": None,
        }

    except Exception as e:
        print(f"[{combo_id}] Error: {e}")
        return {
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "corpus_prompt_id": corpus_prompt_id,
            "corpus_prompt": corpus_prompt,
            "score": None,
            "error": str(e)[:500],
        }


def run_single_test(
    dataset: str,
    prompt: str,
    prompt_id: str,
    corpus_prompt: str = "",
    corpus_prompt_id: str = "gen0_c00",
    batch_size: int = 1000,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
    gpu_id: int | None = None,
) -> dict:
    """Run a single mteb test with a specific prompt using Python API."""
    return _run_single_test_worker((dataset, prompt, prompt_id, corpus_prompt, corpus_prompt_id, batch_size, model_name, gpu_id))


def log_result(result: dict, model: str = "voyageai/voyage-4-large-prompt-test"):
    """Log a result to the model-specific CSV file (thread-safe via file locking)."""
    import fcntl

    results_csv = get_results_csv(model)
    file_exists = results_csv.exists()
    fieldnames = ["timestamp", "dataset", "prompt_id", "prompt", "corpus_prompt_id", "corpus_prompt", "score", "error"]

    with open(results_csv, "a", newline="") as f:
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
                    "corpus_prompt_id": result.get("corpus_prompt_id", ""),
                    "corpus_prompt": result.get("corpus_prompt", ""),
                    "score": result["score"],
                    "error": result.get("error"),
                }
            )
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_already_tested(dataset: str, model: str) -> set[tuple[str, str]]:
    """Load (query_prompt, corpus_prompt) pairs already tested with a valid score.

    Returns a set of (query_prompt, corpus_prompt) tuples that have been
    successfully tested (non-null score) so we can skip re-testing them.
    """
    results_csv = get_results_csv(model)
    if not results_csv.exists():
        return set()
    tested = set()
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == dataset and row.get("score"):
                qp = row.get("prompt", "")
                cp = row.get("corpus_prompt", "")
                tested.add((qp, cp))
    return tested


def filter_already_tested(
    dataset: str,
    prompts: list[tuple[str, str, str, str]],
    model: str,
) -> list[tuple[str, str, str, str]]:
    """Remove prompt combos that have already been successfully tested."""
    tested = get_already_tested(dataset, model)
    if not tested:
        return prompts
    filtered = []
    skipped = 0
    for qid, qp, cid, cp in prompts:
        if (qp, cp) in tested:
            skipped += 1
        else:
            filtered.append((qid, qp, cid, cp))
    if skipped:
        print(f"  Skipping {skipped} already-tested combos ({len(filtered)} remaining)")
    return filtered


def run_tests_sequential(
    dataset: str,
    prompts: list[tuple[str, str, str, str]],
    batch_size: int = 1000,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
) -> list[dict]:
    """Run tests sequentially for a list of prompt combinations.

    Each item in prompts is (query_prompt_id, query_prompt, corpus_prompt_id, corpus_prompt).
    """
    results = []

    for i, (prompt_id, prompt, corpus_prompt_id, corpus_prompt) in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Testing {prompt_id}+{corpus_prompt_id}")
        result = run_single_test(
            dataset, prompt, prompt_id, corpus_prompt, corpus_prompt_id,
            batch_size, model_name,
        )
        results.append(result)
        log_result(result, model_name)

    return results


def _persistent_worker(gpu_id: int, task_queue, result_queue, dataset: str, batch_size: int, model_name: str):
    """Persistent worker process: loads model once, processes many prompt combos.

    Runs on a single GPU, pulls tasks from task_queue, pushes results to result_queue.
    Exits when it receives None from the queue.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Add project root to path
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

    import mteb

    # Load model ONCE
    print(f"[GPU {gpu_id}] Loading model {model_name}...")
    os.environ["PROMPT_TEST_DATASET"] = dataset
    os.environ["PROMPT_TEST_PROMPT"] = "{text}"
    os.environ["PROMPT_TEST_CORPUS_PROMPT"] = "{text}"
    model = mteb.get_model(model_name)
    print(f"[GPU {gpu_id}] Model loaded, ready for tasks")

    while True:
        task = task_queue.get()
        if task is None:
            break

        prompt_id, prompt, corpus_prompt_id, corpus_prompt = task
        combo_id = f"{prompt_id}+{corpus_prompt_id}"

        # Set prompts via env vars (the model reads these on each encode call)
        os.environ["PROMPT_TEST_PROMPT"] = prompt if prompt else "{text}"
        os.environ["PROMPT_TEST_CORPUS_PROMPT"] = corpus_prompt if corpus_prompt else "{text}"

        prompt_display = (prompt[:40] + "...") if len(prompt) > 40 else (prompt or "(empty)")
        corpus_display = (corpus_prompt[:40] + "...") if len(corpus_prompt) > 40 else (corpus_prompt or "(empty)")
        print(f"[{combo_id}] Q: {prompt_display} | C: {corpus_display}")

        try:
            # IMPORTANT: Create fresh task objects each time — mteb.evaluate() mutates them
            fresh_tasks = mteb.get_tasks(tasks=[dataset])
            results = mteb.evaluate(
                model,
                tasks=fresh_tasks,
                encode_kwargs={"batch_size": batch_size},
                show_progress_bar=False,
                overwrite_strategy="always",
            )

            score = None
            if results:
                for result in results:
                    if hasattr(result, "scores") and result.scores:
                        test_scores = result.scores.get("test", [])
                        if test_scores:
                            score = test_scores[0].get("main_score")
                            break

                        all_main_scores = []
                        for subset_key, subset_data in result.scores.items():
                            if isinstance(subset_data, list) and subset_data:
                                for split_data in subset_data:
                                    if isinstance(split_data, dict):
                                        if "main_score" in split_data:
                                            all_main_scores.append(split_data["main_score"])
                                        else:
                                            for lang_key, lang_data in split_data.items():
                                                if isinstance(lang_data, dict) and "main_score" in lang_data:
                                                    all_main_scores.append(lang_data["main_score"])

                        if all_main_scores:
                            score = sum(all_main_scores) / len(all_main_scores)
                            break

            print(f"[{combo_id}] Score: {score:.4f}" if score else f"[{combo_id}] FAILED")

            result_queue.put({
                "dataset": dataset,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "corpus_prompt_id": corpus_prompt_id,
                "corpus_prompt": corpus_prompt,
                "score": score,
                "error": None,
            })

        except Exception as e:
            print(f"[{combo_id}] Error: {e}")
            result_queue.put({
                "dataset": dataset,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "corpus_prompt_id": corpus_prompt_id,
                "corpus_prompt": corpus_prompt,
                "score": None,
                "error": str(e)[:500],
            })


def run_tests_parallel(
    dataset: str,
    prompts: list[tuple[str, str, str, str]],
    batch_size: int = 1000,
    workers: int = 3,
    model_name: str = "voyageai/voyage-4-large-prompt-test",
    num_gpus: int | None = None,
) -> list[dict]:
    """Run tests in parallel using persistent worker processes.

    Each worker loads the model once on its assigned GPU and processes
    multiple prompt combos from a shared queue. This avoids the overhead
    of reloading the model for each test.

    Each item in prompts is (query_prompt_id, query_prompt, corpus_prompt_id, corpus_prompt).
    """
    from multiprocessing import Process, Queue

    # Determine available GPUs
    if num_gpus is None:
        # Infer from CUDA_VISIBLE_DEVICES or default to workers count
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            gpu_ids = [int(g) for g in cuda_devices.split(",")]
        else:
            gpu_ids = list(range(workers))
    else:
        gpu_ids = list(range(num_gpus))

    # Use at most as many workers as GPUs (1 worker per GPU)
    actual_workers = min(workers, len(gpu_ids))
    gpu_ids = gpu_ids[:actual_workers]

    print(f"Starting {actual_workers} persistent workers on GPUs {gpu_ids} for {len(prompts)} combos...")

    task_queue = Queue()
    result_queue = Queue()

    # Start persistent worker processes
    worker_procs = []
    for i in range(actual_workers):
        p = Process(
            target=_persistent_worker,
            args=(gpu_ids[i], task_queue, result_queue, dataset, batch_size, model_name),
        )
        p.start()
        worker_procs.append(p)

    # Enqueue all tasks
    for prompt_tuple in prompts:
        task_queue.put(prompt_tuple)

    # Send poison pills to stop workers
    for _ in range(actual_workers):
        task_queue.put(None)

    # Collect results
    results = []
    for _ in range(len(prompts)):
        result = result_queue.get()
        results.append(result)
        log_result(result, model_name)

    # Wait for workers to finish
    for p in worker_procs:
        p.join()

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
        "--batch-size", type=int, default=1, help="Batch size for mteb"
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
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to distribute workers across (round-robin assignment)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available prompts and exit"
    )
    args = parser.parse_args()

    # Handle --list
    if args.list:
        try:
            prompt_data = load_prompts(args.dataset, args.generation, args.model)
            print(f"Query prompts for {args.dataset} (gen {args.generation}):")
            for i, p in enumerate(prompt_data["query_prompts"]):
                display = p[:60] + "..." if len(p) > 60 else p
                display = display or "(empty baseline)"
                print(f"  [q{i:02d}] {display}")
            print(f"\nCorpus prompts for {args.dataset} (gen {args.generation}):")
            for i, p in enumerate(prompt_data["corpus_prompts"]):
                display = p[:60] + "..." if len(p) > 60 else p
                display = display or "(empty baseline)"
                print(f"  [c{i:02d}] {display}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    # Handle custom prompt
    if args.prompt is not None:
        prompt_tuples = [("custom_q00", args.prompt, "custom_c00", "")]
    elif args.index is not None:
        prompt_data = load_prompts(args.dataset, args.generation, args.model)
        qp = prompt_data["query_prompts"]
        if args.index >= len(qp):
            print(f"Error: Index {args.index} out of range (max {len(qp) - 1})")
            return
        # Test single query prompt against all corpus prompts
        prompt_tuples = build_prompt_matrix(
            [qp[args.index]], prompt_data["corpus_prompts"], args.generation
        )
        # Fix query prompt id to reflect the actual index
        prompt_tuples = [
            (f"gen{args.generation}_q{args.index:02d}", t[1], t[2], t[3])
            for t in prompt_tuples
        ]
    else:
        prompt_data = load_prompts(args.dataset, args.generation, args.model)
        query_prompts = prompt_data["query_prompts"]
        corpus_prompts = prompt_data["corpus_prompts"]
        # Slice query prompts
        end = args.start + args.count if args.count else None
        query_prompts = query_prompts[args.start : end]
        # Build cross-product matrix
        prompt_tuples = build_prompt_matrix(query_prompts, corpus_prompts, args.generation)

    n_combos = len(prompt_tuples)
    print(f"Testing {n_combos} prompt combinations for {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    results_csv = get_results_csv(args.model)
    print(f"Results will be logged to: {results_csv}")

    if args.workers > 1:
        results = run_tests_parallel(
            args.dataset, prompt_tuples, args.batch_size, args.workers, args.model, args.gpus
        )
    else:
        results = run_tests_sequential(args.dataset, prompt_tuples, args.batch_size, args.model)

    # Summary
    successful = [r for r in results if r["score"] is not None]
    if successful:
        best = max(successful, key=lambda x: x["score"])
        print(f"\n{'=' * 60}")
        print(f"Summary: {len(successful)}/{len(results)} successful")
        print(f"Best score: {best['score']:.4f} (Q: {best['prompt_id']}, C: {best.get('corpus_prompt_id', '')})")
        print(f"{'=' * 60}")

    print(f"\nDone! Results logged to {results_csv}")
    print("Run 'python results_tracker.py' to analyze results.")


if __name__ == "__main__":
    main()
