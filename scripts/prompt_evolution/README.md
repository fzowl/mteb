# Prompt Evolution Framework for voyage-4-large

Systematic prompt optimization using LLM-assisted generation and evolutionary strategies.

## Quick Start

```bash
cd scripts/prompt_evolution

# Run the full pipeline (generate, test, evolve) for all datasets
python run_all.py --generations 2 --workers 3

# Or step by step:
python prompt_generator.py              # Generate initial prompts
python run_prompt_test.py --dataset AILACasedocs --workers 3
python results_tracker.py               # Analyze
python evolve_prompts.py --all          # Evolve based on results
```

## The Iterative Process

```
┌─────────────────────────────────────────────────────────────┐
│                    Generation 0                              │
│  1. Generate 25 diverse prompts per dataset (LLM-assisted)  │
│  2. Test all prompts against MTEB benchmarks                │
│  3. Analyze: identify winners AND losers                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Generation N                              │
│  1. Learn from results:                                     │
│     - What patterns worked? (top performers)                │
│     - What patterns failed? (worst performers)              │
│  2. Evolve prompts:                                         │
│     - Crossover: Combine winning elements                   │
│     - Mutation: Refine what works, avoid what fails        │
│     - Exploration: Try new creative approaches             │
│  3. Test evolved prompts                                    │
│  4. Repeat until convergence                                │
└─────────────────────────────────────────────────────────────┘
```

## Scripts

### run_all.py - Master Orchestration
Runs the complete pipeline for all datasets.

```bash
# Full pipeline with 2 generations
python run_all.py --generations 2

# Specific datasets only
python run_all.py --datasets AILACasedocs FreshStackRetrieval --workers 5

# Resume from generation 1
python run_all.py --start-generation 1 --generations 2

# Dry run (see what would happen)
python run_all.py --dry-run
```

### prompt_generator.py - Initial Generation
Generate 25 diverse prompts per dataset using Claude API.

```bash
python prompt_generator.py                        # All datasets
python prompt_generator.py AILACasedocs           # Specific dataset
```

### run_prompt_test.py - Testing
Test prompts using MTEB Python API with parallel support.

```bash
python run_prompt_test.py --dataset AILACasedocs --workers 3   # Parallel
python run_prompt_test.py --dataset AILACasedocs --index 5     # Single prompt
python run_prompt_test.py --dataset AILACasedocs --list        # List prompts
```

### evolve_prompts.py - Evolution
Generate evolved prompts that learn from both successes AND failures.

```bash
python evolve_prompts.py --dataset AILACasedocs   # Single dataset
python evolve_prompts.py --all                    # All with results
python evolve_prompts.py --all --check-convergence  # Skip converged
```

### results_tracker.py - Analysis
Analyze results and export winning prompts.

```bash
python results_tracker.py                         # Summary
python results_tracker.py --dataset AILACasedocs --top 10
python results_tracker.py --export                # Python dict output
python results_tracker.py --stats                 # Statistics
```

## Target Datasets

| Dataset | Baseline | Domain | Priority |
|---------|----------|--------|----------|
| AILACasedocs | 47.49% | Legal | 🔴 High |
| FreshStackRetrieval | 50.79% | Code | 🔴 High |
| AILAStatutes | 54.35% | Legal | 🔴 High |
| DS1000Retrieval | 71.19% | Code | 🟡 Medium |
| LegalQuAD | 74.63% | Legal | 🟡 Medium |
| HC3FinanceRetrieval | 76.83% | Finance | 🟡 Medium |
| ChatDoctorRetrieval | 77.20% | Medical | 🟡 Medium |
| LegalSummarization | 78.09% | Legal | 🟡 Medium |
| FinQARetrieval | 88.99% | Finance | 🟢 Low |

## Files

```
scripts/prompt_evolution/
├── run_all.py             # Master orchestration script
├── prompt_generator.py    # Generate initial prompts (LLM)
├── run_prompt_test.py     # Test prompts via MTEB API
├── results_tracker.py     # Analyze and export results
├── evolve_prompts.py      # Evolve prompts (learns from results)
├── run_parallel.sh        # Shell wrapper
├── prompts/               # Generated prompts (JSON)
│   ├── AILACasedocs.json
│   ├── AILACasedocs_gen1.json
│   └── ...
├── results.csv            # Test results log
├── RESULTS.md             # Documented learnings (auto-updated)
└── README.md
```

## How It Works

### 1. Prompt Format
The model uses: `Instruct: {instruction}\nQuery: {user_query}`

### 2. Prompt Injection via Environment Variables
Prompts are passed via:
- `PROMPT_TEST_DATASET`: Dataset name
- `PROMPT_TEST_PROMPT`: The prompt to test

This allows parallel execution with isolated environments.

### 3. Evolutionary Learning
The `evolve_prompts.py` script:
- Analyzes TOP performers to learn what works
- Analyzes WORST performers to learn what to avoid
- Detects patterns (length, words, structure)
- Generates evolved prompts combining successful patterns

### 4. Convergence Detection
The pipeline detects when scores stop improving and can skip converged datasets.

## Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` environment variable
- `VOYAGE_API_KEY` environment variable
- Dependencies: `anthropic`, `pandas`, `mteb`

## Example Workflow

```bash
# Day 1: Initial exploration
export ANTHROPIC_API_KEY=your_key
export VOYAGE_API_KEY=your_key

# Generate and test initial prompts
python run_all.py --datasets AILACasedocs FreshStackRetrieval AILAStatutes --workers 3

# Check results
python results_tracker.py

# Day 2: Evolution
python run_all.py --start-generation 1 --generations 2 --workers 5

# Review findings
cat RESULTS.md

# Export winning prompts
python results_tracker.py --export > winning_prompts.py
```

## Success Criteria

| Level | Target |
|-------|--------|
| Primary | >5% absolute improvement on 3+ datasets |
| Secondary | Improvement on 7+ of 9 datasets |
| Stretch | 90%+ on FinQARetrieval, ChatDoctorRetrieval, LegalSummarization |
