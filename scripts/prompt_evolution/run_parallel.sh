#!/bin/bash
# Run prompt tests in parallel
#
# Usage:
#   ./run_parallel.sh AILACasedocs          # Test with 3 workers (default)
#   ./run_parallel.sh AILACasedocs 5        # Test with 5 workers
#   ./run_parallel.sh AILACasedocs 3 1      # Test generation 1 prompts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=$1
WORKERS=${2:-3}
GENERATION=${3:-0}

if [ -z "$DATASET" ]; then
    echo "Usage: ./run_parallel.sh <dataset> [workers] [generation]"
    echo ""
    echo "Arguments:"
    echo "  dataset     Dataset name (required)"
    echo "  workers     Number of parallel workers (default: 3)"
    echo "  generation  Prompt generation to test (default: 0)"
    echo ""
    echo "Examples:"
    echo "  ./run_parallel.sh AILACasedocs"
    echo "  ./run_parallel.sh AILACasedocs 5"
    echo "  ./run_parallel.sh AILACasedocs 3 1"
    echo ""
    echo "Available datasets:"
    echo "  AILACasedocs, AILAStatutes, FreshStackRetrieval, DS1000Retrieval,"
    echo "  LegalQuAD, HC3FinanceRetrieval, ChatDoctorRetrieval,"
    echo "  LegalSummarization, FinQARetrieval"
    exit 1
fi

# Determine prompts file
if [ "$GENERATION" -eq 0 ]; then
    PROMPTS_FILE="$SCRIPT_DIR/prompts/${DATASET}.json"
else
    PROMPTS_FILE="$SCRIPT_DIR/prompts/${DATASET}_gen${GENERATION}.json"
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file not found: $PROMPTS_FILE"
    echo "Generate prompts first with: python prompt_generator.py $DATASET"
    exit 1
fi

# Count prompts
NUM_PROMPTS=$(python3 -c "import json; print(len(json.load(open('$PROMPTS_FILE'))))")
echo "Testing $NUM_PROMPTS prompts for $DATASET (gen $GENERATION) with $WORKERS workers"
echo "Prompts file: $PROMPTS_FILE"
echo ""

# Run with multiprocessing
python "$SCRIPT_DIR/run_prompt_test.py" \
    --dataset "$DATASET" \
    --generation "$GENERATION" \
    --workers "$WORKERS"

echo ""
echo "Results saved to: $SCRIPT_DIR/results.csv"
echo "Analyze with: python $SCRIPT_DIR/results_tracker.py --dataset $DATASET"
