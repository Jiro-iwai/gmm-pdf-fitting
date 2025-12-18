#!/bin/bash
# Comprehensive Performance Benchmark Script
# This script runs benchmarks for EM, LP, and Hybrid methods

set -e

echo "=========================================="
echo "Comprehensive Performance Benchmark"
echo "=========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Base configuration file (adjust as needed)
BASE_CONFIG="../configs/config_em.json"

# Create output directory
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

echo "1. Running EM method benchmark..."
python benchmark.py \
    --config "$BASE_CONFIG" \
    --output "$OUTPUT_DIR/benchmark_em.json" \
    --vary-params \
    --plot

echo ""
echo "2. Running LP method benchmark..."
# Update config to use LP method
python -c "
import json
config = json.load(open('$BASE_CONFIG'))
config['method'] = 'lp'
config['L'] = 10
json.dump(config, open('$OUTPUT_DIR/config_lp_temp.json', 'w'), indent=2)
"

python benchmark.py \
    --config "$OUTPUT_DIR/config_lp_temp.json" \
    --output "$OUTPUT_DIR/benchmark_lp.json" \
    --vary-params \
    --plot

echo ""
echo "3. Running Hybrid method benchmark..."
# Update config to use Hybrid method
python -c "
import json
config = json.load(open('$BASE_CONFIG'))
config['method'] = 'hybrid'
config['L'] = 10
config['lp_params'] = {
    'dict_J': 20,
    'dict_L': 10,
    'mu_mode': 'quantile',
    'tail_focus': 'right',
    'tail_alpha': 2.0,
    'solver': 'highs',
    'objective_mode': 'raw_moments',
    'pdf_tolerance': 0.01,
    'lambda_pdf': 1.0,
    'lambda_raw': [1.0, 1.0, 1.0, 1.0],
    'objective_form': 'A'
}
json.dump(config, open('$OUTPUT_DIR/config_hybrid_temp.json', 'w'), indent=2)
"

python benchmark.py \
    --config "$OUTPUT_DIR/config_hybrid_temp.json" \
    --output "$OUTPUT_DIR/benchmark_hybrid.json" \
    --vary-params \
    --plot

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "Results saved in: $OUTPUT_DIR/"
echo "=========================================="

