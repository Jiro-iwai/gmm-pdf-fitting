#!/bin/bash
# Quick Performance Benchmark Script
# Reduced parameter sets for faster execution

set -e

echo "=========================================="
echo "Quick Performance Benchmark"
echo "=========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Create output directory
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

# Create quick benchmark config (reduced parameters)
python -c "
import json

# EM method quick config
config_em = {
    'mu_x': 0.1,
    'sigma_x': 0.4,
    'mu_y': 0.15,
    'sigma_y': 0.9,
    'rho': 0.9,
    'z_range': [-2.0, 4.0],
    'z_npoints': 128,
    'K': 10,
    'L': 10,
    'method': 'em',
    'max_iter': 100,
    'tol': 1e-6,
    'reg_var': 1e-6,
    'n_init': 1,
    'seed': 0,
    'init': 'quantile',
    'use_moment_matching': False,
    'qp_mode': 'hard',
    'soft_lambda': 1e4,
    'output_path': 'pdf_comparison',
    'show_grid_points': False,
    'max_grid_points_display': 200
}
json.dump(config_em, open('$OUTPUT_DIR/config_em_quick.json', 'w'), indent=2)

# LP method quick config
config_lp = {
    'mu_x': 0.1,
    'sigma_x': 0.4,
    'mu_y': 0.15,
    'sigma_y': 0.9,
    'rho': 0.9,
    'z_range': [-2.0, 4.0],
    'z_npoints': 128,
    'K': 10,
    'L': 10,
    'method': 'lp',
    'lp_params': {
        'solver': 'highs',
        'sigma_min_scale': 0.1,
        'sigma_max_scale': 3.0,
        'pdf_tolerance': 0.01,
        'lambda_pdf': 1.0,
        'lambda_raw': [1.0, 1.0, 1.0, 1.0],
        'objective_form': 'A'
    },
    'output_path': 'pdf_comparison',
    'show_grid_points': False,
    'max_grid_points_display': 200
}
json.dump(config_lp, open('$OUTPUT_DIR/config_lp_quick.json', 'w'), indent=2)

# Hybrid method quick config
config_hybrid = {
    'mu_x': 0.1,
    'sigma_x': 0.4,
    'mu_y': 0.15,
    'sigma_y': 0.9,
    'rho': 0.9,
    'z_range': [-2.0, 4.0],
    'z_npoints': 128,
    'K': 10,
    'L': 10,
    'method': 'hybrid',
    'lp_params': {
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
    },
    'max_iter': 50,
    'tol': 1e-6,
    'reg_var': 1e-6,
    'n_init': 1,
    'seed': 0,
    'init': 'custom',
    'use_moment_matching': False,
    'qp_mode': 'hard',
    'soft_lambda': 1e4,
    'output_path': 'pdf_comparison',
    'show_grid_points': False,
    'max_grid_points_display': 200
}
json.dump(config_hybrid, open('$OUTPUT_DIR/config_hybrid_quick.json', 'w'), indent=2)

print('Quick config files created')
"

echo "1. Running EM method benchmark (quick)..."
python benchmark.py \
    --config "$OUTPUT_DIR/config_em_quick.json" \
    --output "$OUTPUT_DIR/benchmark_em_quick.json" \
    2>&1 | tail -50

echo ""
echo "2. Running LP method benchmark (quick)..."
python benchmark.py \
    --config "$OUTPUT_DIR/config_lp_quick.json" \
    --output "$OUTPUT_DIR/benchmark_lp_quick.json" \
    2>&1 | tail -50

echo ""
echo "3. Running Hybrid method benchmark (quick)..."
python benchmark.py \
    --config "$OUTPUT_DIR/config_hybrid_quick.json" \
    --output "$OUTPUT_DIR/benchmark_hybrid_quick.json" \
    2>&1 | tail -50

echo ""
echo "=========================================="
echo "Quick benchmark completed!"
echo "Results saved in: $OUTPUT_DIR/"
echo "=========================================="

