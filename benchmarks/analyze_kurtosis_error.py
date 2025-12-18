#!/usr/bin/env python3
"""
Analyze kurtosis error patterns in benchmark results.

This script analyzes benchmark results to identify factors that contribute
to large kurtosis errors in GMM fitting.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict
from collections import defaultdict


def load_benchmark_results(json_path: str) -> List[Dict]:
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_kurtosis_errors(results: List[Dict]) -> Dict:
    """Analyze kurtosis errors and identify patterns."""
    analysis = {
        'by_method': defaultdict(list),
        'by_objective_mode': defaultdict(list),
        'by_K': defaultdict(list),
        'by_z_npoints': defaultdict(list),
        'by_param_set': defaultdict(list),
        'large_error_cases': [],
        'true_kurtosis_distribution': [],
        'error_vs_true_kurtosis': [],
    }
    
    for r in results:
        kurt_err_abs = abs(r.get('kurtosis_error_rel', 0))
        true_kurt = r.get('mu_x', 0)  # We'll need to compute this
        
        # Group by method
        analysis['by_method'][r['method']].append(kurt_err_abs)
        
        # Group by objective mode
        obj_mode = r.get('objective_mode', 'pdf')
        analysis['by_objective_mode'][obj_mode].append(kurt_err_abs)
        
        # Group by K
        analysis['by_K'][r['K']].append(kurt_err_abs)
        
        # Group by grid resolution
        analysis['by_z_npoints'][r['z_npoints']].append(kurt_err_abs)
        
        # Group by parameter set (if available)
        if 'param_set_idx' in r:
            analysis['by_param_set'][r['param_set_idx']].append(kurt_err_abs)
        
        # Large error cases
        if kurt_err_abs > 10:
            analysis['large_error_cases'].append({
                'method': r['method'],
                'objective_mode': obj_mode,
                'K': r['K'],
                'z_npoints': r['z_npoints'],
                'kurtosis_error': kurt_err_abs,
                'pdf_error': r.get('pdf_error_linf', 0),
                'param_set_idx': r.get('param_set_idx', -1),
            })
    
    return analysis


def print_analysis(analysis: Dict, param_configs: List[Dict] = None):
    """Print analysis results."""
    print("=" * 80)
    print("Kurtosis Error Analysis")
    print("=" * 80)
    
    # Overall statistics
    all_errors = []
    for method_errors in analysis['by_method'].values():
        all_errors.extend(method_errors)
    
    if all_errors:
        print(f"\nOverall Statistics:")
        print(f"  Total cases: {len(all_errors)}")
        print(f"  Mean error: {np.mean(all_errors):.2f}%")
        print(f"  Median error: {np.median(all_errors):.2f}%")
        print(f"  Max error: {np.max(all_errors):.2f}%")
        print(f"  75th percentile: {np.percentile(all_errors, 75):.2f}%")
        print(f"  90th percentile: {np.percentile(all_errors, 90):.2f}%")
        print(f"  95th percentile: {np.percentile(all_errors, 95):.2f}%")
    
    # By method
    print(f"\nBy Method:")
    for method, errors in analysis['by_method'].items():
        if errors:
            print(f"  {method.upper()}:")
            print(f"    Mean: {np.mean(errors):.2f}%, Median: {np.median(errors):.2f}%, Max: {np.max(errors):.2f}%")
            print(f"    Cases > 10%: {sum(1 for e in errors if e > 10)}/{len(errors)}")
            print(f"    Cases > 50%: {sum(1 for e in errors if e > 50)}/{len(errors)}")
    
    # By objective mode
    print(f"\nBy Objective Mode:")
    for mode, errors in analysis['by_objective_mode'].items():
        if errors:
            print(f"  {mode}:")
            print(f"    Mean: {np.mean(errors):.2f}%, Median: {np.median(errors):.2f}%, Max: {np.max(errors):.2f}%")
            print(f"    Cases > 10%: {sum(1 for e in errors if e > 10)}/{len(errors)}")
    
    # By K
    print(f"\nBy K (Number of Components):")
    for K in sorted(analysis['by_K'].keys()):
        errors = analysis['by_K'][K]
        if errors:
            print(f"  K={K}:")
            print(f"    Mean: {np.mean(errors):.2f}%, Median: {np.median(errors):.2f}%, Max: {np.max(errors):.2f}%")
            print(f"    Cases > 10%: {sum(1 for e in errors if e > 10)}/{len(errors)}")
    
    # By grid resolution
    print(f"\nBy Grid Resolution:")
    for npoints in sorted(analysis['by_z_npoints'].keys()):
        errors = analysis['by_z_npoints'][npoints]
        if errors:
            print(f"  {npoints} points:")
            print(f"    Mean: {np.mean(errors):.2f}%, Median: {np.median(errors):.2f}%, Max: {np.max(errors):.2f}%")
    
    # By parameter set
    if analysis['by_param_set'] and param_configs:
        print(f"\nBy Parameter Set:")
        for param_idx in sorted(analysis['by_param_set'].keys()):
            errors = analysis['by_param_set'][param_idx]
            if errors and param_idx < len(param_configs):
                param = param_configs[param_idx]
                print(f"  Set {param_idx + 1} (rho={param['rho']:.2f}, mu_x={param['mu_x']:.2f}, mu_y={param['mu_y']:.2f}, "
                      f"sigma_x={param['sigma_x']:.2f}, sigma_y={param['sigma_y']:.2f}):")
                print(f"    Mean: {np.mean(errors):.2f}%, Median: {np.median(errors):.2f}%, Max: {np.max(errors):.2f}%")
                print(f"    Cases > 10%: {sum(1 for e in errors if e > 10)}/{len(errors)}")
    
    # Large error cases
    if analysis['large_error_cases']:
        print(f"\nLarge Error Cases (>10%):")
        print(f"  Total: {len(analysis['large_error_cases'])}")
        
        # Group by method and mode
        by_method_mode = defaultdict(int)
        by_K = defaultdict(int)
        by_param_set = defaultdict(int)
        
        for case in analysis['large_error_cases']:
            key = f"{case['method']}_{case['objective_mode']}"
            by_method_mode[key] += 1
            by_K[case['K']] += 1
            if 'param_set_idx' in case:
                by_param_set[case['param_set_idx']] += 1
        
        print(f"\n  By Method/Mode:")
        for key, count in sorted(by_method_mode.items()):
            print(f"    {key}: {count}")
        
        print(f"\n  By K:")
        for K, count in sorted(by_K.items()):
            print(f"    K={K}: {count}")
        
        if by_param_set:
            print(f"\n  By Parameter Set:")
            for param_idx, count in sorted(by_param_set.items()):
                print(f"    Set {param_idx + 1}: {count}")
        
        # Show top 10 worst cases
        sorted_cases = sorted(analysis['large_error_cases'], key=lambda x: x['kurtosis_error'], reverse=True)
        print(f"\n  Top 10 Worst Cases:")
        for i, case in enumerate(sorted_cases[:10], 1):
            print(f"    {i}. {case['method']} ({case['objective_mode']}), K={case['K']}, "
                  f"z_npoints={case['z_npoints']}, error={case['kurtosis_error']:.2f}%, "
                  f"PDF_err={case['pdf_error']:.6f}")


def plot_analysis(analysis: Dict, output_path: str = "kurtosis_error_analysis.png"):
    """Create visualization plots for kurtosis error analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Error distribution
    ax = axes[0, 0]
    all_errors = []
    for errors in analysis['by_method'].values():
        all_errors.extend(errors)
    if all_errors:
        ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Kurtosis Error (rel %)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Kurtosis Errors')
        ax.set_yscale('log')
        ax.axvline(np.median(all_errors), color='r', linestyle='--', label=f'Median: {np.median(all_errors):.2f}%')
        ax.legend()
    
    # Plot 2: Error by method
    ax = axes[0, 1]
    methods = []
    means = []
    medians = []
    for method, errors in analysis['by_method'].items():
        if errors:
            methods.append(method.upper())
            means.append(np.mean(errors))
            medians.append(np.median(errors))
    if methods:
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, means, width, label='Mean', alpha=0.7)
        ax.bar(x + width/2, medians, width, label='Median', alpha=0.7)
        ax.set_xlabel('Method')
        ax.set_ylabel('Kurtosis Error (rel %)')
        ax.set_title('Error by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_yscale('log')
    
    # Plot 3: Error by K
    ax = axes[0, 2]
    K_values = []
    means = []
    for K in sorted(analysis['by_K'].keys()):
        errors = analysis['by_K'][K]
        if errors:
            K_values.append(K)
            means.append(np.mean(errors))
    if K_values:
        ax.bar(K_values, means, alpha=0.7)
        ax.set_xlabel('K (Number of Components)')
        ax.set_ylabel('Mean Kurtosis Error (rel %)')
        ax.set_title('Error vs K')
        ax.set_yscale('log')
    
    # Plot 4: Error by grid resolution
    ax = axes[1, 0]
    npoints_values = []
    means = []
    for npoints in sorted(analysis['by_z_npoints'].keys()):
        errors = analysis['by_z_npoints'][npoints]
        if errors:
            npoints_values.append(npoints)
            means.append(np.mean(errors))
    if npoints_values:
        ax.bar(npoints_values, means, alpha=0.7)
        ax.set_xlabel('Grid Resolution (points)')
        ax.set_ylabel('Mean Kurtosis Error (rel %)')
        ax.set_title('Error vs Grid Resolution')
        ax.set_yscale('log')
    
    # Plot 5: Error by objective mode
    ax = axes[1, 1]
    modes = []
    means = []
    for mode, errors in analysis['by_objective_mode'].items():
        if errors:
            modes.append(mode)
            means.append(np.mean(errors))
    if modes:
        ax.bar(modes, means, alpha=0.7)
        ax.set_xlabel('Objective Mode')
        ax.set_ylabel('Mean Kurtosis Error (rel %)')
        ax.set_title('Error by Objective Mode')
        ax.set_yscale('log')
    
    # Plot 6: Large error cases breakdown
    ax = axes[1, 2]
    if analysis['large_error_cases']:
        by_method_mode = defaultdict(int)
        for case in analysis['large_error_cases']:
            key = f"{case['method']}\n({case['objective_mode']})"
            by_method_mode[key] += 1
        
        if by_method_mode:
            labels = list(by_method_mode.keys())
            counts = list(by_method_mode.values())
            ax.bar(labels, counts, alpha=0.7)
            ax.set_xlabel('Method/Mode')
            ax.set_ylabel('Number of Cases > 10%')
            ax.set_title('Large Error Cases by Method/Mode')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze kurtosis errors in benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--param-configs",
        type=str,
        default=None,
        help="Path to JSON file containing parameter configurations (optional)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kurtosis_error_analysis.png",
        help="Output path for plots"
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_benchmark_results(args.input)
    
    # Load parameter configs if provided
    param_configs = None
    if args.param_configs:
        with open(args.param_configs, 'r') as f:
            param_configs = json.load(f)
    
    # Analyze
    analysis = analyze_kurtosis_errors(results)
    
    # Print analysis
    print_analysis(analysis, param_configs)
    
    # Generate plots if requested
    if args.plot:
        plot_analysis(analysis, args.output)


if __name__ == "__main__":
    main()

