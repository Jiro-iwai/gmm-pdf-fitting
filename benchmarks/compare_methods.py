#!/usr/bin/env python3
"""
Compare EM and LP methods across different modes and provide recommendations.

This script analyzes benchmark results to compare:
- EM method (PDF-only vs Moments mode)
- LP method (PDF vs Moments mode)
- Performance metrics: execution time, PDF error, moment errors
"""

import json
import numpy as np
import argparse
from typing import List, Dict
from collections import defaultdict


def load_results(json_path: str) -> List[Dict]:
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compare_methods(results: List[Dict]) -> Dict:
    """Compare EM and LP methods across different modes."""
    comparison = {
        'em_pdf': [],
        'em_moments': [],
        'lp_pdf': [],
        'lp_moments': [],
    }
    
    for r in results:
        method = r['method']
        mode = r.get('objective_mode', 'pdf')
        
        key = f"{method}_{mode}"
        if key in comparison:
            comparison[key].append(r)
    
    return comparison


def calculate_statistics(cases: List[Dict]) -> Dict:
    """Calculate statistics for a group of cases."""
    if not cases:
        return None
    
    return {
        'count': len(cases),
        'execution_time': {
            'mean': np.mean([c['execution_time'] for c in cases]),
            'median': np.median([c['execution_time'] for c in cases]),
            'min': np.min([c['execution_time'] for c in cases]),
            'max': np.max([c['execution_time'] for c in cases]),
        },
        'pdf_error_linf': {
            'mean': np.mean([c['pdf_error_linf'] for c in cases]),
            'median': np.median([c['pdf_error_linf'] for c in cases]),
        },
        'pdf_error_l2': {
            'mean': np.mean([c['pdf_error_l2'] for c in cases]),
        },
        'mean_error_rel': {
            'mean': np.mean([abs(c['mean_error_rel']) for c in cases]),
        },
        'std_error_rel': {
            'mean': np.mean([abs(c['std_error_rel']) for c in cases]),
        },
        'skewness_error_rel': {
            'mean': np.mean([abs(c['skewness_error_rel']) for c in cases]),
        },
        'kurtosis_error_rel': {
            'mean': np.mean([abs(c['kurtosis_error_rel']) for c in cases]),
            'median': np.median([abs(c['kurtosis_error_rel']) for c in cases]),
            'max': np.max([abs(c['kurtosis_error_rel']) for c in cases]),
        },
    }


def print_comparison_table(comparison: Dict):
    """Print comparison table."""
    print("=" * 100)
    print("Method Comparison: EM vs LP")
    print("=" * 100)
    
    methods = [
        ('EM (PDF-only)', 'em_pdf'),
        ('EM (Moments)', 'em_moments'),
        ('LP (PDF)', 'lp_pdf'),
        ('LP (Moments)', 'lp_moments'),
    ]
    
    stats = {}
    for name, key in methods:
        stats[name] = calculate_statistics(comparison[key])
    
    # Execution time comparison
    print("\n1. Execution Time (seconds)")
    print("-" * 100)
    print(f"{'Method':<20} {'Count':<8} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-" * 100)
    for name, key in methods:
        s = stats[name]
        if s:
            print(f"{name:<20} {s['count']:<8} {s['execution_time']['mean']:<12.6f} "
                  f"{s['execution_time']['median']:<12.6f} {s['execution_time']['min']:<12.6f} "
                  f"{s['execution_time']['max']:<12.6f}")
    
    # PDF error comparison
    print("\n2. PDF Error (L∞)")
    print("-" * 100)
    print(f"{'Method':<20} {'Mean':<15} {'Median':<15}")
    print("-" * 100)
    for name, key in methods:
        s = stats[name]
        if s:
            print(f"{name:<20} {s['pdf_error_linf']['mean']:<15.6f} {s['pdf_error_linf']['median']:<15.6f}")
    
    # Moment errors comparison
    print("\n3. Moment Errors (Relative %)")
    print("-" * 100)
    print(f"{'Method':<20} {'Mean':<12} {'Std':<12} {'Skewness':<12} {'Kurtosis':<12}")
    print("-" * 100)
    for name, key in methods:
        s = stats[name]
        if s:
            print(f"{name:<20} {s['mean_error_rel']['mean']:<12.4f} {s['std_error_rel']['mean']:<12.4f} "
                  f"{s['skewness_error_rel']['mean']:<12.4f} {s['kurtosis_error_rel']['mean']:<12.4f}")
    
    # Detailed kurtosis error
    print("\n4. Kurtosis Error Details")
    print("-" * 100)
    print(f"{'Method':<20} {'Mean':<12} {'Median':<12} {'Max':<12}")
    print("-" * 100)
    for name, key in methods:
        s = stats[name]
        if s:
            print(f"{name:<20} {s['kurtosis_error_rel']['mean']:<12.4f} "
                  f"{s['kurtosis_error_rel']['median']:<12.4f} {s['kurtosis_error_rel']['max']:<12.4f}")


def print_recommendations(comparison: Dict):
    """Print recommendations based on comparison."""
    stats = {}
    for key in ['em_pdf', 'em_moments', 'lp_pdf', 'lp_moments']:
        stats[key] = calculate_statistics(comparison[key])
    
    print("\n" + "=" * 100)
    print("Recommendations")
    print("=" * 100)
    
    # Find best method for each criterion
    print("\n1. Best Method by Criterion:")
    print("-" * 100)
    
    # Execution time
    times = {}
    for key, s in stats.items():
        if s:
            times[key] = s['execution_time']['mean']
    if times:
        fastest = min(times.items(), key=lambda x: x[1])
        print(f"  Fastest execution: {fastest[0].replace('_', ' ').upper()} ({fastest[1]:.6f}s)")
    
    # PDF error
    pdf_errors = {}
    for key, s in stats.items():
        if s:
            pdf_errors[key] = s['pdf_error_linf']['mean']
    if pdf_errors:
        best_pdf = min(pdf_errors.items(), key=lambda x: x[1])
        print(f"  Best PDF accuracy: {best_pdf[0].replace('_', ' ').upper()} ({best_pdf[1]:.6f})")
    
    # Kurtosis error
    kurt_errors = {}
    for key, s in stats.items():
        if s:
            kurt_errors[key] = s['kurtosis_error_rel']['mean']
    if kurt_errors:
        best_kurt = min(kurt_errors.items(), key=lambda x: x[1])
        print(f"  Best kurtosis accuracy: {best_kurt[0].replace('_', ' ').upper()} ({best_kurt[1]:.2f}%)")
    
    # Overall moment accuracy
    moment_errors = {}
    for key, s in stats.items():
        if s:
            moment_errors[key] = (
                s['mean_error_rel']['mean'] +
                s['std_error_rel']['mean'] +
                s['skewness_error_rel']['mean'] +
                s['kurtosis_error_rel']['mean']
            ) / 4
    if moment_errors:
        best_moments = min(moment_errors.items(), key=lambda x: x[1])
        print(f"  Best overall moment accuracy: {best_moments[0].replace('_', ' ').upper()} ({best_moments[1]:.2f}%)")
    
    # Use case recommendations
    print("\n2. Recommended Settings by Use Case:")
    print("-" * 100)
    
    print("\n  A. High Accuracy Required (Moment Matching Critical):")
    if stats['em_moments'] and stats['lp_moments']:
        em_kurt = stats['em_moments']['kurtosis_error_rel']['mean']
        lp_kurt = stats['lp_moments']['kurtosis_error_rel']['mean']
        if em_kurt < lp_kurt:
            print(f"    → EM (Moments mode) with K≥5")
            print(f"      Reason: Best kurtosis accuracy ({em_kurt:.2f}%)")
        else:
            print(f"    → LP (Moments mode)")
            print(f"      Reason: Best kurtosis accuracy ({lp_kurt:.2f}%)")
    
    print("\n  B. Fast Execution Required:")
    if times:
        fastest_key = min(times.items(), key=lambda x: x[1])[0]
        fastest_name = fastest_key.replace('_', ' ').upper()
        print(f"    → {fastest_name}")
        print(f"      Reason: Fastest execution ({times[fastest_key]:.6f}s)")
    
    print("\n  C. Best PDF Approximation:")
    if pdf_errors:
        best_pdf_key = min(pdf_errors.items(), key=lambda x: x[1])[0]
        best_pdf_name = best_pdf_key.replace('_', ' ').upper()
        print(f"    → {best_pdf_name}")
        print(f"      Reason: Lowest PDF error ({pdf_errors[best_pdf_key]:.6f})")
    
    print("\n  D. Balanced Performance (Good PDF + Good Moments):")
    # Calculate balanced score
    balanced_scores = {}
    for key, s in stats.items():
        if s:
            # Normalize and combine PDF error and moment errors
            pdf_score = s['pdf_error_linf']['mean']
            moment_score = (
                s['mean_error_rel']['mean'] +
                s['std_error_rel']['mean'] +
                s['skewness_error_rel']['mean'] +
                s['kurtosis_error_rel']['mean']
            ) / 4
            # Weighted combination (can be adjusted)
            balanced_scores[key] = pdf_score * 0.5 + moment_score * 0.5
    if balanced_scores:
        best_balanced = min(balanced_scores.items(), key=lambda x: x[1])
        best_balanced_name = best_balanced[0].replace('_', ' ').upper()
        print(f"    → {best_balanced_name}")
        print(f"      Reason: Best balance between PDF and moment accuracy")
    
    print("\n3. Method-Specific Recommendations:")
    print("-" * 100)
    
    print("\n  EM Method:")
    if stats['em_pdf'] and stats['em_moments']:
        print(f"    - PDF-only mode: Fast ({stats['em_pdf']['execution_time']['mean']:.6f}s), "
              f"but higher moment errors (kurtosis: {stats['em_pdf']['kurtosis_error_rel']['mean']:.2f}%)")
        print(f"    - Moments mode: Slightly slower ({stats['em_moments']['execution_time']['mean']:.6f}s), "
              f"but much better moment accuracy (kurtosis: {stats['em_moments']['kurtosis_error_rel']['mean']:.2f}%)")
        print(f"    - Recommendation: Use Moments mode with K≥5 for accurate moment matching")
    
    print("\n  LP Method:")
    if stats['lp_pdf'] and stats['lp_moments']:
        print(f"    - PDF mode: Fast ({stats['lp_pdf']['execution_time']['mean']:.6f}s), "
              f"focuses on PDF accuracy")
        print(f"    - Moments mode: Optimizes moment errors (kurtosis: {stats['lp_moments']['kurtosis_error_rel']['mean']:.2f}%)")
        print(f"    - Recommendation: Use Moments mode when moment accuracy is important")
    
    print("\n4. K Value Recommendations:")
    print("-" * 100)
    print("    - K=3: Not recommended (insufficient degrees of freedom)")
    print("    - K=4: Acceptable with Moments mode (kurtosis error ~1.8%)")
    print("    - K≥5: Recommended for accurate moment matching (kurtosis error <1%)")
    print("    - K≥10: Optimal for high accuracy requirements")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare EM and LP methods and provide recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results JSON file"
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Compare methods
    comparison = compare_methods(results)
    
    # Print comparison
    print_comparison_table(comparison)
    
    # Print recommendations
    print_recommendations(comparison)


if __name__ == "__main__":
    main()

