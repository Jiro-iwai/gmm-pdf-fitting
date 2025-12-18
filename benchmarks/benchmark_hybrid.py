#!/usr/bin/env python3
"""
Benchmark for Hybrid method (LP → EM → QP).

This script evaluates the Hybrid method performance across various configurations.
"""

import argparse
import json
import sys
import os
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict

# Add src directory to path to import gmm_fitting package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from gmm_fitting import (
    load_config,
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
    gmm1d_pdf,
    compute_pdf_statistics,
    GMM1DParams,
    solve_lp_pdf_rawmoments_linf,
    build_gaussian_dictionary,
    compute_basis_matrices,
    compute_gmm_moments_from_weights,
    compute_errors,
    EPSILON,
)


@dataclass
class HybridBenchmarkResult:
    """Results from a single Hybrid benchmark run."""
    K: int
    z_npoints: int
    dict_J: int
    dict_L: int
    tail_focus: str
    tail_alpha: float
    execution_time: float = 0.0
    lp_runtime: float = 0.0
    em_runtime: float = 0.0
    qp_runtime: float = 0.0
    pdf_error_linf: float = 0.0
    cdf_error_linf: float = 0.0
    tail_l1_error: float = 0.0
    mean_error: float = 0.0
    std_error: float = 0.0
    variance_error: float = 0.0
    skewness_error: float = 0.0
    kurtosis_error: float = 0.0
    mean_error_rel: float = 0.0
    std_error_rel: float = 0.0
    variance_error_rel: float = 0.0
    skewness_error_rel: float = 0.0
    kurtosis_error_rel: float = 0.0
    use_moment_matching: bool = False
    n_iterations: int = 0
    converged: bool = False


def benchmark_hybrid_method(
    z: np.ndarray,
    f_true: np.ndarray,
    K: int,
    dict_J: int,
    dict_L: int,
    config: Dict,
    tail_focus: str = "none",
    tail_alpha: float = 1.0,
    use_moment_matching: bool = False,
    n_trials: int = 1
) -> HybridBenchmarkResult:
    """
    Benchmark Hybrid method.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values
    K : int
        Number of components
    dict_J : int
        Number of mean locations in dictionary
    dict_L : int
        Number of sigma levels per mean location
    config : dict
        Configuration dictionary
    tail_focus : str
        Tail focus mode: "none", "right", "left", "both"
    tail_alpha : float
        Tail emphasis strength
    use_moment_matching : bool
        Whether to use moment matching QP
    n_trials : int
        Number of trials (for averaging)
    
    Returns:
    --------
    HybridBenchmarkResult
        Benchmark results
    """
    times = []
    lp_times = []
    em_times = []
    qp_times = []
    pdf_errors_linf = []
    cdf_errors_linf = []
    tail_l1_errors = []
    moment_errors = []
    n_iterations_list = []
    
    lp_params = config.get("lp_params", {})
    reg_var = config.get("reg_var", 1e-6)
    max_iter = config.get("max_iter", 100)
    tol = config.get("tol", 1e-6)
    n_init = config.get("n_init", 1)
    seed = config.get("seed", 0)
    qp_mode = config.get("qp_mode", "hard")
    soft_lambda = config.get("soft_lambda", 1e4)
    
    for trial in range(n_trials):
        total_start = time.time()
        
        # Step 1: Build dictionary and solve LP
        lp_start = time.time()
        dictionary = build_gaussian_dictionary(
            z, f_true,
            J=dict_J,
            L=dict_L,
            mu_mode=lp_params.get("mu_mode", "quantile"),
            sigma_min_scale=lp_params.get("sigma_min_scale", 0.1),
            sigma_max_scale=lp_params.get("sigma_max_scale", 3.0),
            tail_focus=tail_focus,
            tail_alpha=tail_alpha,
            quantile_levels=lp_params.get("quantile_levels", None),
        )
        basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
        
        lp_result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=dictionary["mus"],
            sigmas=dictionary["sigmas"],
            z=z,
            f=f_true,
            pdf_tolerance=lp_params.get("pdf_tolerance", None),
            lambda_pdf=lp_params.get("lambda_pdf", 1.0),
            lambda_raw=tuple(lp_params.get("lambda_raw", [1.0, 1.0, 1.0, 1.0])),
            solver=lp_params.get("solver", "highs"),
            objective_form=lp_params.get("objective_form", "A"),
        )
        lp_time = time.time() - lp_start
        
        # Step 2: Select top K components
        w_all = lp_result["w"]
        idx_top_k = np.argsort(w_all)[::-1][:K]
        pi_init = w_all[idx_top_k]
        mu_init = dictionary["mus"][idx_top_k]
        var_init = dictionary["sigmas"][idx_top_k]**2
        
        pi_init = pi_init / np.sum(pi_init)
        var_init = np.maximum(var_init, reg_var)
        
        # Step 3: Run EM
        em_start = time.time()
        init_params_custom = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f_true,
            K=K,
            max_iter=max_iter,
            tol=tol,
            reg_var=reg_var,
            n_init=n_init,
            seed=seed,
            init="custom",
            init_params=init_params_custom,
            use_moment_matching=use_moment_matching,
            qp_mode=qp_mode,
            soft_lambda=soft_lambda,
        )
        em_time = time.time() - em_start
        
        # Extract QP time if moment matching was used
        qp_time = 0.0
        if use_moment_matching and hasattr(params, '_qp_info'):
            qp_time = params._qp_info.get('qp_time', 0.0)
        
        total_time = time.time() - total_start
        
        times.append(total_time)
        lp_times.append(lp_time)
        em_times.append(em_time)
        qp_times.append(qp_time)
        n_iterations_list.append(n_iter)
        
        # Evaluate GMM PDF
        f_hat = gmm1d_pdf(z, params)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        # Compute errors
        errors = compute_errors(z, f_true, f_hat)
        pdf_errors_linf.append(errors["linf_pdf"])
        cdf_errors_linf.append(errors["linf_cdf"])
        tail_l1_errors.append(errors["tail_l1_error"])
        
        # Compute moment errors
        stats_true = compute_pdf_statistics(z, f_true)
        mean_hat, var_hat, skew_hat, kurt_hat = compute_gmm_moments_from_weights(
            params.pi, params.mu, np.sqrt(params.var)
        )
        stats_hat = {
            'mean': mean_hat,
            'std': np.sqrt(var_hat),
            'skewness': skew_hat,
            'kurtosis': kurt_hat
        }
        
        moment_errs = {
            'mean': stats_hat['mean'] - stats_true['mean'],
            'std': stats_hat['std'] - stats_true['std'],
            'variance': stats_hat['std']**2 - stats_true['std']**2,
            'skewness': stats_hat['skewness'] - stats_true['skewness'],
            'kurtosis': stats_hat['kurtosis'] - stats_true['kurtosis'],
        }
        moment_errors.append(moment_errs)
    
    # Average results
    result = HybridBenchmarkResult(
        K=K,
        z_npoints=len(z),
        dict_J=dict_J,
        dict_L=dict_L,
        tail_focus=tail_focus,
        tail_alpha=tail_alpha,
        execution_time=np.mean(times),
        lp_runtime=np.mean(lp_times),
        em_runtime=np.mean(em_times),
        qp_runtime=np.mean(qp_times),
        pdf_error_linf=np.mean(pdf_errors_linf),
        cdf_error_linf=np.mean(cdf_errors_linf),
        tail_l1_error=np.mean(tail_l1_errors),
        mean_error=np.mean([e['mean'] for e in moment_errors]),
        std_error=np.mean([e['std'] for e in moment_errors]),
        variance_error=np.mean([e['variance'] for e in moment_errors]),
        skewness_error=np.mean([e['skewness'] for e in moment_errors]),
        kurtosis_error=np.mean([e['kurtosis'] for e in moment_errors]),
        use_moment_matching=use_moment_matching,
        n_iterations=int(np.mean(n_iterations_list)),
        converged=True,
    )
    
    # Compute relative errors
    KURTOSIS_THRESHOLD = 0.01
    SKEWNESS_THRESHOLD = 0.01
    
    stats_true = compute_pdf_statistics(z, f_true)
    abs_mean = abs(stats_true['mean']) if abs(stats_true['mean']) > EPSILON else 1.0
    abs_std = abs(stats_true['std']) if abs(stats_true['std']) > EPSILON else 1.0
    abs_var = abs(stats_true['std']**2) if abs(stats_true['std']**2) > EPSILON else 1.0
    
    abs_skew_true = abs(stats_true['skewness'])
    abs_kurt_true = abs(stats_true['kurtosis'])
    
    result.mean_error_rel = (result.mean_error / abs_mean) * 100.0
    result.std_error_rel = (result.std_error / abs_std) * 100.0
    result.variance_error_rel = (result.variance_error / abs_var) * 100.0
    
    if abs_skew_true > SKEWNESS_THRESHOLD:
        result.skewness_error_rel = (result.skewness_error / abs_skew_true) * 100.0
    else:
        result.skewness_error_rel = (result.skewness_error / SKEWNESS_THRESHOLD) * 100.0
    
    if abs_kurt_true > KURTOSIS_THRESHOLD:
        result.kurtosis_error_rel = (result.kurtosis_error / abs_kurt_true) * 100.0
    else:
        result.kurtosis_error_rel = (result.kurtosis_error / KURTOSIS_THRESHOLD) * 100.0
    
    return result


def run_hybrid_benchmark(
    config_path: str,
    output_path: str = "benchmark_hybrid_results.json",
):
    """
    Run comprehensive benchmark for Hybrid method.
    
    Parameters:
    -----------
    config_path : str
        Path to base configuration file
    output_path : str
        Path to save benchmark results (JSON)
    """
    print("=" * 80)
    print("Hybrid Method Performance Benchmark")
    print("=" * 80)
    
    # Load base configuration
    base_config = load_config(config_path)
    
    # Extract parameters
    mu_x = base_config["mu_x"]
    sigma_x = base_config["sigma_x"]
    mu_y = base_config["mu_y"]
    sigma_y = base_config["sigma_y"]
    rho = base_config["rho"]
    z_range = base_config["z_range"]
    
    # Test configurations
    K_values = [3, 4, 5, 10, 15, 20]  # Unified across all methods
    z_npoints_values = [8, 16, 32, 64, 128, 256, 512]  # Extended grid resolutions
    dict_J_values = [20, 30, 40]  # 4*K, 6*K, 8*K for K=5
    dict_L_values = [5, 10]
    tail_focus_values = ["none", "right", "left", "both"]
    tail_alpha_values = [1.0, 2.0]
    use_moment_matching_values = [False, True]
    
    all_results: List[HybridBenchmarkResult] = []
    
    # Generate true PDF
    z_fine = np.linspace(z_range[0], z_range[1], 1000)
    f_true_fine = max_pdf_bivariate_normal(
        z_fine, mu_x, sigma_x**2, mu_y, sigma_y**2, rho
    )
    f_true_fine = normalize_pdf_on_grid(z_fine, f_true_fine)
    
    total_runs = len(K_values) * len(z_npoints_values) * len(dict_J_values) * len(dict_L_values) * len(tail_focus_values) * len(tail_alpha_values) * len(use_moment_matching_values)
    current_run = 0
    
    print(f"\nRunning benchmarks...")
    print(f"  K values: {K_values}")
    print(f"  Grid resolutions: {z_npoints_values}")
    print(f"  Dictionary J values: {dict_J_values}")
    print(f"  Dictionary L values: {dict_L_values}")
    print(f"  Tail focus modes: {tail_focus_values}")
    print(f"  Tail alpha values: {tail_alpha_values}")
    print(f"  Moment matching: {use_moment_matching_values}")
    print(f"  Total runs: {total_runs}")
    
    for z_npoints in z_npoints_values:
        print(f"\n  Grid resolution: {z_npoints} points")
        
        z = np.linspace(z_range[0], z_range[1], z_npoints)
        f_true = max_pdf_bivariate_normal(
            z, mu_x, sigma_x**2, mu_y, sigma_y**2, rho
        )
        f_true = normalize_pdf_on_grid(z, f_true)
        
        for K in K_values:
            for dict_J in dict_J_values:
                for dict_L in dict_L_values:
                    for tail_focus in tail_focus_values:
                        for tail_alpha in tail_alpha_values:
                            for use_mm in use_moment_matching_values:
                                current_run += 1
                                print(f"    [{current_run}/{total_runs}] K={K}, J={dict_J}, L={dict_L}, "
                                      f"tail={tail_focus}, alpha={tail_alpha}, mm={use_mm}...", 
                                      end=" ", flush=True)
                                
                                try:
                                    result = benchmark_hybrid_method(
                                        z, f_true, K, dict_J, dict_L, base_config,
                                        tail_focus=tail_focus,
                                        tail_alpha=tail_alpha,
                                        use_moment_matching=use_mm,
                                        n_trials=1
                                    )
                                    all_results.append(result)
                                    print(f"✓ ({result.execution_time:.4f}s)", end=" ")
                                except Exception as e:
                                    print(f"✗ ({str(e)[:30]})", end=" ")
                                print()
    
    # Save results
    results_dict = [asdict(r) for r in all_results]
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Hybrid benchmark completed. Results saved to: {output_path}")
    print(f"Total runs: {len(all_results)}")
    print(f"{'=' * 80}")
    
    # Print summary
    print("\nSummary Statistics:")
    print(f"  Average execution time: {np.mean([r.execution_time for r in all_results]):.4f}s")
    print(f"  Average LP runtime: {np.mean([r.lp_runtime for r in all_results]):.4f}s")
    print(f"  Average EM runtime: {np.mean([r.em_runtime for r in all_results]):.4f}s")
    print(f"  Average QP runtime: {np.mean([r.qp_runtime for r in all_results]):.4f}s")
    print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in all_results]):.6e}")
    print(f"  Average CDF L∞ error: {np.mean([r.cdf_error_linf for r in all_results]):.6e}")
    print(f"  Average tail L1 error: {np.mean([r.tail_l1_error for r in all_results]):.6e}")
    print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in all_results]):.2f}%")
    
    return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Performance benchmark for Hybrid method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_hybrid_results.json",
        help="Path to save benchmark results (JSON)"
    )
    
    args = parser.parse_args()
    
    run_hybrid_benchmark(args.config, args.output)


if __name__ == "__main__":
    main()

