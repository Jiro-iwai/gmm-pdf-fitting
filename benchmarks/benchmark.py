#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for GMM Fitting Methods

This script evaluates the performance of both EM and LP methods for GMM fitting
across various parameter configurations, measuring:
- Execution time
- PDF approximation accuracy (L∞ error)
- Moment matching accuracy (mean, variance, skewness, kurtosis)
- Convergence behavior
- Scalability with respect to K (number of components) and grid resolution
"""

import argparse
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from gmm_fitting import (
    load_config,
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
    gmm1d_pdf,
    compute_pdf_statistics,
    GMM1DParams,
    fit_gmm_lp_simple,
    solve_lp_pdf_rawmoments_linf,
    build_gaussian_dictionary,
    compute_basis_matrices,
    compute_gmm_moments_from_weights,
    compute_errors,
    compute_pdf_raw_moments,
    EPSILON,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    K: int
    z_npoints: int
    objective_mode: str = "pdf"
    execution_time: float = 0.0
    pdf_error_linf: float = 0.0
    pdf_error_l2: float = 0.0
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
    n_iterations: int = 0
    converged: bool = False
    n_nonzero_components: int = 0
    lp_objective: float = 0.0
    log_likelihood: float = 0.0
    # Additional fields for new methods
    tail_focus: str = "none"
    dict_J: int = 0
    dict_L: int = 0
    lp_runtime: float = 0.0
    em_runtime: float = 0.0
    qp_runtime: float = 0.0
    # Parameter set information (added when varying base params)
    param_set_idx: int = 0
    mu_x: float = 0.0
    sigma_x: float = 0.0
    mu_y: float = 0.0
    sigma_y: float = 0.0
    rho: float = 0.0


def compute_pdf_errors(z: np.ndarray, f_true: np.ndarray, f_hat: np.ndarray) -> Tuple[float, float]:
    """
    Compute L∞ and L2 errors between true and approximated PDFs.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values
    f_hat : np.ndarray
        Approximated PDF values
    
    Returns:
    --------
    linf_error : float
        L∞ error: max|f_true - f_hat|
    l2_error : float
        L2 error: sqrt(∫(f_true - f_hat)² dz)
    """
    f_true_norm = normalize_pdf_on_grid(z, f_true)
    f_hat_norm = normalize_pdf_on_grid(z, f_hat)
    
    # L∞ error
    linf_error = np.max(np.abs(f_true_norm - f_hat_norm))
    
    # L2 error (using trapezoidal rule)
    diff_sq = (f_true_norm - f_hat_norm)**2
    l2_error = np.sqrt(np.trapezoid(diff_sq, z))
    
    return linf_error, l2_error


def benchmark_em_method(
    z: np.ndarray,
    f_true: np.ndarray,
    K: int,
    config: Dict,
    n_trials: int = 1
) -> BenchmarkResult:
    """
    Benchmark EM method.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values
    K : int
        Number of components
    config : dict
        Configuration dictionary
    n_trials : int
        Number of trials (for averaging)
    
    Returns:
    --------
    BenchmarkResult
        Benchmark results
    """
    import warnings
    
    times = []
    pdf_errors_linf = []
    pdf_errors_l2 = []
    moment_errors = []
    n_iterations_list = []
    converged_list = []
    log_likelihoods = []
    
    # Suppress warnings during benchmark (will be summarized at the end)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for trial in range(n_trials):
            start_time = time.time()
            
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true,
                K=K,
                max_iter=config.get("max_iter", 400),
                tol=config.get("tol", 1e-10),
                reg_var=config.get("reg_var", 1e-6),
                n_init=config.get("n_init", 8),
                seed=config.get("seed", 1) + trial,
                init=config.get("init", "quantile"),
                use_moment_matching=config.get("use_moment_matching", False),
                qp_mode=config.get("qp_mode", "hard"),
                soft_lambda=config.get("soft_lambda", 1e4),
            )
        
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            # Evaluate GMM PDF
            f_hat = gmm1d_pdf(z, params)
            f_hat = normalize_pdf_on_grid(z, f_hat)
            
            # Compute PDF errors
            linf_err, l2_err = compute_pdf_errors(z, f_true, f_hat)
            pdf_errors_linf.append(linf_err)
            pdf_errors_l2.append(l2_err)
            
            # Compute moment errors
            stats_true = compute_pdf_statistics(z, f_true)
            stats_hat = compute_pdf_statistics(z, f_hat)
            
            moment_errs = {
                'mean': stats_hat['mean'] - stats_true['mean'],
                'std': stats_hat['std'] - stats_true['std'],
                'variance': stats_hat['std']**2 - stats_true['std']**2,
                'skewness': stats_hat['skewness'] - stats_true['skewness'],
                'kurtosis': stats_hat['kurtosis'] - stats_true['kurtosis'],
            }
            moment_errors.append(moment_errs)
            
            n_iterations_list.append(n_iter)
            converged_list.append(n_iter < config.get("max_iter", 400))
            log_likelihoods.append(ll)
        
        # Log warnings for summary (only for moment matching mode with K<5)
        if config.get("use_moment_matching", False) and K < 5 and w:
            # Warnings are already logged, no need to print here
            pass
    
    # Average results
    result = BenchmarkResult(
        method="em",
        K=K,
        z_npoints=len(z),
        execution_time=np.mean(times),
        pdf_error_linf=np.mean(pdf_errors_linf),
        pdf_error_l2=np.mean(pdf_errors_l2),
        mean_error=np.mean([e['mean'] for e in moment_errors]),
        std_error=np.mean([e['std'] for e in moment_errors]),
        variance_error=np.mean([e['variance'] for e in moment_errors]),
        skewness_error=np.mean([e['skewness'] for e in moment_errors]),
        kurtosis_error=np.mean([e['kurtosis'] for e in moment_errors]),
        n_iterations=int(np.mean(n_iterations_list)),
        converged=all(converged_list),
        n_nonzero_components=K,
        log_likelihood=np.mean(log_likelihoods),
    )
    
    # Compute relative errors
    # For kurtosis and skewness, use a larger threshold to avoid unstable relative errors
    # when true values are very close to zero
    KURTOSIS_THRESHOLD = 0.01  # If |true_kurtosis| < 0.01, use threshold as denominator
    SKEWNESS_THRESHOLD = 0.01  # If |true_skewness| < 0.01, use threshold as denominator
    
    stats_true = compute_pdf_statistics(z, f_true)
    abs_mean = abs(stats_true['mean']) if abs(stats_true['mean']) > EPSILON else 1.0
    abs_std = abs(stats_true['std']) if abs(stats_true['std']) > EPSILON else 1.0
    abs_var = abs(stats_true['std']**2) if abs(stats_true['std']**2) > EPSILON else 1.0
    
    # For skewness and kurtosis, use threshold when true value is very small
    abs_skew_true = abs(stats_true['skewness'])
    abs_kurt_true = abs(stats_true['kurtosis'])
    abs_skew = abs_skew_true if abs_skew_true > SKEWNESS_THRESHOLD else SKEWNESS_THRESHOLD
    abs_kurt = abs_kurt_true if abs_kurt_true > KURTOSIS_THRESHOLD else KURTOSIS_THRESHOLD
    
    result.mean_error_rel = (result.mean_error / abs_mean) * 100.0
    result.std_error_rel = (result.std_error / abs_std) * 100.0
    result.variance_error_rel = (result.variance_error / abs_var) * 100.0
    
    # For skewness and kurtosis, cap relative error calculation when true value is very small
    if abs_skew_true > SKEWNESS_THRESHOLD:
        result.skewness_error_rel = (result.skewness_error / abs_skew_true) * 100.0
    else:
        # Use absolute error scaled by threshold when true value is too small
        result.skewness_error_rel = (result.skewness_error / SKEWNESS_THRESHOLD) * 100.0
    
    if abs_kurt_true > KURTOSIS_THRESHOLD:
        result.kurtosis_error_rel = (result.kurtosis_error / abs_kurt_true) * 100.0
    else:
        # Use absolute error scaled by threshold when true value is too small
        result.kurtosis_error_rel = (result.kurtosis_error / KURTOSIS_THRESHOLD) * 100.0
    
    return result


def benchmark_lp_method(
    z: np.ndarray,
    f_true: np.ndarray,
    K: int,
    L: int,
    config: Dict,
    objective_mode: str = "pdf",
    tail_focus: str = "none",
    tail_alpha: float = 1.0,
    n_trials: int = 1
) -> BenchmarkResult:
    """
    Benchmark LP method.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values
    K : int
        Number of segments
    L : int
        Number of sigma levels per segment
    config : dict
        Configuration dictionary
    objective_mode : str
        Objective mode: "pdf" or "moments"
    n_trials : int
        Number of trials (for averaging)
    
    Returns:
    --------
    BenchmarkResult
        Benchmark results
    """
    times = []
    pdf_errors_linf = []
    pdf_errors_l2 = []
    moment_errors = []
    n_nonzero_list = []
    lp_objectives = []
    
    lp_params = config.get("lp_params", {})
    
    # Update lp_params with tail_focus if provided
    lp_params_updated = lp_params.copy()
    if tail_focus != "none":
        lp_params_updated["tail_focus"] = tail_focus
        lp_params_updated["tail_alpha"] = tail_alpha
    
    for trial in range(n_trials):
        start_time = time.time()
        
        lp_result, lp_timing = fit_gmm_lp_simple(
            z, f_true,
            K=K,
            L=L,
            lp_params=lp_params_updated,
            objective_mode=objective_mode
        )
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        # Extract results
        weights = lp_result["weights"]
        mus_all = lp_result["mus"]
        sigmas_all = lp_result["sigmas"]
        
        # For moments/raw_moments modes, use all components; for pdf mode, extract non-zero
        if objective_mode in ["moments", "raw_moments"]:
            weights_nonzero = weights
            mus_nonzero = mus_all
            sigmas_nonzero = sigmas_all
        else:
            nonzero_mask = weights > 1e-10
            if np.any(nonzero_mask):
                weights_nonzero = weights[nonzero_mask]
                mus_nonzero = mus_all[nonzero_mask]
                sigmas_nonzero = sigmas_all[nonzero_mask]
                weights_nonzero = weights_nonzero / np.sum(weights_nonzero)
            else:
                top_k_indices = np.argsort(weights)[-K:][::-1]
                weights_nonzero = weights[top_k_indices]
                mus_nonzero = mus_all[top_k_indices]
                sigmas_nonzero = sigmas_all[top_k_indices]
                weights_nonzero = weights_nonzero / np.sum(weights_nonzero)
        
        params = GMM1DParams(
            pi=weights_nonzero,
            mu=mus_nonzero,
            var=sigmas_nonzero**2
        )
        
        # Evaluate GMM PDF
        f_hat = gmm1d_pdf(z, params)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        # Compute PDF errors
        linf_err, l2_err = compute_pdf_errors(z, f_true, f_hat)
        pdf_errors_linf.append(linf_err)
        pdf_errors_l2.append(l2_err)
        
        # Compute moment errors
        stats_true = compute_pdf_statistics(z, f_true)
        if objective_mode in ["moments", "raw_moments"]:
            # Use direct moment computation for accuracy
            mean_hat, var_hat, skew_hat, kurt_hat = compute_gmm_moments_from_weights(
                weights_nonzero, mus_nonzero, sigmas_nonzero
            )
            stats_hat = {
                'mean': mean_hat,
                'std': np.sqrt(var_hat),
                'skewness': skew_hat,
                'kurtosis': kurt_hat
            }
        else:
            stats_hat = compute_pdf_statistics(z, f_hat)
        
        moment_errs = {
            'mean': stats_hat['mean'] - stats_true['mean'],
            'std': stats_hat['std'] - stats_true['std'],
            'variance': stats_hat['std']**2 - stats_true['std']**2,
            'skewness': stats_hat['skewness'] - stats_true['skewness'],
            'kurtosis': stats_hat['kurtosis'] - stats_true['kurtosis'],
        }
        moment_errors.append(moment_errs)
        
        n_nonzero_list.append(len(weights_nonzero))
        lp_objectives.append(lp_result["lp_objective"])
    
    # Average results
    result = BenchmarkResult(
        method="lp",
        K=K,
        z_npoints=len(z),
        objective_mode=objective_mode,
        execution_time=np.mean(times),
        pdf_error_linf=np.mean(pdf_errors_linf),
        pdf_error_l2=np.mean(pdf_errors_l2),
        mean_error=np.mean([e['mean'] for e in moment_errors]),
        std_error=np.mean([e['std'] for e in moment_errors]),
        variance_error=np.mean([e['variance'] for e in moment_errors]),
        skewness_error=np.mean([e['skewness'] for e in moment_errors]),
        kurtosis_error=np.mean([e['kurtosis'] for e in moment_errors]),
        n_nonzero_components=int(np.mean(n_nonzero_list)),
        lp_objective=np.mean(lp_objectives),
    )
    
    # Compute relative errors
    # For kurtosis and skewness, use a larger threshold to avoid unstable relative errors
    # when true values are very close to zero
    KURTOSIS_THRESHOLD = 0.01  # If |true_kurtosis| < 0.01, use threshold as denominator
    SKEWNESS_THRESHOLD = 0.01  # If |true_skewness| < 0.01, use threshold as denominator
    
    stats_true = compute_pdf_statistics(z, f_true)
    abs_mean = abs(stats_true['mean']) if abs(stats_true['mean']) > EPSILON else 1.0
    abs_std = abs(stats_true['std']) if abs(stats_true['std']) > EPSILON else 1.0
    abs_var = abs(stats_true['std']**2) if abs(stats_true['std']**2) > EPSILON else 1.0
    
    # For skewness and kurtosis, use threshold when true value is very small
    abs_skew_true = abs(stats_true['skewness'])
    abs_kurt_true = abs(stats_true['kurtosis'])
    abs_skew = abs_skew_true if abs_skew_true > SKEWNESS_THRESHOLD else SKEWNESS_THRESHOLD
    abs_kurt = abs_kurt_true if abs_kurt_true > KURTOSIS_THRESHOLD else KURTOSIS_THRESHOLD
    
    result.mean_error_rel = (result.mean_error / abs_mean) * 100.0
    result.std_error_rel = (result.std_error / abs_std) * 100.0
    result.variance_error_rel = (result.variance_error / abs_var) * 100.0
    
    # For skewness and kurtosis, cap relative error calculation when true value is very small
    if abs_skew_true > SKEWNESS_THRESHOLD:
        result.skewness_error_rel = (result.skewness_error / abs_skew_true) * 100.0
    else:
        # Use absolute error scaled by threshold when true value is too small
        result.skewness_error_rel = (result.skewness_error / SKEWNESS_THRESHOLD) * 100.0
    
    if abs_kurt_true > KURTOSIS_THRESHOLD:
        result.kurtosis_error_rel = (result.kurtosis_error / abs_kurt_true) * 100.0
    else:
        # Use absolute error scaled by threshold when true value is too small
        result.kurtosis_error_rel = (result.kurtosis_error / KURTOSIS_THRESHOLD) * 100.0
    
    return result


def run_comprehensive_benchmark(
    config_path: str, 
    output_path: str = "benchmark_results.json",
    vary_base_params: bool = False,
    param_configs: Optional[List[Dict]] = None
):
    """
    Run comprehensive benchmark across various parameter configurations.
    
    Parameters:
    -----------
    config_path : str
        Path to base configuration file
    output_path : str
        Path to save benchmark results (JSON)
    vary_base_params : bool
        If True, test multiple base parameter configurations
    param_configs : List[Dict], optional
        List of parameter configurations to test. Each dict should contain
        mu_x, sigma_x, mu_y, sigma_y, rho, z_range. If None and vary_base_params=True,
        uses default parameter sets.
    """
    print("=" * 80)
    print("Comprehensive Performance Benchmark")
    print("=" * 80)
    
    # Load base configuration
    # Use direct JSON loading to preserve all keys including quick_benchmark
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Define parameter configurations to test
    if vary_base_params and param_configs is None:
        # Default parameter sets to test
        # Covering various scenarios:
        # - Different correlation levels (low, medium, high)
        # - Different mean offsets
        # - Different variance ratios
        # - Symmetric and asymmetric cases
        param_configs = [
            # Original sets
            {"mu_x": 0.0, "sigma_x": 0.8, "mu_y": 0.0, "sigma_y": 1.6, "rho": 0.9, "z_range": [-6.0, 8.0]},
            {"mu_x": 0.1, "sigma_x": 0.4, "mu_y": 0.15, "sigma_y": 0.9, "rho": 0.9, "z_range": [-2.0, 4.0]},
            {"mu_x": 0.05, "sigma_x": 0.4, "mu_y": 0.15, "sigma_y": 0.9, "rho": 0.9, "z_range": [-2.0, 4.0]},
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.5, "z_range": [-4.0, 4.0]},
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.95, "z_range": [-4.0, 4.0]},
            # Additional sets: Low correlation
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.0, "z_range": [-4.0, 4.0]},
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.3, "z_range": [-4.0, 4.0]},
            # Additional sets: Different mean offsets
            {"mu_x": 0.5, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.7, "z_range": [-3.0, 5.0]},
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.5, "sigma_y": 1.0, "rho": 0.7, "z_range": [-3.0, 5.0]},
            {"mu_x": 0.3, "sigma_x": 0.8, "mu_y": -0.2, "sigma_y": 1.2, "rho": 0.6, "z_range": [-4.0, 4.0]},
            # Additional sets: Different variance ratios
            {"mu_x": 0.0, "sigma_x": 0.5, "mu_y": 0.0, "sigma_y": 2.0, "rho": 0.8, "z_range": [-5.0, 6.0]},
            {"mu_x": 0.0, "sigma_x": 2.0, "mu_y": 0.0, "sigma_y": 0.5, "rho": 0.8, "z_range": [-6.0, 5.0]},
            {"mu_x": 0.2, "sigma_x": 0.6, "mu_y": 0.1, "sigma_y": 1.4, "rho": 0.85, "z_range": [-3.0, 5.0]},
            # Additional sets: High correlation edge cases
            {"mu_x": 0.0, "sigma_x": 1.0, "mu_y": 0.0, "sigma_y": 1.0, "rho": 0.99, "z_range": [-4.0, 4.0]},
            {"mu_x": 0.1, "sigma_x": 0.5, "mu_y": 0.1, "sigma_y": 0.5, "rho": 0.95, "z_range": [-2.0, 3.0]},
        ]
    elif not vary_base_params:
        # Use only base configuration
        param_configs = [{
            "mu_x": base_config["mu_x"],
            "sigma_x": base_config["sigma_x"],
            "mu_y": base_config["mu_y"],
            "sigma_y": base_config["sigma_y"],
            "rho": base_config["rho"],
            "z_range": base_config["z_range"]
        }]
    
    all_results: List[BenchmarkResult] = []
    
    for param_idx, param_set in enumerate(param_configs):
        print(f"\n{'=' * 80}")
        print(f"Parameter Set {param_idx + 1}/{len(param_configs)}")
        print(f"{'=' * 80}")
        print(f"  mu_x={param_set['mu_x']}, sigma_x={param_set['sigma_x']}")
        print(f"  mu_y={param_set['mu_y']}, sigma_y={param_set['sigma_y']}, rho={param_set['rho']}")
        print(f"  z_range={param_set['z_range']}")
        
        # Create config with these parameters
        config = base_config.copy()
        config.update(param_set)
        
        # Extract parameters
        mu_x = config["mu_x"]
        sigma_x = config["sigma_x"]
        mu_y = config["mu_y"]
        sigma_y = config["sigma_y"]
        rho = config["rho"]
        z_range = config["z_range"]
    
        # Define test configurations
        # For EM method, exclude K=3 due to insufficient degrees of freedom for moment matching
        # Use reduced sets for faster execution (can be overridden via config)
        # Check quick_benchmark flag from base_config (before update) or config (after update)
        use_quick_mode = base_config.get("quick_benchmark", False) or config.get("quick_benchmark", False)
        if use_quick_mode:
            print(f"  Using QUICK mode (reduced parameter sets)")
            K_values = [5, 10] if config.get("method") == "em" else [5, 10]
            z_npoints_values = [128, 256]
            L_values = [10] if config.get("method") == "lp" else [10]
        else:
            K_values = [4, 5, 10, 15, 20] if config.get("method") == "em" else [5, 10, 15, 20]
            z_npoints_values = [64, 128, 256, 512]
            L_values = [5, 10, 15] if config.get("method") == "lp" else [10]
        
        results: List[BenchmarkResult] = []
        
        # Generate true PDF on fine grid
        z_fine = np.linspace(z_range[0], z_range[1], 1000)
        f_true_fine = max_pdf_bivariate_normal(
            z_fine, mu_x, sigma_x**2, mu_y, sigma_y**2, rho
        )
        f_true_fine = normalize_pdf_on_grid(z_fine, f_true_fine)
        
        print(f"\nRunning benchmarks...")
        print(f"  K values: {K_values}")
        print(f"  Grid resolutions: {z_npoints_values}")
        if config.get("method") == "lp":
            print(f"  L values: {L_values}")
            print(f"  LP modes: PDF, Moments, Raw Moments")
        elif config.get("method") == "em":
            print(f"  EM modes: PDF-only and Moment-matching")
        elif config.get("method") == "hybrid":
            print(f"  Hybrid modes: with/without moment matching")
        
        # Calculate total runs
        if config.get("method") == "em":
            total_runs = len(K_values) * len(z_npoints_values) * len(L_values) * 2  # 2 modes
        elif config.get("method") == "lp":
            total_runs = len(K_values) * len(z_npoints_values) * len(L_values) * 3  # 3 modes (pdf, moments, raw_moments)
        elif config.get("method") == "hybrid":
            total_runs = len(K_values) * len(z_npoints_values) * len(L_values) * 2  # 2 modes (with/without moment matching)
        else:
            total_runs = len(K_values) * len(z_npoints_values) * len(L_values)
        current_run = 0
    
        for z_npoints in z_npoints_values:
            print(f"\n  Grid resolution: {z_npoints} points")
            
            # Generate grid and true PDF
            z = np.linspace(z_range[0], z_range[1], z_npoints)
            f_true = max_pdf_bivariate_normal(
                z, mu_x, sigma_x**2, mu_y, sigma_y**2, rho
            )
            f_true = normalize_pdf_on_grid(z, f_true)
            
            for K in K_values:
                for L in L_values:
                    current_run += 1
                    print(f"    [{current_run}/{total_runs}] K={K}, L={L}...", end=" ", flush=True)
                    
                    try:
                        if config.get("method") == "em":
                            # Test both PDF-only and moment-matching modes
                            for use_mm in [False, True]:
                                try:
                                    config_mm = config.copy()
                                    config_mm["use_moment_matching"] = use_mm
                                    result = benchmark_em_method(z, f_true, K, config_mm, n_trials=1)
                                    result.objective_mode = "moments" if use_mm else "pdf"
                                    # Store parameter set info (will be added to dataclass fields)
                                    if vary_base_params or len(param_configs) > 1:
                                        result.param_set_idx = param_idx
                                        result.mu_x = mu_x
                                        result.sigma_x = sigma_x
                                        result.mu_y = mu_y
                                        result.sigma_y = sigma_y
                                        result.rho = rho
                                    results.append(result)
                                    mode_str = "moments" if use_mm else "pdf"
                                    print(f"✓ {mode_str} ({result.execution_time:.4f}s)", end=" ")
                                except Exception as e:
                                    mode_str = "moments" if use_mm else "pdf"
                                    print(f"✗ {mode_str} ({str(e)[:30]})", end=" ")
                            print()
                        elif config.get("method") == "lp":
                            # Test PDF, moments, and raw_moments modes
                            for obj_mode in ["pdf", "moments", "raw_moments"]:
                                try:
                                    result = benchmark_lp_method(
                                        z, f_true, K, L, config, 
                                        objective_mode=obj_mode,
                                        tail_focus="none",
                                        tail_alpha=1.0,
                                        n_trials=1
                                    )
                                    # Store parameter set info (will be added to dataclass fields)
                                    if vary_base_params or len(param_configs) > 1:
                                        result.param_set_idx = param_idx
                                        result.mu_x = mu_x
                                        result.sigma_x = sigma_x
                                        result.mu_y = mu_y
                                        result.sigma_y = sigma_y
                                        result.rho = rho
                                    results.append(result)
                                    print(f"✓ {obj_mode} ({result.execution_time:.4f}s)", end=" ")
                                except Exception as e:
                                    print(f"✗ {obj_mode} ({str(e)[:30]})", end=" ")
                            print()
                        elif config.get("method") == "hybrid":
                            # Test Hybrid method with different configurations
                            lp_params = config.get("lp_params", {})
                            dict_J = lp_params.get("dict_J", 4 * K)
                            dict_L = lp_params.get("dict_L", L)
                            
                            # Test with and without moment matching
                            for use_mm in [False, True]:
                                try:
                                    from benchmark_hybrid import benchmark_hybrid_method
                                    result_hybrid = benchmark_hybrid_method(
                                        z, f_true, K, dict_J, dict_L, config,
                                        tail_focus=lp_params.get("tail_focus", "none"),
                                        tail_alpha=lp_params.get("tail_alpha", 1.0),
                                        use_moment_matching=use_mm,
                                        n_trials=1
                                    )
                                    
                                    # Convert to BenchmarkResult format
                                    result = BenchmarkResult(
                                        method="hybrid",
                                        K=K,
                                        z_npoints=len(z),
                                        objective_mode="raw_moments",
                                        execution_time=result_hybrid.execution_time,
                                        pdf_error_linf=result_hybrid.pdf_error_linf,
                                        pdf_error_l2=0.0,  # Not computed in hybrid benchmark
                                        cdf_error_linf=result_hybrid.cdf_error_linf,
                                        tail_l1_error=result_hybrid.tail_l1_error,
                                        mean_error=result_hybrid.mean_error,
                                        std_error=result_hybrid.std_error,
                                        variance_error=result_hybrid.variance_error,
                                        skewness_error=result_hybrid.skewness_error,
                                        kurtosis_error=result_hybrid.kurtosis_error,
                                        mean_error_rel=result_hybrid.mean_error_rel,
                                        std_error_rel=result_hybrid.std_error_rel,
                                        variance_error_rel=result_hybrid.variance_error_rel,
                                        skewness_error_rel=result_hybrid.skewness_error_rel,
                                        kurtosis_error_rel=result_hybrid.kurtosis_error_rel,
                                        n_iterations=result_hybrid.n_iterations,
                                        converged=result_hybrid.converged,
                                        n_nonzero_components=K,
                                        tail_focus=result_hybrid.tail_focus,
                                        dict_J=dict_J,
                                        dict_L=dict_L,
                                        lp_runtime=result_hybrid.lp_runtime,
                                        em_runtime=result_hybrid.em_runtime,
                                        qp_runtime=result_hybrid.qp_runtime,
                                    )
                                    
                                    if vary_base_params or len(param_configs) > 1:
                                        result.param_set_idx = param_idx
                                        result.mu_x = mu_x
                                        result.sigma_x = sigma_x
                                        result.mu_y = mu_y
                                        result.sigma_y = sigma_y
                                        result.rho = rho
                                    
                                    results.append(result)
                                    mm_str = "mm" if use_mm else "no-mm"
                                    print(f"✓ hybrid-{mm_str} ({result.execution_time:.4f}s)", end=" ")
                                except Exception as e:
                                    mm_str = "mm" if use_mm else "no-mm"
                                    print(f"✗ hybrid-{mm_str} ({str(e)[:30]})", end=" ")
                            print()
                    except Exception as e:
                        print(f"✗ Error: {str(e)[:50]}")
        
        all_results.extend(results)
        print(f"\n  Parameter set {param_idx + 1} completed: {len(results)} runs")
    
    # Save results
    # Convert to dict, handling optional parameter fields
    results_dict = []
    for r in all_results:
        r_dict = asdict(r)
        # Remove None or unset parameter fields if not varying params
        if not (vary_base_params or len(param_configs) > 1):
            for key in ['param_set_idx', 'mu_x', 'sigma_x', 'mu_y', 'sigma_y', 'rho']:
                if key in r_dict and r_dict[key] == 0:
                    del r_dict[key]
        results_dict.append(r_dict)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Benchmark completed. Results saved to: {output_path}")
    print(f"Total parameter sets: {len(param_configs)}")
    print(f"Total runs: {len(all_results)}")
    print(f"{'=' * 80}")
    
    # Print summary statistics
    print_summary_statistics(all_results)
    
    # Print statistics by parameter set if multiple sets were tested
    if len(param_configs) > 1:
        print_summary_statistics_by_params(all_results, param_configs)
    
    return all_results


def print_summary_statistics_by_params(results: List[BenchmarkResult], param_configs: List[Dict]):
    """Print summary statistics grouped by parameter set."""
    print("\n" + "=" * 80)
    print("Summary Statistics by Parameter Set")
    print("=" * 80)
    
    for param_idx, param_set in enumerate(param_configs):
        param_results = [r for r in results if hasattr(r, 'param_set_idx') and r.param_set_idx == param_idx]
        if not param_results:
            continue
        
        print(f"\nParameter Set {param_idx + 1}:")
        print(f"  mu_x={param_set['mu_x']}, sigma_x={param_set['sigma_x']}")
        print(f"  mu_y={param_set['mu_y']}, sigma_y={param_set['sigma_y']}, rho={param_set['rho']}")
        print(f"  z_range={param_set['z_range']}")
        
        em_pdf = [r for r in param_results if r.method == "em" and r.objective_mode == "pdf"]
        em_moments = [r for r in param_results if r.method == "em" and r.objective_mode == "moments"]
        lp_pdf = [r for r in param_results if r.method == "lp" and r.objective_mode == "pdf"]
        lp_moments = [r for r in param_results if r.method == "lp" and r.objective_mode == "moments"]
        lp_rawmoments = [r for r in param_results if r.method == "lp" and r.objective_mode == "raw_moments"]
        hybrid_results = [r for r in param_results if r.method == "hybrid"]
        
        if em_pdf:
            print(f"  EM (PDF): time={np.mean([r.execution_time for r in em_pdf]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in em_pdf]):.6f}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in em_pdf]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in em_pdf]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in em_pdf]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in em_pdf]):.2f}%")
        if em_moments:
            print(f"  EM (Moments): time={np.mean([r.execution_time for r in em_moments]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in em_moments]):.6f}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in em_moments]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in em_moments]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in em_moments]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in em_moments]):.2f}%")
        if lp_pdf:
            print(f"  LP (PDF): time={np.mean([r.execution_time for r in lp_pdf]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in lp_pdf]):.6f}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in lp_pdf]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in lp_pdf]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in lp_pdf]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in lp_pdf]):.2f}%")
        if lp_moments:
            print(f"  LP (Moments): time={np.mean([r.execution_time for r in lp_moments]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in lp_moments]):.6f}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in lp_moments]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in lp_moments]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in lp_moments]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in lp_moments]):.2f}%")
        if lp_rawmoments:
            print(f"  LP (Raw Moments): time={np.mean([r.execution_time for r in lp_rawmoments]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in lp_rawmoments]):.6f}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in lp_rawmoments]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in lp_rawmoments]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in lp_rawmoments]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in lp_rawmoments]):.2f}%")
        if hybrid_results:
            print(f"  Hybrid: time={np.mean([r.execution_time for r in hybrid_results]):.4f}s, "
                  f"PDF_err={np.mean([r.pdf_error_linf for r in hybrid_results]):.6f}")
            print(f"    LP time: {np.mean([r.lp_runtime for r in hybrid_results]):.4f}s, "
                  f"EM time: {np.mean([r.em_runtime for r in hybrid_results]):.4f}s, "
                  f"QP time: {np.mean([r.qp_runtime for r in hybrid_results]):.4f}s")
            print(f"    CDF L∞: {np.mean([r.cdf_error_linf for r in hybrid_results]):.6e}, "
                  f"Tail L1: {np.mean([r.tail_l1_error for r in hybrid_results]):.6e}")
            print(f"    Moment errors (rel): mean={np.mean([abs(r.mean_error_rel) for r in hybrid_results]):.2f}%, "
                  f"std={np.mean([abs(r.std_error_rel) for r in hybrid_results]):.2f}%, "
                  f"skew={np.mean([abs(r.skewness_error_rel) for r in hybrid_results]):.2f}%, "
                  f"kurt={np.mean([abs(r.kurtosis_error_rel) for r in hybrid_results]):.2f}%")


def print_summary_statistics(results: List[BenchmarkResult]):
    """Print summary statistics from benchmark results."""
    print("\n" + "=" * 80)
    print("Summary Statistics (Overall)")
    print("=" * 80)
    
    # Group by method and objective mode
    em_pdf_results = [r for r in results if r.method == "em" and r.objective_mode == "pdf"]
    em_moments_results = [r for r in results if r.method == "em" and r.objective_mode == "moments"]
    lp_pdf_results = [r for r in results if r.method == "lp" and r.objective_mode == "pdf"]
    lp_moments_results = [r for r in results if r.method == "lp" and r.objective_mode == "moments"]
    lp_rawmoments_results = [r for r in results if r.method == "lp" and r.objective_mode == "raw_moments"]
    hybrid_results = [r for r in results if r.method == "hybrid"]
    
    # Check for large kurtosis errors in EM moments mode
    if em_moments_results:
        kurt_errors = [abs(r.kurtosis_error_rel) for r in em_moments_results]
        large_kurt_cases = [r for r in em_moments_results if abs(r.kurtosis_error_rel) > 10]
        if large_kurt_cases:
            # Note: K=3 cases are excluded from benchmark, but check for any remaining issues
            print(f"\n⚠️  Warning: {len(large_kurt_cases)} cases show large kurtosis errors "
                  f"(mean: {np.mean([abs(r.kurtosis_error_rel) for r in large_kurt_cases]):.1f}%). "
                  f"This may indicate moment matching difficulties.")
    
    if em_pdf_results:
        print("\nEM Method (PDF-only mode):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in em_pdf_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in em_pdf_results]):.6f}")
        print(f"  Average PDF L2 error: {np.mean([r.pdf_error_l2 for r in em_pdf_results]):.6f}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in em_pdf_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in em_pdf_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in em_pdf_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in em_pdf_results]):.4f}%")
    
    if em_moments_results:
        print("\nEM Method (Moment-matching mode):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in em_moments_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in em_moments_results]):.6f}")
        print(f"  Average PDF L2 error: {np.mean([r.pdf_error_l2 for r in em_moments_results]):.6f}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in em_moments_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in em_moments_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in em_moments_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in em_moments_results]):.4f}%")
    
    if lp_pdf_results:
        print("\nLP Method (PDF mode):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in lp_pdf_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in lp_pdf_results]):.6f}")
        print(f"  Average PDF L2 error: {np.mean([r.pdf_error_l2 for r in lp_pdf_results]):.6f}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in lp_pdf_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in lp_pdf_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in lp_pdf_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in lp_pdf_results]):.4f}%")
    
    if lp_moments_results:
        print("\nLP Method (Moments mode):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in lp_moments_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in lp_moments_results]):.6f}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in lp_moments_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in lp_moments_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in lp_moments_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in lp_moments_results]):.4f}%")
    
    if lp_rawmoments_results:
        print("\nLP Method (Raw Moments mode):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in lp_rawmoments_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in lp_rawmoments_results]):.6e}")
        print(f"  Average CDF L∞ error: {np.mean([r.cdf_error_linf for r in lp_rawmoments_results]):.6e}")
        print(f"  Average tail L1 error: {np.mean([r.tail_l1_error for r in lp_rawmoments_results]):.6e}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in lp_rawmoments_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in lp_rawmoments_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in lp_rawmoments_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in lp_rawmoments_results]):.4f}%")
    
    if hybrid_results:
        print("\nHybrid Method (LP→EM→QP):")
        print(f"  Average execution time: {np.mean([r.execution_time for r in hybrid_results]):.4f}s")
        print(f"    LP runtime: {np.mean([r.lp_runtime for r in hybrid_results]):.4f}s")
        print(f"    EM runtime: {np.mean([r.em_runtime for r in hybrid_results]):.4f}s")
        print(f"    QP runtime: {np.mean([r.qp_runtime for r in hybrid_results]):.4f}s")
        print(f"  Average PDF L∞ error: {np.mean([r.pdf_error_linf for r in hybrid_results]):.6e}")
        print(f"  Average CDF L∞ error: {np.mean([r.cdf_error_linf for r in hybrid_results]):.6e}")
        print(f"  Average tail L1 error: {np.mean([r.tail_l1_error for r in hybrid_results]):.6e}")
        print(f"  Average mean error (rel): {np.mean([abs(r.mean_error_rel) for r in hybrid_results]):.4f}%")
        print(f"  Average std error (rel): {np.mean([abs(r.std_error_rel) for r in hybrid_results]):.4f}%")
        print(f"  Average skewness error (rel): {np.mean([abs(r.skewness_error_rel) for r in hybrid_results]):.4f}%")
        print(f"  Average kurtosis error (rel): {np.mean([abs(r.kurtosis_error_rel) for r in hybrid_results]):.4f}%")


def plot_benchmark_results(results: List[BenchmarkResult], output_path: str = "benchmark_plots.png"):
    """
    Create visualization plots for benchmark results.
    
    Parameters:
    -----------
    results : List[BenchmarkResult]
        Benchmark results
    output_path : str
        Path to save plots
    """
    # Group results
    em_pdf_results = [r for r in results if r.method == "em" and r.objective_mode == "pdf"]
    em_moments_results = [r for r in results if r.method == "em" and r.objective_mode == "moments"]
    lp_pdf_results = [r for r in results if r.method == "lp" and r.objective_mode == "pdf"]
    lp_moments_results = [r for r in results if r.method == "lp" and r.objective_mode == "moments"]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Plot 1: Execution time vs K
    ax = axes[0, 0]
    if em_pdf_results:
        K_em = [r.K for r in em_pdf_results]
        time_em = [r.execution_time for r in em_pdf_results]
        ax.scatter(K_em, time_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        K_em_m = [r.K for r in em_moments_results]
        time_em_m = [r.execution_time for r in em_moments_results]
        ax.scatter(K_em_m, time_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        K_lp = [r.K for r in lp_pdf_results]
        time_lp = [r.execution_time for r in lp_pdf_results]
        ax.scatter(K_lp, time_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        K_lp_m = [r.K for r in lp_moments_results]
        time_lp_m = [r.execution_time for r in lp_moments_results]
        ax.scatter(K_lp_m, time_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("K (Number of Components)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Execution Time vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PDF L∞ error vs K
    ax = axes[0, 1]
    if em_pdf_results:
        K_em = [r.K for r in em_pdf_results]
        err_em = [r.pdf_error_linf for r in em_pdf_results]
        ax.scatter(K_em, err_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        K_em_m = [r.K for r in em_moments_results]
        err_em_m = [r.pdf_error_linf for r in em_moments_results]
        ax.scatter(K_em_m, err_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        K_lp = [r.K for r in lp_pdf_results]
        err_lp = [r.pdf_error_linf for r in lp_pdf_results]
        ax.scatter(K_lp, err_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        K_lp_m = [r.K for r in lp_moments_results]
        err_lp_m = [r.pdf_error_linf for r in lp_moments_results]
        ax.scatter(K_lp_m, err_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("K (Number of Components)")
    ax.set_ylabel("PDF L∞ Error")
    ax.set_title("PDF L∞ Error vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Standard deviation error vs K
    ax = axes[0, 2]
    if em_pdf_results:
        K_em = [r.K for r in em_pdf_results]
        std_em = [abs(r.std_error_rel) for r in em_pdf_results]
        ax.scatter(K_em, std_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        K_em_m = [r.K for r in em_moments_results]
        std_em_m = [abs(r.std_error_rel) for r in em_moments_results]
        ax.scatter(K_em_m, std_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        K_lp = [r.K for r in lp_pdf_results]
        std_lp = [abs(r.std_error_rel) for r in lp_pdf_results]
        ax.scatter(K_lp, std_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        K_lp_m = [r.K for r in lp_moments_results]
        std_lp_m = [abs(r.std_error_rel) for r in lp_moments_results]
        ax.scatter(K_lp_m, std_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("K (Number of Components)")
    ax.set_ylabel("Std Dev Error (rel %)")
    ax.set_title("Standard Deviation Error vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Execution time vs grid resolution
    ax = axes[1, 0]
    if em_pdf_results:
        n_em = [r.z_npoints for r in em_pdf_results]
        time_em = [r.execution_time for r in em_pdf_results]
        ax.scatter(n_em, time_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        n_em_m = [r.z_npoints for r in em_moments_results]
        time_em_m = [r.execution_time for r in em_moments_results]
        ax.scatter(n_em_m, time_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        n_lp = [r.z_npoints for r in lp_pdf_results]
        time_lp = [r.execution_time for r in lp_pdf_results]
        ax.scatter(n_lp, time_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        n_lp_m = [r.z_npoints for r in lp_moments_results]
        time_lp_m = [r.execution_time for r in lp_moments_results]
        ax.scatter(n_lp_m, time_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("Grid Resolution (points)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Execution Time vs Grid Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: PDF L2 error vs grid resolution
    ax = axes[1, 1]
    if em_pdf_results:
        n_em = [r.z_npoints for r in em_pdf_results]
        err_em = [r.pdf_error_l2 for r in em_pdf_results]
        ax.scatter(n_em, err_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        n_em_m = [r.z_npoints for r in em_moments_results]
        err_em_m = [r.pdf_error_l2 for r in em_moments_results]
        ax.scatter(n_em_m, err_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        n_lp = [r.z_npoints for r in lp_pdf_results]
        err_lp = [r.pdf_error_l2 for r in lp_pdf_results]
        ax.scatter(n_lp, err_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        n_lp_m = [r.z_npoints for r in lp_moments_results]
        err_lp_m = [r.pdf_error_l2 for r in lp_moments_results]
        ax.scatter(n_lp_m, err_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("Grid Resolution (points)")
    ax.set_ylabel("PDF L2 Error")
    ax.set_title("PDF L2 Error vs Grid Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: Skewness error vs K
    ax = axes[1, 2]
    if em_pdf_results:
        K_em = [r.K for r in em_pdf_results]
        skew_em = [abs(r.skewness_error_rel) for r in em_pdf_results]
        ax.scatter(K_em, skew_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        K_em_m = [r.K for r in em_moments_results]
        skew_em_m = [abs(r.skewness_error_rel) for r in em_moments_results]
        ax.scatter(K_em_m, skew_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        K_lp = [r.K for r in lp_pdf_results]
        skew_lp = [abs(r.skewness_error_rel) for r in lp_pdf_results]
        ax.scatter(K_lp, skew_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        K_lp_m = [r.K for r in lp_moments_results]
        skew_lp_m = [abs(r.skewness_error_rel) for r in lp_moments_results]
        ax.scatter(K_lp_m, skew_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("K (Number of Components)")
    ax.set_ylabel("Skewness Error (rel %)")
    ax.set_title("Skewness Error vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Kurtosis error vs K
    ax = axes[2, 0]
    if em_pdf_results:
        K_em = [r.K for r in em_pdf_results]
        kurt_em = [abs(r.kurtosis_error_rel) for r in em_pdf_results]
        ax.scatter(K_em, kurt_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        K_em_m = [r.K for r in em_moments_results]
        kurt_em_m = [abs(r.kurtosis_error_rel) for r in em_moments_results]
        ax.scatter(K_em_m, kurt_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        K_lp = [r.K for r in lp_pdf_results]
        kurt_lp = [abs(r.kurtosis_error_rel) for r in lp_pdf_results]
        ax.scatter(K_lp, kurt_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        K_lp_m = [r.K for r in lp_moments_results]
        kurt_lp_m = [abs(r.kurtosis_error_rel) for r in lp_moments_results]
        ax.scatter(K_lp_m, kurt_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("K (Number of Components)")
    ax.set_ylabel("Kurtosis Error (rel %)")
    ax.set_title("Kurtosis Error vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Standard deviation error vs grid resolution
    ax = axes[2, 1]
    if em_pdf_results:
        n_em = [r.z_npoints for r in em_pdf_results]
        std_em = [abs(r.std_error_rel) for r in em_pdf_results]
        ax.scatter(n_em, std_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        n_em_m = [r.z_npoints for r in em_moments_results]
        std_em_m = [abs(r.std_error_rel) for r in em_moments_results]
        ax.scatter(n_em_m, std_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        n_lp = [r.z_npoints for r in lp_pdf_results]
        std_lp = [abs(r.std_error_rel) for r in lp_pdf_results]
        ax.scatter(n_lp, std_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        n_lp_m = [r.z_npoints for r in lp_moments_results]
        std_lp_m = [abs(r.std_error_rel) for r in lp_moments_results]
        ax.scatter(n_lp_m, std_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("Grid Resolution (points)")
    ax.set_ylabel("Std Dev Error (rel %)")
    ax.set_title("Standard Deviation Error vs Grid Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Execution time vs accuracy trade-off
    ax = axes[2, 2]
    if em_pdf_results:
        time_em = [r.execution_time for r in em_pdf_results]
        err_em = [r.pdf_error_linf for r in em_pdf_results]
        ax.scatter(time_em, err_em, label="EM (PDF)", alpha=0.6)
    if em_moments_results:
        time_em_m = [r.execution_time for r in em_moments_results]
        err_em_m = [r.pdf_error_linf for r in em_moments_results]
        ax.scatter(time_em_m, err_em_m, label="EM (Moments)", alpha=0.6)
    if lp_pdf_results:
        time_lp = [r.execution_time for r in lp_pdf_results]
        err_lp = [r.pdf_error_linf for r in lp_pdf_results]
        ax.scatter(time_lp, err_lp, label="LP (PDF)", alpha=0.6)
    if lp_moments_results:
        time_lp_m = [r.execution_time for r in lp_moments_results]
        err_lp_m = [r.pdf_error_linf for r in lp_moments_results]
        ax.scatter(time_lp_m, err_lp_m, label="LP (Moments)", alpha=0.6)
    ax.set_xlabel("Execution Time (s)")
    ax.set_ylabel("PDF L∞ Error")
    ax.set_title("Time vs Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive performance benchmark for GMM fitting methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --config configs/config_default.json
  python benchmark.py --config configs/config_lp.json --output benchmark_lp.json --plot
  python benchmark.py --config configs/config_default.json --vary-params --output benchmark_varied.json --plot
        """
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
        default="benchmark_results.json",
        help="Path to save benchmark results (JSON)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--vary-params",
        action="store_true",
        help="Test multiple base parameter configurations (mu_x, sigma_x, mu_y, sigma_y, rho)"
    )
    parser.add_argument(
        "--param-configs",
        type=str,
        default=None,
        help="Path to JSON file containing list of parameter configurations to test"
    )
    
    args = parser.parse_args()
    
    # Load parameter configurations if provided
    param_configs = None
    if args.param_configs:
        with open(args.param_configs, 'r') as f:
            param_configs = json.load(f)
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        args.config, 
        args.output,
        vary_base_params=args.vary_params,
        param_configs=param_configs
    )
    
    # Generate plots if requested
    if args.plot:
        plot_output = args.output.replace('.json', '_plots.png')
        plot_benchmark_results(results, plot_output)


if __name__ == "__main__":
    main()

