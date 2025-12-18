"""
Main execution script for GMM fitting.

This script reads configuration from JSON file, computes the PDF of max(X, Y) where
(X, Y) is bivariate normal, fits a GMM using either EM or LP algorithm,
computes statistics, and generates comparison plots.
"""

import argparse
import sys
import os
import time
import numpy as np

# Add src directory to path to import gmm_fitting package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from gmm_fitting import (
    load_config,
    prepare_init_params,
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
    gmm1d_pdf,
    compute_pdf_statistics,
    GMM1DParams,
    print_section_header,
    print_em_results,
    print_execution_time,
    print_moment_matching_info,
    print_statistics_comparison,
    print_gmm_parameters,
    print_plot_output,
    plot_pdf_comparison,
    fit_gmm_lp_simple,
    solve_lp_pdf_rawmoments_linf,
    build_gaussian_dictionary,
    compute_basis_matrices,
    compute_pdf_raw_moments,
    compute_errors,
)
import numpy as np


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Approximate PDF of max(X, Y) using GMM where (X, Y) is bivariate normal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/config_default.json
  python main.py --config configs/config_lp.json
  
For more information, see README.md and docs/ directory.
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file. Example configs are in configs/ directory."
    )
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # If --config is not provided, show help
    if args.config is None:
        parser.print_help()
        sys.exit(0)

    # Load configuration
    config = load_config(args.config)
    
    # Extract parameters
    mu_x = config["mu_x"]
    sigma_x = config["sigma_x"]
    mu_y = config["mu_y"]
    sigma_y = config["sigma_y"]
    rho = config["rho"]
    z_range = config["z_range"]
    z_npoints = config["z_npoints"]
    K = config["K"]
    max_iter = config["max_iter"]
    tol = config["tol"]
    reg_var = config["reg_var"]
    n_init = config["n_init"]
    seed = config["seed"]
    init = config["init"]
    use_moment_matching = config["use_moment_matching"]
    qp_mode = config["qp_mode"]
    soft_lambda = config["soft_lambda"]
    output_path = config["output_path"]
    show_grid_points = config["show_grid_points"]
    max_grid_points_display = config["max_grid_points_display"]
    
    # Prepare initialization parameters
    init_params = prepare_init_params(config, init, mu_x, sigma_x, mu_y, sigma_y, rho)
    
    # Print configuration info
    print(f"Configuration file: {args.config}")
    print(f"Parameters: mu_x={mu_x}, sigma_x={sigma_x}, mu_y={mu_y}, sigma_y={sigma_y}, rho={rho}")
    
    # Generate uniform grid
    z = np.linspace(z_range[0], z_range[1], z_npoints)
    print(f"Using uniform grid with {len(z)} points")
    
    # Compute true PDF: f_Z(z) where Z = max(X, Y)
    f_true = max_pdf_bivariate_normal(z, mu_x, sigma_x**2, mu_y, sigma_y**2, rho)
    f_true = normalize_pdf_on_grid(z, f_true)
    
    # Determine fitting method
    method = config.get("method", "em")  # "em", "lp", or "hybrid"
    
    # Fit GMM using selected method (measure execution time)
    total_start_time = time.time()
    em_start_time = time.time()
    
    if method == "em":
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f_true,
            K=K,
            max_iter=max_iter,
            tol=tol,
            reg_var=reg_var,
            n_init=n_init,
            seed=seed,
            init=init,
            init_params=init_params,
            use_moment_matching=use_moment_matching,
            qp_mode=qp_mode,
            soft_lambda=soft_lambda,
        )
        total_em_time = time.time() - em_start_time
        
        # Extract QP execution time if moment matching was used
        qp_elapsed_time = 0.0
        qp_info = None
        if use_moment_matching and hasattr(params, '_qp_info'):
            qp_info = params._qp_info
            qp_elapsed_time = qp_info.get('qp_time', 0.0)
        
        # Subtract QP time from total to get pure EM time
        em_elapsed_time = total_em_time - qp_elapsed_time
        
        # Convert to log-likelihood for compatibility (LP doesn't use LL)
        # For EM, ll is already log-likelihood
        ll_value = ll
        
    elif method == "lp":
        # Get LP parameters from config
        # For simple LP method: K segments, L sigma levels per segment
        L = config.get("L", 6)  # Number of sigma levels per segment
        
        lp_params = config.get("lp_params", {
            "solver": "highs",
            "sigma_min_scale": 0.1,
            "sigma_max_scale": 3.0
        })
        
        # Get objective mode (default: "pdf")
        # Note: load_config now includes objective_mode in the returned config
        objective_mode = config.get("objective_mode", "pdf")
        
        # Fit using simple LP method (no greedy selection, no QP)
        lp_result, lp_timing = fit_gmm_lp_simple(
            z, f_true,
            K=K,
            L=L,
            lp_params=lp_params,
            objective_mode=objective_mode
        )
        
        em_elapsed_time = time.time() - em_start_time
        qp_elapsed_time = 0.0
        qp_info = None
        
        # Convert LP result to GMM1DParams format
        # For simple LP, all K*L bases are returned, but many weights may be zero
        # Extract only non-zero components for GMM1DParams
        weights = lp_result["weights"]
        mus_all = lp_result["mus"]
        sigmas_all = lp_result["sigmas"]
        
        # For moments mode, use all components to preserve moment accuracy
        # For pdf mode, extract non-zero components
        if objective_mode == "moments":
            # Use ALL components to preserve moment accuracy
            # Small weights are important for higher moments (especially kurtosis)
            weights_nonzero = weights
            mus_nonzero = mus_all
            sigmas_nonzero = sigmas_all
            # Do NOT renormalize for moments mode to preserve moment accuracy
            # The LP solver already ensures sum(weights) = 1
        else:
            # For pdf mode, extract non-zero components and renormalize
            nonzero_mask = weights > 1e-10
            if np.any(nonzero_mask):
                weights_nonzero = weights[nonzero_mask]
                mus_nonzero = mus_all[nonzero_mask]
                sigmas_nonzero = sigmas_all[nonzero_mask]
                # Renormalize weights
                weights_nonzero = weights_nonzero / np.sum(weights_nonzero)
            else:
                # Fallback: use top K components by weight
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
        
        # Store LP diagnostics and timing
        params._lp_info = lp_result["diagnostics"]
        params._lp_objective = lp_result["lp_objective"]
        params._lp_timing = lp_timing
        
        # For LP, we don't have log-likelihood, use negative objective as proxy
        n_iter = lp_result["diagnostics"].get("n_nonzero", len(weights_nonzero))
        ll_value = -lp_result["lp_objective"]  # Use negative objective as proxy
        
    elif method == "hybrid":
        # Hybrid method: LP → EM → QP
        lp_params = config.get("lp_params", {})
        
        # Get dictionary parameters (default: dict_J = 4*K, dict_L = L)
        dict_J = lp_params.get("dict_J", 4 * K)
        dict_L = lp_params.get("dict_L", config.get("L", 10))
        
        # Get LP objective mode (default: "raw_moments")
        objective_mode = lp_params.get("objective_mode", "raw_moments")
        
        # Step 1: Build dictionary and solve LP
        lp_start_time = time.time()
        
        # Build dictionary with tail focus
        dictionary = build_gaussian_dictionary(
            z, f_true,
            J=dict_J,
            L=dict_L,
            mu_mode=lp_params.get("mu_mode", "quantile"),
            sigma_min_scale=lp_params.get("sigma_min_scale", 0.1),
            sigma_max_scale=lp_params.get("sigma_max_scale", 3.0),
            tail_focus=lp_params.get("tail_focus", "none"),
            tail_alpha=lp_params.get("tail_alpha", 1.0),
            quantile_levels=lp_params.get("quantile_levels", None),
        )
        mus_dict = dictionary["mus"]
        sigmas_dict = dictionary["sigmas"]
        
        # Compute basis matrix
        basis = compute_basis_matrices(z, mus_dict, sigmas_dict)
        Phi_pdf = basis["Phi_pdf"]
        
        # Solve LP with raw moments
        if objective_mode == "raw_moments":
            lp_result_raw = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=Phi_pdf,
                mus=mus_dict,
                sigmas=sigmas_dict,
                z=z,
                f=f_true,
                pdf_tolerance=lp_params.get("pdf_tolerance", None),
                lambda_pdf=lp_params.get("lambda_pdf", 1.0),
                lambda_raw=tuple(lp_params.get("lambda_raw", [1.0, 1.0, 1.0, 1.0])),
                solver=lp_params.get("solver", "highs"),
                objective_form=lp_params.get("objective_form", "A"),
            )
            
            w_all = lp_result_raw["w"]
            lp_diagnostics = lp_result_raw["diagnostics"]
        else:
            raise ValueError(f"Hybrid method requires objective_mode='raw_moments', got '{objective_mode}'")
        
        lp_elapsed_time = time.time() - lp_start_time
        
        # Step 2: Select top K components from LP solution
        idx_top_k = np.argsort(w_all)[::-1][:K]
        pi_init = w_all[idx_top_k]
        mu_init = mus_dict[idx_top_k]
        var_init = sigmas_dict[idx_top_k]**2
        
        # Normalize pi_init
        pi_sum = np.sum(pi_init)
        if pi_sum <= 0:
            raise ValueError("LP solution has no positive weights for top K components")
        pi_init = pi_init / pi_sum
        
        # Clip var_init to reg_var
        var_init = np.maximum(var_init, reg_var)
        
        # Step 3: Run EM with custom initialization
        em_start_time = time.time()
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
        total_em_time = time.time() - em_start_time
        
        # Extract QP execution time if moment matching was used
        qp_elapsed_time = 0.0
        qp_info = None
        if use_moment_matching and hasattr(params, '_qp_info'):
            qp_info = params._qp_info
            qp_elapsed_time = qp_info.get('qp_time', 0.0)
        
        # Subtract QP time from total to get pure EM time
        em_elapsed_time = total_em_time - qp_elapsed_time
        
        ll_value = ll
        
        # Store Hybrid diagnostics
        params._hybrid_info = {
            "lp_runtime_sec": lp_elapsed_time,
            "em_runtime_sec": em_elapsed_time,
            "qp_runtime_sec": qp_elapsed_time,
            "lp_diagnostics": lp_diagnostics,
            "n_dict": len(mus_dict),
            "dict_J": dict_J,
            "dict_L": dict_L,
        }
        
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'em', 'lp', or 'hybrid'")
    
    total_elapsed_time = time.time() - total_start_time

    # Evaluate GMM PDF at grid points
    f_hat = gmm1d_pdf(z, params)
    f_hat = normalize_pdf_on_grid(z, f_hat)

    # Compute statistics for comparison
    # For moments mode with LP method, use direct moment computation for GMM (more accurate)
    # For other cases, use numerical integration on a fine grid
    objective_mode_for_stats = config.get("objective_mode", "pdf") if method == "lp" else "pdf"
    if method == "lp" and objective_mode_for_stats == "moments" and hasattr(params, '_lp_info'):
        # Use direct moment computation for GMM in moments mode (more accurate)
        from gmm_utils import compute_gmm_moments_from_weights
        # Compute GMM moments directly from params (all components are included for moments mode)
        mean_hat, var_hat, skew_hat, kurt_hat = compute_gmm_moments_from_weights(
            params.pi, params.mu, np.sqrt(params.var)
        )
        stats_hat = {
            'mean': mean_hat,
            'std': np.sqrt(var_hat),
            'skewness': skew_hat,
            'kurtosis': kurt_hat
        }
        
        # Compute true PDF statistics on original grid range (for consistency)
        z_stats = np.linspace(z_range[0], z_range[1], max(1000, z_npoints * 10))
        f_true_stats = np.interp(z_stats, z, f_true)
        stats_true = compute_pdf_statistics(z_stats, f_true_stats)
    else:
        # Use numerical integration for other cases
        z_stats = np.linspace(z_range[0], z_range[1], max(1000, z_npoints * 10))
        f_true_stats = np.interp(z_stats, z, f_true)
        f_hat_stats = gmm1d_pdf(z_stats, params)
        f_hat_stats = normalize_pdf_on_grid(z_stats, f_hat_stats)
        
        stats_true = compute_pdf_statistics(z_stats, f_true_stats)
        stats_hat = compute_pdf_statistics(z_stats, f_hat_stats)
    
    # Print results using formatted output functions
    if method == "em":
        method_name = "EM"
    elif method == "lp":
        # Check if using simple or greedy LP method
        if hasattr(params, '_lp_info') and "n_bases" in params._lp_info:
            method_name = "LP"
        else:
            method_name = "LP+Greedy"
    elif method == "hybrid":
        method_name = "Hybrid (LP→EM→QP)"
    else:
        method_name = "Unknown"
    print_section_header(f"{method_name.upper()} ALGORITHM RESULTS")
    if method == "em":
        print_em_results(ll_value, n_iter, max_iter)
    else:
        print(f"LP objective value: {ll_value:.10f}")
        if hasattr(params, '_lp_info'):
            lp_info = params._lp_info
            if "n_bases" in lp_info:
                # Simple LP method
                print(f"Dictionary size: {lp_info['n_bases']} bases (K={K}, L={lp_info.get('L', 'N/A')})")
                print(f"Non-zero components: {lp_info.get('n_nonzero', 'N/A')}")
            else:
                # Greedy LP method
                print(f"Components selected: {n_iter} / {K}")
            print(f"PDF error bound (t_pdf): {lp_info.get('t_pdf', 0):.6e}")
    
    # Get LP timing if available
    lp_timing = None
    if method == "lp" and hasattr(params, '_lp_timing'):
        lp_timing = params._lp_timing
    elif method == "hybrid" and hasattr(params, '_hybrid_info'):
        # For hybrid, show LP, EM, and QP times separately
        hybrid_info = params._hybrid_info
        lp_timing = {"lp": hybrid_info["lp_runtime_sec"]}
        em_elapsed_time = hybrid_info["em_runtime_sec"]
        qp_elapsed_time = hybrid_info["qp_runtime_sec"]
    
    print_execution_time(em_elapsed_time, qp_elapsed_time, total_elapsed_time, 
                        use_moment_matching, method=method, lp_timing=lp_timing)
    
    # Print Hybrid-specific information
    if method == "hybrid" and hasattr(params, '_hybrid_info'):
        hybrid_info = params._hybrid_info
        print(f"\nHybrid Method Details:")
        print(f"  Dictionary size: {hybrid_info['dict_J']} × {hybrid_info['dict_L']} = {hybrid_info['n_dict']} bases")
        print(f"  LP runtime: {hybrid_info['lp_runtime_sec']:.6f}s")
        print(f"  EM runtime: {hybrid_info['em_runtime_sec']:.6f}s")
        if hybrid_info['qp_runtime_sec'] > 0:
            print(f"  QP runtime: {hybrid_info['qp_runtime_sec']:.6f}s")
        lp_diag = hybrid_info['lp_diagnostics']
        if 't_pdf' in lp_diag:
            print(f"  LP PDF error: {lp_diag['t_pdf']:.6e}")
        if 'raw_abs_err' in lp_diag:
            print(f"  LP raw moment errors: {lp_diag['raw_abs_err']}")
    
    if (method == "em" or method == "hybrid") and use_moment_matching and qp_info is not None:
        print_moment_matching_info(qp_info)
    
    print_statistics_comparison(stats_true, stats_hat)
    print_gmm_parameters(params)
    
    # Compute error metrics (for all methods)
    errors = compute_errors(z, f_true, f_hat)
    
    # Print error metrics
    print(f"\nError Metrics:")
    print(f"  PDF L∞ error: {errors['linf_pdf']:.6e}")
    print(f"  CDF L∞ error: {errors['linf_cdf']:.6e}")
    print(f"  Quantile errors:")
    for p, err in errors['quantile_abs_errors'].items():
        print(f"    p={p:.3f}: {err:.6e}")
    print(f"  Tail-weighted L1 error (p0=0.9): {errors['tail_l1_error']:.6e}")
    
    # Generate and save PDF comparison plot
    plot_pdf_comparison(
        z, f_true, f_hat, output_path,
        mu_x, sigma_x, mu_y, sigma_y, rho, ll_value,
        show_grid_points=show_grid_points,
        max_grid_points_display=max_grid_points_display,
        gmm_params=params
    )
    print_plot_output(output_path)


if __name__ == "__main__":
    main()

