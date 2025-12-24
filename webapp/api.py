"""
FastAPI application for GMM fitting web service.
"""

import sys
import os
import time
import base64
import io
import json
import numpy as np
import torch
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src directory to path to import gmm_fitting package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from gmm_fitting import (
    prepare_init_params,
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
    gmm1d_pdf,
    compute_pdf_statistics,
    GMM1DParams,
    plot_pdf_comparison,
    fit_gmm_lp_simple,
    solve_lp_pdf_rawmoments_linf,
    build_gaussian_dictionary,
    compute_basis_matrices,
    compute_pdf_raw_moments,
    compute_errors,
    MIN_PDF_VALUE,
)
from webapp.models import (
    ComputeRequest,
    ComputeResponse,
    GMMComponent,
    Statistics,
    ErrorMetrics,
    ExecutionTime,
)

# LAMF model cache (singleton pattern)
_lamf_model_cache: Dict[str, Any] = {
    "model": None,
    "metadata": None,
    "z_grid": None,
}


def get_lamf_model():
    """Get cached LAMF model, loading if necessary."""
    global _lamf_model_cache
    
    if _lamf_model_cache["model"] is None:
        from lamf.model import LAMFFitter
        
        # Default LAMF model path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "lamf", "checkpoints_v5_linf_30ep")
        model_path = os.path.join(model_dir, "best_model.pt")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LAMF model not found at {model_path}")
        
        # Load metadata if exists
        metadata = {
            "N": 96,
            "K": 5,
            "T": 6,
            "z_min": -15.0,
            "z_max": 15.0,
        }
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata.update(json.load(f))
        
        # Create model
        model = LAMFFitter(
            N=metadata["N"],
            K=metadata["K"],
            T=metadata.get("T", 6),
            init_hidden_dim=metadata.get("init_hidden_dim", 256),
            init_num_layers=metadata.get("init_num_layers", 3),
            refine_hidden_dim=metadata.get("refine_hidden_dim", 128),
            refine_num_layers=metadata.get("refine_num_layers", 2),
            sigma_min=metadata.get("sigma_min", 1e-3),
        )
        
        # Load weights
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Create z grid for the model
        z_grid = np.linspace(metadata["z_min"], metadata["z_max"], metadata["N"])
        
        _lamf_model_cache["model"] = model
        _lamf_model_cache["metadata"] = metadata
        _lamf_model_cache["z_grid"] = z_grid
        
        print(f"LAMF model loaded from {model_path}")
    
    return _lamf_model_cache["model"], _lamf_model_cache["metadata"], _lamf_model_cache["z_grid"]


def fit_gmm_lamf(z: np.ndarray, f_true: np.ndarray, K: int = 5) -> tuple:
    """
    Fit GMM using LAMF model.
    
    The LAMF model is trained on relative coordinates (mean=0), so:
    1. Input PDF is shifted to have mean=0 before inference
    2. Output means are shifted back to original coordinates
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values
    K : int
        Number of GMM components (must match model K)
    
    Returns:
    --------
    gmm_params : GMM1DParams
        Fitted GMM parameters
    inference_time : float
        Inference time in seconds
    """
    model, metadata, model_z = get_lamf_model()
    
    if K != metadata["K"]:
        raise ValueError(f"LAMF model requires K={metadata['K']}, got K={K}")
    
    # Compute global mean of input PDF for coordinate transformation
    dz_input = z[1] - z[0] if len(z) > 1 else 1.0
    w_input = f_true * dz_input
    w_input = w_input / w_input.sum()  # Normalize
    global_mean = np.sum(w_input * z)
    
    # Shift input grid to have mean=0 (relative coordinates)
    z_shifted = z - global_mean
    
    # Interpolate shifted PDF to model grid
    # Model grid is [-15, 15], input may be different
    f_interp = np.interp(model_z, z_shifted, f_true, left=0.0, right=0.0)
    
    # Normalize to mass values (w)
    dz = model_z[1] - model_z[0]
    w = f_interp * dz
    w_sum = w.sum()
    if w_sum > 1e-12:
        w = w / w_sum
    else:
        # Fallback: uniform distribution if PDF is too small
        w = np.ones_like(w) / len(w)
    
    # Convert to tensors
    z_tensor = torch.from_numpy(model_z.astype(np.float32))
    w_tensor = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)  # (1, N)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        result = model(z_tensor, w_tensor)
    inference_time = time.time() - start_time
    
    # Extract parameters
    pi = result["pi"][0].numpy()
    mu = result["mu"][0].numpy()
    sigma = result["sigma"][0].numpy()
    
    # Shift means back to original coordinates
    mu = mu + global_mean
    
    # Create GMM1DParams
    gmm_params = GMM1DParams(
        pi=pi,
        mu=mu,
        var=sigma**2,
    )
    
    return gmm_params, inference_time

app = FastAPI(
    title="GMM Fitting API",
    description="API for approximating PDF of max(X,Y) using Gaussian Mixture Models",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_gmm_fitting(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core computation function extracted from main.py.
    
    Returns a dictionary with all results.
    """
    # Extract parameters
    mu_x = config_dict["mu_x"]
    sigma_x = config_dict["sigma_x"]
    mu_y = config_dict["mu_y"]
    sigma_y = config_dict["sigma_y"]
    rho = config_dict["rho"]
    z_range = config_dict["z_range"]
    z_npoints = config_dict["z_npoints"]
    K = config_dict["K"]
    method = config_dict.get("method", "em")
    
    # Prepare initialization parameters
    init_params = prepare_init_params(
        config_dict, 
        config_dict.get("init", "quantile"),
        mu_x, sigma_x, mu_y, sigma_y, rho
    )
    
    # Generate uniform grid
    z = np.linspace(z_range[0], z_range[1], z_npoints)
    
    # Compute true PDF
    f_true = max_pdf_bivariate_normal(z, mu_x, sigma_x**2, mu_y, sigma_y**2, rho)
    f_true = normalize_pdf_on_grid(z, f_true)
    
    # Fit GMM
    total_start_time = time.time()
    em_start_time = time.time()
    lp_start_time = time.time()  # Initialize for all methods
    
    # Initialize timing variables for all methods (must be before any conditional blocks)
    lp_elapsed_time: float = 0.0
    em_elapsed_time: float = 0.0
    qp_elapsed_time: float = 0.0
    
    if method == "em":
        max_iter = config_dict.get("max_iter", 400)
        tol = config_dict.get("tol", 1e-10)
        reg_var = config_dict.get("reg_var", 1e-6)
        n_init = config_dict.get("n_init", 8)
        seed = config_dict.get("seed", 1)
        init = config_dict.get("init", "quantile")
        use_moment_matching = config_dict.get("use_moment_matching", False)
        qp_mode = config_dict.get("qp_mode", "hard")
        soft_lambda = config_dict.get("soft_lambda", 1e4)
        
        # MDN parameters
        mdn_params = config_dict.get("mdn_params", {})
        mdn_model_path = mdn_params.get("model_path") if mdn_params else None
        mdn_device = mdn_params.get("device", "auto") if mdn_params else "auto"
        
        # LAMF parameters
        lamf_params = config_dict.get("lamf_params", {})
        lamf_model_path = lamf_params.get("model_path") if lamf_params else None
        lamf_device = lamf_params.get("device", "auto") if lamf_params else "auto"
        
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
            mdn_model_path=mdn_model_path,
            mdn_device=mdn_device if init == "mdn" else lamf_device,
            lamf_model_path=lamf_model_path,
        )
        total_em_time = time.time() - em_start_time
        
        qp_elapsed_time = 0.0
        qp_info = None
        if use_moment_matching and hasattr(params, '_qp_info'):
            qp_info = params._qp_info
            qp_elapsed_time = qp_info.get('qp_time', 0.0)
        
        # Get initialization time
        init_elapsed_time = 0.0
        if hasattr(params, '_init_time'):
            init_elapsed_time = params._init_time
        
        # Subtract QP time from total to get pure EM time
        em_elapsed_time = total_em_time - qp_elapsed_time
        
        ll_value = ll
        diagnostics = None
        
    elif method == "lp":
        L = config_dict.get("L", 5)
        objective_mode = config_dict.get("objective_mode", "pdf")
        
        lp_params_dict = config_dict.get("lp_params", {})
        lp_params_dict.setdefault("solver", config_dict.get("solver", "highs"))
        lp_params_dict.setdefault("sigma_min_scale", config_dict.get("sigma_min_scale", 0.1))
        lp_params_dict.setdefault("sigma_max_scale", config_dict.get("sigma_max_scale", 3.0))
        
        if objective_mode == "raw_moments":
            lp_params_dict.setdefault("lambda_raw", config_dict.get("lambda_raw", [1.0, 1.0, 1.0, 1.0]))
            lp_params_dict.setdefault("objective_form", config_dict.get("objective_form", "A"))
            lp_params_dict.setdefault("pdf_tolerance", config_dict.get("pdf_tolerance", None))
        
        lp_start_time = time.time()  # Reset for lp method
        lp_result, lp_timing = fit_gmm_lp_simple(
            z, f_true,
            K=K,
            L=L,
            lp_params=lp_params_dict,
            objective_mode=objective_mode
        )
        lp_elapsed_time = time.time() - lp_start_time
        
        em_elapsed_time = 0.0
        qp_elapsed_time = 0.0
        init_elapsed_time = 0.0  # LP doesn't have init time
        qp_info = None
        
        weights = lp_result["weights"]
        mus_all = lp_result["mus"]
        sigmas_all = lp_result["sigmas"]
        
        if objective_mode == "moments":
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
        
        params._lp_info = lp_result["diagnostics"]
        params._lp_objective = lp_result["lp_objective"]
        params._lp_timing = lp_timing
        
        n_iter = int(lp_result["diagnostics"].get("n_nonzero", len(weights_nonzero)))
        ll_value = -lp_result["lp_objective"]
        # Convert numpy types in diagnostics to Python native types
        diagnostics = {}
        for key, value in lp_result["diagnostics"].items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                diagnostics[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                diagnostics[key] = float(value)
            elif isinstance(value, np.ndarray):
                diagnostics[key] = value.tolist()
            else:
                diagnostics[key] = value
        
    elif method == "hybrid":
        lp_params_dict = config_dict.get("lp_params", {})
        dict_J = lp_params_dict.get("dict_J", 4 * K)
        dict_L = lp_params_dict.get("dict_L", config_dict.get("L", 10))
        objective_mode = lp_params_dict.get("objective_mode", "raw_moments")
        
        lp_start_time = time.time()  # Reset for hybrid method
        
        dictionary = build_gaussian_dictionary(
            z, f_true,
            J=dict_J,
            L=dict_L,
            mu_mode=lp_params_dict.get("mu_mode", "quantile"),
            sigma_min_scale=lp_params_dict.get("sigma_min_scale", 0.1),
            sigma_max_scale=lp_params_dict.get("sigma_max_scale", 3.0),
            tail_focus=lp_params_dict.get("tail_focus", "none"),
            tail_alpha=lp_params_dict.get("tail_alpha", 1.0),
            quantile_levels=lp_params_dict.get("quantile_levels", None),
        )
        mus_dict = dictionary["mus"]
        sigmas_dict = dictionary["sigmas"]
        
        basis = compute_basis_matrices(z, mus_dict, sigmas_dict)
        Phi_pdf = basis["Phi_pdf"]
        
        if objective_mode == "raw_moments":
            lp_result_raw = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=Phi_pdf,
                mus=mus_dict,
                sigmas=sigmas_dict,
                z=z,
                f=f_true,
                pdf_tolerance=lp_params_dict.get("pdf_tolerance", None),
                lambda_pdf=lp_params_dict.get("lambda_pdf", 1.0),
                lambda_raw=tuple(lp_params_dict.get("lambda_raw", [1.0, 1.0, 1.0, 1.0])),
                solver=lp_params_dict.get("solver", "highs"),
                objective_form=lp_params_dict.get("objective_form", "A"),
            )
            
            w_all = lp_result_raw["w"]
            lp_diagnostics = lp_result_raw["diagnostics"]
        else:
            raise ValueError(f"Hybrid method requires objective_mode='raw_moments', got '{objective_mode}'")
        
        lp_elapsed_time = time.time() - lp_start_time
        
        idx_top_k = np.argsort(w_all)[::-1][:K]
        pi_init = w_all[idx_top_k]
        mu_init = mus_dict[idx_top_k]
        var_init = sigmas_dict[idx_top_k]**2
        
        pi_sum = np.sum(pi_init)
        if pi_sum <= 0:
            raise ValueError("LP solution has no positive weights for top K components")
        pi_init = pi_init / pi_sum
        
        reg_var = config_dict.get("reg_var", 1e-6)
        var_init = np.maximum(var_init, reg_var)
        
        em_start_time = time.time()
        init_params_custom = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        max_iter = config_dict.get("max_iter", 400)
        tol = config_dict.get("tol", 1e-10)
        n_init = config_dict.get("n_init", 8)
        seed = config_dict.get("seed", 1)
        use_moment_matching = config_dict.get("use_moment_matching", False)
        qp_mode = config_dict.get("qp_mode", "hard")
        soft_lambda = config_dict.get("soft_lambda", 1e4)
        
        # MDN parameters (for consistency, though Hybrid uses custom init)
        mdn_params = config_dict.get("mdn_params", {})
        mdn_model_path = mdn_params.get("model_path") if mdn_params else None
        mdn_device = mdn_params.get("device", "auto") if mdn_params else "auto"
        
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
            mdn_model_path=mdn_model_path,
            mdn_device=mdn_device,
        )
        total_em_time = time.time() - em_start_time
        
        qp_elapsed_time = 0.0
        qp_info = None
        if use_moment_matching and hasattr(params, '_qp_info'):
            qp_info = params._qp_info
            qp_elapsed_time = qp_info.get('qp_time', 0.0)
        
        # Get initialization time (for hybrid, init="custom" so should be 0)
        init_elapsed_time = 0.0
        if hasattr(params, '_init_time'):
            init_elapsed_time = params._init_time
        
        # Subtract QP time from total to get pure EM time
        em_elapsed_time = total_em_time - qp_elapsed_time
        
        ll_value = ll
        
        diagnostics = {
            "lp_runtime_sec": float(lp_elapsed_time),
            "em_runtime_sec": float(em_elapsed_time),
            "qp_runtime_sec": float(qp_elapsed_time),
            "lp_diagnostics": lp_diagnostics,
            "n_dict": int(len(mus_dict)),
            "dict_J": int(dict_J),
            "dict_L": int(dict_L),
        }
    elif method == "lamf":
        # LAMF method - direct neural network inference
        lamf_start_time = time.time()
        
        try:
            params, lamf_inference_time = fit_gmm_lamf(z, f_true, K=K)
        except FileNotFoundError as e:
            raise ValueError(f"LAMF model not found: {e}")
        except ValueError as e:
            raise ValueError(f"LAMF error: {e}")
        
        lamf_elapsed_time = time.time() - lamf_start_time
        em_elapsed_time = lamf_elapsed_time  # Use em_time field for LAMF time
        lp_elapsed_time = 0.0
        qp_elapsed_time = 0.0
        init_elapsed_time = 0.0  # LAMF doesn't have separate init time
        
        n_iter = 6  # LAMF uses T=6 refinement steps
        ll_value = None  # LAMF doesn't compute log-likelihood
        diagnostics = {
            "lamf_inference_time_ms": float(lamf_inference_time * 1000),
            "T": 6,
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    total_elapsed_time = time.time() - total_start_time
    
    # Evaluate GMM PDF
    f_hat = gmm1d_pdf(z, params)
    f_hat = normalize_pdf_on_grid(z, f_hat)
    
    # Compute statistics (with error handling)
    try:
        z_stats = np.linspace(z_range[0], z_range[1], max(1000, z_npoints * 10))
        f_true_stats = np.interp(z_stats, z, f_true)
        f_hat_stats = gmm1d_pdf(z_stats, params)
        f_hat_stats = normalize_pdf_on_grid(z_stats, f_hat_stats)
        
        stats_true = compute_pdf_statistics(z_stats, f_true_stats)
        stats_hat = compute_pdf_statistics(z_stats, f_hat_stats)
    except Exception as e:
        # Fallback to default statistics if computation fails
        stats_true = {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        stats_hat = {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    
    # Compute errors
    errors = compute_errors(z, f_true, f_hat)
    
    # Prepare GMM components
    gmm_components = [
        {
            "pi": float(params.pi[k]),
            "mu": float(params.mu[k]),
            "sigma": float(np.sqrt(params.var[k]))
        }
        for k in range(len(params.pi))
    ]
    
    return {
        "method": method,
        "z": z.tolist(),
        "f_true": f_true.tolist(),
        "f_hat": f_hat.tolist(),
        "gmm_components": gmm_components,
        "statistics_true": stats_true,
        "statistics_hat": stats_hat,
        "error_metrics": errors,
        "execution_time": {
            "em_time": em_elapsed_time,
            "lp_time": lp_elapsed_time if method in ["lp", "hybrid"] else None,
            "qp_time": qp_elapsed_time if qp_elapsed_time > 0 else None,
            "init_time": init_elapsed_time if method in ["em", "hybrid"] else None,
            "total_time": total_elapsed_time
        },
        "log_likelihood": float(ll_value) if ll_value is not None else None,
        "n_iterations": int(n_iter) if n_iter is not None else None,
        "diagnostics": diagnostics,
        "params": params,
        "z_range": z_range,
    }


def generate_plot_base64(z: np.ndarray, f_true: np.ndarray, f_hat: np.ndarray,
                         params: GMM1DParams, config_dict: Dict[str, Any]) -> str:
    """Generate plot and return as base64 encoded string."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    mu_x = config_dict["mu_x"]
    sigma_x = config_dict["sigma_x"]
    mu_y = config_dict["mu_y"]
    sigma_y = config_dict["sigma_y"]
    rho = config_dict["rho"]
    ll_value = config_dict.get("log_likelihood", 0.0)
    show_grid_points = config_dict.get("show_grid_points", True)
    max_grid_points_display = config_dict.get("max_grid_points_display", 200)
    
    # Generate plot directly to memory buffer
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Import utilities
    # MIN_PDF_VALUE is now imported from gmm_fitting
    
    f_true_pos = np.maximum(f_true, MIN_PDF_VALUE)
    f_hat_pos = np.maximum(f_hat, MIN_PDF_VALUE)
    
    # Top subplot: Linear scale
    ax1.plot(z, f_true, 'b-', linewidth=2, label='True PDF', alpha=0.8)
    ax1.plot(z, f_hat, 'r--', linewidth=2, label='GMM approximation', alpha=0.8)
    
    # Prepare grid points for display (used in both subplots)
    z_display = None
    f_true_display = None
    if show_grid_points:
        if len(z) > max_grid_points_display:
            indices = np.linspace(0, len(z) - 1, max_grid_points_display, dtype=int)
            z_display = z[indices]
            f_true_display = f_true[indices]
        else:
            z_display = z
            f_true_display = f_true
        ax1.scatter(z_display, f_true_display, c='blue', s=15, alpha=0.6, 
                   marker='o', edgecolors='darkblue', linewidths=0.5, 
                   label=f'Grid points (n={len(z)})', zorder=5)
    
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('PDF Comparison (Linear Scale)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(z.min(), z.max())
    
    # Bottom subplot: Logarithmic scale
    ax2.semilogy(z, f_true_pos, 'b-', linewidth=2, label='True PDF', alpha=0.8)
    ax2.semilogy(z, f_hat_pos, 'r--', linewidth=2, label='GMM approximation', alpha=0.8)
    
    if show_grid_points and z_display is not None and f_true_display is not None:
        f_true_display_pos = np.maximum(f_true_display, MIN_PDF_VALUE)
        ax2.scatter(z_display, f_true_display_pos, c='blue', s=15, alpha=0.6,
                   marker='o', edgecolors='darkblue', linewidths=0.5,
                   label=f'Grid points (n={len(z)})', zorder=5)
    
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('Probability Density (log scale)', fontsize=12)
    ax2.set_title('PDF Comparison (Log Scale)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(z.min(), z.max())
    
    # Handle None log-likelihood (e.g., for LAMF method)
    ll_str = f"{ll_value:.6f}" if ll_value is not None else "N/A"
    fig.suptitle(
        f'PDF Comparison: μ_X={mu_x:.2f}, σ_X={sigma_x:.2f}, μ_Y={mu_y:.2f}, σ_Y={sigma_y:.2f}, ρ={rho:.2f} | Log-likelihood: {ll_str}',
        fontsize=13,
        y=0.995
    )
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return f"data:image/png;base64,{plot_base64}"


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GMM Fitting API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/load-config")
async def load_config_file(file: UploadFile = File(...)):
    """
    Load configuration from JSON file and convert to ComputeRequest format.
    """
    try:
        # Read file content
        content = await file.read()
        config = json.loads(content.decode("utf-8"))
        
        # Convert config to ComputeRequest format
        request_data = {
            "bivariate_params": {
                "mu_x": config.get("mu_x", 0.0),
                "sigma_x": config.get("sigma_x", 0.8),
                "mu_y": config.get("mu_y", 0.0),
                "sigma_y": config.get("sigma_y", 1.6),
                "rho": config.get("rho", 0.9),
            },
            "grid_params": {
                "z_range": config.get("z_range", [-6.0, 8.0]),
                "z_npoints": config.get("z_npoints", 2500),
            },
            "K": config.get("K", 5),
            "method": config.get("method", "em"),
            "show_grid_points": config.get("show_grid_points", True),
            "max_grid_points_display": config.get("max_grid_points_display", 200),
        }
        
        # Add method-specific parameters
        method = request_data["method"]
        if method == "em":
            request_data["em_params"] = {
                "max_iter": config.get("max_iter", 400),
                "tol": config.get("tol", 1e-10),
                "reg_var": config.get("reg_var", 1e-6),
                "n_init": config.get("n_init", 8),
                "seed": config.get("seed", 1),
                "init": config.get("init", "quantile"),
                "use_moment_matching": config.get("use_moment_matching", False),
                "qp_mode": config.get("qp_mode", "hard"),
                "soft_lambda": config.get("soft_lambda", 1e4),
            }
        elif method == "lp":
            lp_params = config.get("lp_params", {})
            request_data["lp_params"] = {
                "L": config.get("L", lp_params.get("L", 5)),
                "objective_mode": config.get("objective_mode", lp_params.get("objective_mode", "pdf")),
                "solver": lp_params.get("solver", "highs"),
                "sigma_min_scale": lp_params.get("sigma_min_scale", 0.1),
                "sigma_max_scale": lp_params.get("sigma_max_scale", 3.0),
                "lambda_raw": lp_params.get("lambda_raw", None),
                "objective_form": lp_params.get("objective_form", "A"),
                "pdf_tolerance": lp_params.get("pdf_tolerance", None),
            }
        elif method == "hybrid":
            lp_params = config.get("lp_params", {})
            request_data["hybrid_params"] = {
                "dict_J": lp_params.get("dict_J", None),
                "dict_L": lp_params.get("dict_L", config.get("L", 10)),
                "objective_mode": lp_params.get("objective_mode", "raw_moments"),
                "mu_mode": lp_params.get("mu_mode", "quantile"),
                "tail_focus": lp_params.get("tail_focus", "none"),
                "tail_alpha": lp_params.get("tail_alpha", 1.0),
                "sigma_min_scale": lp_params.get("sigma_min_scale", 0.1),
                "sigma_max_scale": lp_params.get("sigma_max_scale", 3.0),
                "pdf_tolerance": lp_params.get("pdf_tolerance", None),
                "lambda_raw": lp_params.get("lambda_raw", None),
            }
            # Also add EM params for hybrid
            request_data["em_params"] = {
                "max_iter": config.get("max_iter", 400),
                "tol": config.get("tol", 1e-10),
                "reg_var": config.get("reg_var", 1e-6),
                "n_init": config.get("n_init", 8),
                "seed": config.get("seed", 1),
                "use_moment_matching": config.get("use_moment_matching", False),
                "qp_mode": config.get("qp_mode", "hard"),
                "soft_lambda": config.get("soft_lambda", 1e4),
            }
        
        return request_data
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {str(e)}")


@app.post("/api/compute", response_model=ComputeResponse)
async def compute_gmm(request: ComputeRequest):
    """
    Main computation endpoint for GMM fitting.
    """
    try:
        # Convert request to config dictionary
        config_dict = {
            "mu_x": request.bivariate_params.mu_x,
            "sigma_x": request.bivariate_params.sigma_x,
            "mu_y": request.bivariate_params.mu_y,
            "sigma_y": request.bivariate_params.sigma_y,
            "rho": request.bivariate_params.rho,
            "z_range": request.grid_params.z_range,
            "z_npoints": request.grid_params.z_npoints,
            "K": request.K,
            "method": request.method,
            "show_grid_points": request.show_grid_points,
            "max_grid_points_display": request.max_grid_points_display,
            "output_path": "temp_plot",  # Temporary file
        }
        
        # Add method-specific parameters
        if request.method == "em" and request.em_params:
            config_dict.update({
                "max_iter": request.em_params.max_iter,
                "tol": request.em_params.tol,
                "reg_var": request.em_params.reg_var,
                "n_init": request.em_params.n_init,
                "seed": request.em_params.seed,
                "init": request.em_params.init,
                "use_moment_matching": request.em_params.use_moment_matching,
                "qp_mode": request.em_params.qp_mode,
                "soft_lambda": request.em_params.soft_lambda,
            })
            # Add MDN parameters if provided
            if request.em_params.mdn_params:
                config_dict["mdn_params"] = {
                    "model_path": request.em_params.mdn_params.model_path,
                    "device": request.em_params.mdn_params.device,
                }
            # Add LAMF parameters if provided
            if request.em_params.lamf_params:
                config_dict["lamf_params"] = {
                    "model_path": request.em_params.lamf_params.model_path,
                    "device": request.em_params.lamf_params.device,
                }
        elif request.method == "lp" and request.lp_params:
            config_dict.update({
                "L": request.lp_params.L,
                "objective_mode": request.lp_params.objective_mode,
                "solver": request.lp_params.solver,
                "sigma_min_scale": request.lp_params.sigma_min_scale,
                "sigma_max_scale": request.lp_params.sigma_max_scale,
            })
            # Set optional parameters at top level (for compute_gmm_fitting compatibility)
            # For raw_moments mode, set defaults if not provided
            if request.lp_params.objective_mode == "raw_moments":
                config_dict["objective_form"] = request.lp_params.objective_form or "A"
                # lambda_raw defaults to [1.0, 1.0, 1.0, 1.0] if not provided
                config_dict["lambda_raw"] = request.lp_params.lambda_raw or [1.0, 1.0, 1.0, 1.0]
            else:
                if request.lp_params.lambda_raw:
                    config_dict["lambda_raw"] = request.lp_params.lambda_raw
                if request.lp_params.objective_form:
                    config_dict["objective_form"] = request.lp_params.objective_form
            if request.lp_params.pdf_tolerance:
                config_dict["pdf_tolerance"] = request.lp_params.pdf_tolerance
            # Also set lp_params dict (for compatibility with main.py style configs)
            config_dict["lp_params"] = {
                "solver": request.lp_params.solver,
                "sigma_min_scale": request.lp_params.sigma_min_scale,
                "sigma_max_scale": request.lp_params.sigma_max_scale,
            }
            # Set optional parameters in lp_params
            if request.lp_params.objective_mode == "raw_moments":
                # For raw_moments mode, set defaults if not provided
                config_dict["lp_params"]["objective_form"] = request.lp_params.objective_form or "A"
                config_dict["lp_params"]["lambda_raw"] = request.lp_params.lambda_raw or [1.0, 1.0, 1.0, 1.0]
            else:
                if request.lp_params.lambda_raw:
                    config_dict["lp_params"]["lambda_raw"] = request.lp_params.lambda_raw
                if request.lp_params.objective_form:
                    config_dict["lp_params"]["objective_form"] = request.lp_params.objective_form
            if request.lp_params.pdf_tolerance:
                config_dict["lp_params"]["pdf_tolerance"] = request.lp_params.pdf_tolerance
        elif request.method == "hybrid" and request.hybrid_params:
            config_dict["lp_params"] = {
                "dict_J": request.hybrid_params.dict_J or (4 * request.K),
                "dict_L": request.hybrid_params.dict_L,
                "objective_mode": request.hybrid_params.objective_mode,
                "mu_mode": request.hybrid_params.mu_mode,
                "tail_focus": request.hybrid_params.tail_focus,
                "tail_alpha": request.hybrid_params.tail_alpha,
                "sigma_min_scale": request.hybrid_params.sigma_min_scale,
                "sigma_max_scale": request.hybrid_params.sigma_max_scale,
            }
            if request.hybrid_params.pdf_tolerance:
                config_dict["lp_params"]["pdf_tolerance"] = request.hybrid_params.pdf_tolerance
            if request.hybrid_params.lambda_raw:
                config_dict["lp_params"]["lambda_raw"] = request.hybrid_params.lambda_raw
            # Add EM params for hybrid
            if request.em_params:
                config_dict.update({
                    "max_iter": request.em_params.max_iter,
                    "tol": request.em_params.tol,
                    "reg_var": request.em_params.reg_var,
                    "n_init": request.em_params.n_init,
                    "seed": request.em_params.seed,
                    "use_moment_matching": request.em_params.use_moment_matching,
                    "qp_mode": request.em_params.qp_mode,
                    "soft_lambda": request.em_params.soft_lambda,
                })
        
        # Perform computation
        result = compute_gmm_fitting(config_dict)
        
        # Generate plot
        z_array = np.array(result["z"])
        f_true_array = np.array(result["f_true"])
        f_hat_array = np.array(result["f_hat"])
        config_dict["log_likelihood"] = result["log_likelihood"]
        
        plot_data_url = generate_plot_base64(
            z_array, f_true_array, f_hat_array,
            result["params"], config_dict
        )
        
        # Ensure statistics are always present (with defaults if missing)
        stats_true = result.get("statistics_true", {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0})
        stats_hat = result.get("statistics_hat", {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0})
        
        # Build response
        response = ComputeResponse(
            success=True,
            method=result["method"],
            z=result["z"],
            f_true=result["f_true"],
            f_hat=result["f_hat"],
            gmm_components=[
                GMMComponent(**comp) for comp in result["gmm_components"]
            ],
            statistics_true=Statistics(**stats_true),
            statistics_hat=Statistics(**stats_hat),
            error_metrics=ErrorMetrics(**result["error_metrics"]),
            execution_time=ExecutionTime(**result["execution_time"]),
            log_likelihood=result["log_likelihood"],
            n_iterations=result["n_iterations"],
            diagnostics=result["diagnostics"],
            plot_data_url=plot_data_url,
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

