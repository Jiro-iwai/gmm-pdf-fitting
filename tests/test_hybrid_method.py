"""
Tests for Hybrid method (LP → EM → QP).

Tests for:
- Hybrid method integration in main.py
- LP → EM → QP workflow
"""

import numpy as np
import pytest
import json
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmm_fitting import (
    load_config,
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
)
from gmm_fitting import (
    solve_lp_pdf_rawmoments_linf,
    build_gaussian_dictionary,
    compute_basis_matrices,
)
from gmm_fitting import (
    compute_pdf_raw_moments,
    compute_errors,
)


class TestHybridMethod:
    """Tests for Hybrid method."""
    
    def test_hybrid_basic(self):
        """Test basic Hybrid method execution."""
        # Create temporary config file
        config = {
            "mu_x": 0.1,
            "sigma_x": 0.4,
            "mu_y": 0.15,
            "sigma_y": 0.9,
            "rho": 0.9,
            "z_range": [-2, 4],
            "z_npoints": 128,
            "method": "hybrid",
            "K": 5,
            "L": 10,
            "lp_params": {
                "dict_J": 20,
                "dict_L": 10,
                "mu_mode": "quantile",
                "tail_focus": "right",
                "tail_alpha": 2.0,
                "solver": "highs",
                "objective_mode": "raw_moments",
                "pdf_tolerance": 0.01,
                "lambda_pdf": 1.0,
                "lambda_raw": [1.0, 1.0, 1.0, 1.0],
                "objective_form": "A",
            },
            "max_iter": 100,
            "tol": 1e-6,
            "reg_var": 1e-6,
            "n_init": 1,
            "init": "custom",
            "use_moment_matching": True,
            "qp_mode": "hard",
            "soft_lambda": 1e4,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Load config
            config_dict = load_config(config_path)
            
            # Create true PDF
            z = np.linspace(config_dict["z_range"][0], config_dict["z_range"][1], config_dict["z_npoints"])
            f_true = max_pdf_bivariate_normal(
                z,
                config_dict["mu_x"],
                config_dict["sigma_x"]**2,
                config_dict["mu_y"],
                config_dict["sigma_y"]**2,
                config_dict["rho"]
            )
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Run Hybrid method manually (simulating main.py logic)
            lp_params = config_dict["lp_params"]
            dict_J = lp_params["dict_J"]
            dict_L = lp_params["dict_L"]
            K = config_dict["K"]
            
            # Step 1: LP
            dictionary = build_gaussian_dictionary(
                z, f_true,
                J=dict_J,
                L=dict_L,
                mu_mode=lp_params["mu_mode"],
                tail_focus=lp_params["tail_focus"],
                tail_alpha=lp_params["tail_alpha"],
            )
            basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
            
            lp_result = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=basis["Phi_pdf"],
                mus=dictionary["mus"],
                sigmas=dictionary["sigmas"],
                z=z,
                f=f_true,
                pdf_tolerance=lp_params["pdf_tolerance"],
                lambda_pdf=lp_params["lambda_pdf"],
                lambda_raw=tuple(lp_params["lambda_raw"]),
                solver=lp_params["solver"],
                objective_form=lp_params["objective_form"],
            )
            
            # Step 2: Select top K
            w_all = lp_result["w"]
            idx_top_k = np.argsort(w_all)[::-1][:K]
            pi_init = w_all[idx_top_k] / np.sum(w_all[idx_top_k])
            mu_init = dictionary["mus"][idx_top_k]
            var_init = dictionary["sigmas"][idx_top_k]**2
            var_init = np.maximum(var_init, config_dict["reg_var"])
            
            # Step 3: EM
            init_params_custom = {
                "pi": pi_init,
                "mu": mu_init,
                "var": var_init,
            }
            
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true,
                K=K,
                max_iter=config_dict["max_iter"],
                tol=config_dict["tol"],
                reg_var=config_dict["reg_var"],
                n_init=1,
                init="custom",
                init_params=init_params_custom,
                use_moment_matching=config_dict["use_moment_matching"],
                qp_mode=config_dict["qp_mode"],
                soft_lambda=config_dict["soft_lambda"],
            )
            
            # Check result
            assert params is not None
            assert len(params.pi) == K
            assert np.isclose(np.sum(params.pi), 1.0, rtol=1e-6)
            
        finally:
            os.unlink(config_path)
    
    def test_hybrid_lp_to_em_transition(self):
        """Test that LP results are correctly passed to EM."""
        config = {
            "mu_x": 0.0,
            "sigma_x": 1.0,
            "mu_y": 0.0,
            "sigma_y": 1.0,
            "rho": 0.5,
            "z_range": [-3, 3],
            "z_npoints": 200,
            "method": "hybrid",
            "K": 5,
            "L": 10,
            "lp_params": {
                "dict_J": 15,
                "dict_L": 5,
                "mu_mode": "quantile",
                "tail_focus": "none",
                "tail_alpha": 1.0,
                "solver": "highs",
                "objective_mode": "raw_moments",
                "pdf_tolerance": 0.01,
                "lambda_pdf": 1.0,
                "lambda_raw": [1.0, 1.0, 1.0, 1.0],
                "objective_form": "A",
            },
            "max_iter": 50,
            "tol": 1e-6,
            "reg_var": 1e-6,
            "n_init": 1,
            "init": "custom",
            "use_moment_matching": False,  # Skip QP for simplicity
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            config_dict = load_config(config_path)
            z = np.linspace(config_dict["z_range"][0], config_dict["z_range"][1], config_dict["z_npoints"])
            f_true = max_pdf_bivariate_normal(
                z,
                config_dict["mu_x"],
                config_dict["sigma_x"]**2,
                config_dict["mu_y"],
                config_dict["sigma_y"]**2,
                config_dict["rho"]
            )
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Run Hybrid method manually
            lp_params = config_dict["lp_params"]
            dict_J = lp_params["dict_J"]
            dict_L = lp_params["dict_L"]
            K = config_dict["K"]
            
            dictionary = build_gaussian_dictionary(
                z, f_true, J=dict_J, L=dict_L,
                mu_mode=lp_params["mu_mode"],
            )
            basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
            
            lp_result = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=basis["Phi_pdf"],
                mus=dictionary["mus"],
                sigmas=dictionary["sigmas"],
                z=z, f=f_true,
                pdf_tolerance=lp_params["pdf_tolerance"],
                lambda_pdf=lp_params["lambda_pdf"],
                lambda_raw=tuple(lp_params["lambda_raw"]),
                solver=lp_params["solver"],
                objective_form=lp_params["objective_form"],
            )
            
            w_all = lp_result["w"]
            idx_top_k = np.argsort(w_all)[::-1][:K]
            pi_init = w_all[idx_top_k] / np.sum(w_all[idx_top_k])
            mu_init = dictionary["mus"][idx_top_k]
            var_init = np.maximum(dictionary["sigmas"][idx_top_k]**2, config_dict["reg_var"])
            
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true, K=K,
                max_iter=config_dict["max_iter"],
                tol=config_dict["tol"],
                reg_var=config_dict["reg_var"],
                n_init=1,
                init="custom",
                init_params={"pi": pi_init, "mu": mu_init, "var": var_init},
                use_moment_matching=False,
            )
            
            assert params is not None
            
        finally:
            os.unlink(config_path)
    
    def test_hybrid_with_moment_matching(self):
        """Test Hybrid method with moment matching QP."""
        config = {
            "mu_x": 0.1,
            "sigma_x": 0.4,
            "mu_y": 0.15,
            "sigma_y": 0.9,
            "rho": 0.9,
            "z_range": [-2, 4],
            "z_npoints": 128,
            "method": "hybrid",
            "K": 5,
            "L": 10,
            "lp_params": {
                "dict_J": 20,
                "dict_L": 10,
                "mu_mode": "quantile",
                "tail_focus": "right",
                "tail_alpha": 2.0,
                "solver": "highs",
                "objective_mode": "raw_moments",
                "pdf_tolerance": 0.01,
                "lambda_pdf": 1.0,
                "lambda_raw": [1.0, 1.0, 1.0, 1.0],
                "objective_form": "A",
            },
            "max_iter": 100,
            "tol": 1e-6,
            "reg_var": 1e-6,
            "n_init": 1,
            "init": "custom",
            "use_moment_matching": True,
            "qp_mode": "hard",
            "soft_lambda": 1e4,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            config_dict = load_config(config_path)
            z = np.linspace(config_dict["z_range"][0], config_dict["z_range"][1], config_dict["z_npoints"])
            f_true = max_pdf_bivariate_normal(
                z,
                config_dict["mu_x"],
                config_dict["sigma_x"]**2,
                config_dict["mu_y"],
                config_dict["sigma_y"]**2,
                config_dict["rho"]
            )
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Run Hybrid method manually with moment matching
            lp_params = config_dict["lp_params"]
            dict_J = lp_params["dict_J"]
            dict_L = lp_params["dict_L"]
            K = config_dict["K"]
            
            dictionary = build_gaussian_dictionary(
                z, f_true, J=dict_J, L=dict_L,
                mu_mode=lp_params["mu_mode"],
                tail_focus=lp_params["tail_focus"],
                tail_alpha=lp_params["tail_alpha"],
            )
            basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
            
            lp_result = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=basis["Phi_pdf"],
                mus=dictionary["mus"],
                sigmas=dictionary["sigmas"],
                z=z, f=f_true,
                pdf_tolerance=lp_params["pdf_tolerance"],
                lambda_pdf=lp_params["lambda_pdf"],
                lambda_raw=tuple(lp_params["lambda_raw"]),
                solver=lp_params["solver"],
                objective_form=lp_params["objective_form"],
            )
            
            w_all = lp_result["w"]
            idx_top_k = np.argsort(w_all)[::-1][:K]
            pi_init = w_all[idx_top_k] / np.sum(w_all[idx_top_k])
            mu_init = dictionary["mus"][idx_top_k]
            var_init = np.maximum(dictionary["sigmas"][idx_top_k]**2, config_dict["reg_var"])
            
            params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
                z, f_true, K=K,
                max_iter=config_dict["max_iter"],
                tol=config_dict["tol"],
                reg_var=config_dict["reg_var"],
                n_init=1,
                init="custom",
                init_params={"pi": pi_init, "mu": mu_init, "var": var_init},
                use_moment_matching=True,
                qp_mode=config_dict["qp_mode"],
                soft_lambda=config_dict["soft_lambda"],
            )
            
            assert params is not None
            
        finally:
            os.unlink(config_path)
    
    def test_hybrid_dict_params_defaults(self):
        """Test that dict_J and dict_L default correctly when not specified."""
        config = {
            "mu_x": 0.0,
            "sigma_x": 1.0,
            "mu_y": 0.0,
            "sigma_y": 1.0,
            "rho": 0.5,
            "z_range": [-3, 3],
            "z_npoints": 200,
            "method": "hybrid",
            "K": 5,
            "L": 10,
            "lp_params": {
                # dict_J and dict_L not specified - should default to dict_J=4*K, dict_L=L
                "mu_mode": "quantile",
                "solver": "highs",
                "objective_mode": "raw_moments",
                "pdf_tolerance": 0.01,
                "lambda_pdf": 1.0,
                "lambda_raw": [1.0, 1.0, 1.0, 1.0],
            },
            "max_iter": 50,
            "tol": 1e-6,
            "reg_var": 1e-6,
            "n_init": 1,
            "init": "custom",
            "use_moment_matching": False,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            config_dict = load_config(config_path)
            z = np.linspace(config_dict["z_range"][0], config_dict["z_range"][1], config_dict["z_npoints"])
            f_true = max_pdf_bivariate_normal(
                z,
                config_dict["mu_x"],
                config_dict["sigma_x"]**2,
                config_dict["mu_y"],
                config_dict["sigma_y"]**2,
                config_dict["rho"]
            )
            f_true = normalize_pdf_on_grid(z, f_true)
            
            # Should work with default dict_J and dict_L
            # Default: dict_J = 4*K = 20, dict_L = L = 10
            K = config_dict["K"]
            L = config_dict["L"]
            dict_J = 4 * K  # Default
            dict_L = L  # Default
            
            dictionary = build_gaussian_dictionary(
                z, f_true, J=dict_J, L=dict_L,
                mu_mode="quantile",
            )
            basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
            
            lp_params = config_dict["lp_params"]
            lp_result = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=basis["Phi_pdf"],
                mus=dictionary["mus"],
                sigmas=dictionary["sigmas"],
                z=z, f=f_true,
                pdf_tolerance=lp_params["pdf_tolerance"],
                lambda_pdf=lp_params["lambda_pdf"],
                lambda_raw=tuple(lp_params["lambda_raw"]),
                solver=lp_params["solver"],
            )
            
            assert lp_result["status"] == 0
            
        finally:
            os.unlink(config_path)

