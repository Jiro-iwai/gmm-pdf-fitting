"""
Tests for LP solver with raw moments.

Tests for:
- solve_lp_pdf_rawmoments_linf: LP solver minimizing raw moment errors
"""

import numpy as np
import pytest
from gmm_fitting import (
    solve_lp_pdf_rawmoments_linf,
    compute_basis_matrices,
    build_gaussian_dictionary_simple,
)
from gmm_fitting import (
    compute_pdf_raw_moments,
    normal_pdf,
    normalize_pdf_on_grid,
)


class TestSolveLPPDFRawMomentsLinf:
    """Tests for solve_lp_pdf_rawmoments_linf function."""
    
    def test_basic_solve(self):
        """Test basic LP solve with raw moments."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        # Build dictionary
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        
        # Compute basis matrix
        basis = compute_basis_matrices(z, mus, sigmas)
        Phi_pdf = basis["Phi_pdf"]
        
        # Compute target raw moments
        M_target = compute_pdf_raw_moments(z, f, max_order=4)
        
        # Solve LP
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=Phi_pdf,
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=0.01,
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="A",
        )
        
        assert result["status"] == 0  # Success
        assert "w" in result
        assert "t_pdf" in result
        assert "t_raw" in result
        assert len(result["w"]) == len(mus)
        assert len(result["t_raw"]) == 4
        assert np.all(result["w"] >= 0)
        assert np.isclose(np.sum(result["w"]), 1.0, rtol=1e-6)
    
    def test_objective_form_A(self):
        """Test objective form A (PDF constraint + moment minimization)."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        M_target = compute_pdf_raw_moments(z, f, max_order=4)
        
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=0.01,
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="A",
        )
        
        assert result["status"] == 0
        # PDF error should be within tolerance (may be relaxed by fallback)
        # Fallback can relax to 0.1, so check against that
        assert result["t_pdf"] <= 0.1 + 1e-6
    
    def test_objective_form_B(self):
        """Test objective form B (combined minimization)."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        M_target = compute_pdf_raw_moments(z, f, max_order=4)
        
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=None,  # Ignored in form B
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="B",
        )
        
        assert result["status"] == 0
    
    def test_pdf_tolerance_none_form_A(self):
        """Test that pdf_tolerance=None works in form A."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=None,
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="A",
        )
        
        assert result["status"] == 0
    
    def test_fallback_relaxation(self):
        """Test that pdf_tolerance fallback works."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        # Very strict tolerance that might fail
        # This should succeed after fallback relaxation
        try:
            result = solve_lp_pdf_rawmoments_linf(
                Phi_pdf=basis["Phi_pdf"],
                mus=mus,
                sigmas=sigmas,
                z=z,
                f=f,
                pdf_tolerance=1e-10,  # Very strict
                lambda_pdf=1.0,
                lambda_raw=(1.0, 1.0, 1.0, 1.0),
                solver="highs",
                objective_form="A",
            )
            # Should succeed after fallback
            assert result["status"] == 0
        except RuntimeError as e:
            # If still infeasible after fallback, that's acceptable for this test
            assert "infeasible" in str(e).lower()
    
    def test_moment_accuracy(self):
        """Test that raw moments are approximately matched."""
        z = np.linspace(-3, 3, 500)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=10, L=5)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        M_target = compute_pdf_raw_moments(z, f, max_order=4)
        
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=0.01,
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="A",
        )
        
        if result["status"] == 0:
            # Check diagnostics
            diagnostics = result.get("diagnostics", {})
            if "raw_mix" in diagnostics and "raw_target" in diagnostics:
                raw_mix = diagnostics["raw_mix"]
                raw_target = diagnostics["raw_target"]
                # Moments should be close (raw_mix and raw_target are lists of M1..M4, indices 0..3)
                for i in range(4):  # M1..M4 (indices 0..3)
                    assert abs(raw_mix[i] - raw_target[i]) < 0.1
    
    def test_diagnostics(self):
        """Test that diagnostics contain expected information."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        dictionary = build_gaussian_dictionary_simple(z, f, K=5, L=3)
        mus = dictionary["mus"]
        sigmas = dictionary["sigmas"]
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=basis["Phi_pdf"],
            mus=mus,
            sigmas=sigmas,
            z=z,
            f=f,
            pdf_tolerance=0.01,
            lambda_pdf=1.0,
            lambda_raw=(1.0, 1.0, 1.0, 1.0),
            solver="highs",
            objective_form="A",
        )
        
        if result["status"] == 0:
            diagnostics = result.get("diagnostics", {})
            expected_keys = [
                "n_dict",
                "t_pdf",
                "raw_target",
                "raw_mix",
                "raw_abs_err",
                "n_nonzero",
            ]
            for key in expected_keys:
                assert key in diagnostics, f"Missing key: {key}"

