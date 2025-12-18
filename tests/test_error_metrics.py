"""
Tests for error metrics computation functions.

Tests for:
- compute_errors: PDF/CDF L∞ errors, quantile errors, tail-weighted L1 error
"""

import numpy as np
import pytest
from gmm_fitting import (
    compute_errors,
    normal_pdf,
    pdf_to_cdf_trapz,
    normalize_pdf_on_grid,
)


class TestComputeErrors:
    """Tests for compute_errors function."""
    
    def test_identical_pdfs(self):
        """Test that identical PDFs produce zero errors."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        f_hat = f_true.copy()
        
        errors = compute_errors(z, f_true, f_hat)
        
        assert errors["linf_pdf"] < 1e-10
        assert errors["linf_cdf"] < 1e-10
        assert all(abs(err) < 1e-10 for err in errors["quantile_abs_errors"].values())
        assert errors["tail_l1_error"] < 1e-10
    
    def test_pdf_linf_error(self):
        """Test PDF L∞ error calculation."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Shifted PDF
        f_hat = normal_pdf(z, 0.1, 1.0)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        errors = compute_errors(z, f_true, f_hat)
        
        assert errors["linf_pdf"] > 0
        assert isinstance(errors["linf_pdf"], float)
    
    def test_cdf_linf_error(self):
        """Test CDF L∞ error calculation."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Shifted PDF
        f_hat = normal_pdf(z, 0.2, 1.0)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        errors = compute_errors(z, f_true, f_hat)
        
        assert errors["linf_cdf"] > 0
        assert isinstance(errors["linf_cdf"], float)
        # CDF error should be bounded
        assert errors["linf_cdf"] <= 1.0
    
    def test_quantile_errors(self):
        """Test quantile error calculation."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Shifted PDF
        f_hat = normal_pdf(z, 0.1, 1.0)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        quantile_ps = [0.9, 0.99, 0.999]
        errors = compute_errors(z, f_true, f_hat, quantile_ps=quantile_ps)
        
        assert "quantiles_true" in errors
        assert "quantiles_hat" in errors
        assert "quantile_abs_errors" in errors
        
        for p in quantile_ps:
            assert p in errors["quantiles_true"]
            assert p in errors["quantiles_hat"]
            assert p in errors["quantile_abs_errors"]
            assert errors["quantile_abs_errors"][p] >= 0
    
    def test_quantile_calculation(self):
        """Test that quantiles are calculated correctly."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        f_hat = f_true.copy()
        
        errors = compute_errors(z, f_true, f_hat, quantile_ps=[0.5])
        
        # For standard normal, median (0.5 quantile) should be near 0
        assert abs(errors["quantiles_true"][0.5]) < 0.1
        assert abs(errors["quantiles_hat"][0.5]) < 0.1
    
    def test_tail_l1_error(self):
        """Test tail-weighted L1 error calculation."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # PDF with different tail behavior
        f_hat = normal_pdf(z, 0.0, 1.5)  # Wider distribution
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        errors = compute_errors(z, f_true, f_hat, tail_weight_p0=0.9)
        
        assert errors["tail_l1_error"] >= 0
        assert isinstance(errors["tail_l1_error"], float)
    
    def test_tail_l1_error_custom_p0(self):
        """Test tail L1 error with custom p0."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        f_hat = normal_pdf(z, 0.0, 1.2)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        errors_p09 = compute_errors(z, f_true, f_hat, tail_weight_p0=0.9)
        errors_p095 = compute_errors(z, f_true, f_hat, tail_weight_p0=0.95)
        
        # Different p0 should give different tail errors
        assert errors_p09["tail_l1_error"] != errors_p095["tail_l1_error"]
    
    def test_cdf_monotonicity(self):
        """Test that CDF monotonicity is enforced."""
        z = np.linspace(-3, 3, 100)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Create f_hat with potential non-monotonic CDF due to numerical errors
        f_hat = f_true + 1e-10 * np.random.randn(100)
        f_hat = np.maximum(f_hat, 0)
        f_hat = normalize_pdf_on_grid(z, f_hat)
        
        # Should not raise error
        errors = compute_errors(z, f_true, f_hat)
        
        assert "linf_cdf" in errors
        assert errors["linf_cdf"] >= 0
    
    def test_custom_quantile_levels(self):
        """Test with custom quantile probability levels."""
        z = np.linspace(-3, 3, 1000)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        f_hat = f_true.copy()
        
        custom_ps = [0.5, 0.75, 0.9, 0.95]
        errors = compute_errors(z, f_true, f_hat, quantile_ps=custom_ps)
        
        for p in custom_ps:
            assert p in errors["quantiles_true"]
            assert p in errors["quantiles_hat"]
            assert p in errors["quantile_abs_errors"]
    
    def test_return_type(self):
        """Test that return value is a dictionary with expected keys."""
        z = np.linspace(-3, 3, 100)
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        f_hat = f_true.copy()
        
        errors = compute_errors(z, f_true, f_hat)
        
        assert isinstance(errors, dict)
        required_keys = [
            "linf_pdf",
            "linf_cdf",
            "quantiles_true",
            "quantiles_hat",
            "quantile_abs_errors",
            "tail_l1_error",
        ]
        for key in required_keys:
            assert key in errors

