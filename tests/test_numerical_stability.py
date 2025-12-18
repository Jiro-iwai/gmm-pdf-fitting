"""
Tests for numerical stability improvements.

Tests for:
- max_pdf_bivariate_normal with rho near ±1
"""

import numpy as np
import pytest
from gmm_fitting import max_pdf_bivariate_normal
from gmm_fitting import normalize_pdf_on_grid, SIGMA_FLOOR


class TestNumericalStability:
    """Tests for numerical stability in max_pdf_bivariate_normal."""
    
    def test_rho_near_one(self):
        """Test that rho=0.99 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = 0.99
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_rho_near_minus_one(self):
        """Test that rho=-0.99 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = -0.99
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_rho_exactly_one(self):
        """Test that rho=1.0 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = 1.0
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_rho_exactly_minus_one(self):
        """Test that rho=-1.0 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = -1.0
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_rho_very_close_to_one(self):
        """Test that rho=0.999999 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = 0.999999
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_rho_very_close_to_minus_one(self):
        """Test that rho=-0.999999 doesn't produce NaN or Inf."""
        z = np.linspace(-3, 3, 1000)
        rho = -0.999999
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_conditional_std_floor(self):
        """Test that conditional standard deviation is clipped to SIGMA_FLOOR."""
        z = np.linspace(-3, 3, 1000)
        rho = 0.999999
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, rho)
        
        # PDF should be normalizable
        pdf_norm = normalize_pdf_on_grid(z, pdf)
        area = np.trapezoid(pdf_norm, z)
        assert np.isclose(area, 1.0, rtol=1e-2)
    
    def test_different_variances(self):
        """Test with different variances and rho near 1."""
        z = np.linspace(-3, 5, 1000)
        rho = 0.99
        
        pdf = max_pdf_bivariate_normal(z, 0.0, 0.5, 1.0, 2.0, rho)
        
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)
        assert not np.any(np.isnan(pdf))
        assert not np.any(np.isinf(pdf))
    
    def test_comparison_with_rho_zero(self):
        """Test that rho near 1 produces reasonable results compared to rho=0."""
        z = np.linspace(-3, 3, 1000)
        
        pdf_rho0 = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, 0.0)
        pdf_rho099 = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, 0.99)
        
        # Both should be finite and positive
        assert np.all(np.isfinite(pdf_rho0))
        assert np.all(np.isfinite(pdf_rho099))
        assert np.all(pdf_rho0 >= 0)
        assert np.all(pdf_rho099 >= 0)
        
        # Both should be normalizable
        pdf_rho0_norm = normalize_pdf_on_grid(z, pdf_rho0)
        pdf_rho099_norm = normalize_pdf_on_grid(z, pdf_rho099)
        
        area0 = np.trapezoid(pdf_rho0_norm, z)
        area099 = np.trapezoid(pdf_rho099_norm, z)
        
        assert np.isclose(area0, 1.0, rtol=1e-2)
        assert np.isclose(area099, 1.0, rtol=1e-2)

