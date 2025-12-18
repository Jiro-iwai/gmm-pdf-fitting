"""
Tests for PDF calculation functions.
"""
import numpy as np
import pytest
from gmm_fitting import (
    max_pdf_bivariate_normal,
    max_pdf_bivariate_normal_decomposed,
)
from gmm_fitting import (
    normal_pdf,
    normalize_pdf_on_grid
)


class TestNormalPDF:
    """Tests for normal_pdf function."""
    
    def test_normal_pdf_basic(self):
        """Test basic normal PDF calculation."""
        x = np.array([0.0, 1.0, -1.0])
        mu = 0.0
        var = 1.0
        result = normal_pdf(x, mu, var)
        
        assert len(result) == 3
        assert np.all(result > 0)
        # At x=0, PDF should be 1/sqrt(2*pi) ≈ 0.3989
        assert np.isclose(result[0], 1.0 / np.sqrt(2.0 * np.pi), rtol=1e-5)
    
    def test_normal_pdf_shifted(self):
        """Test normal PDF with non-zero mean."""
        x = np.array([1.0])
        mu = 1.0
        var = 1.0
        result = normal_pdf(x, mu, var)
        
        # At mean, PDF should be 1/sqrt(2*pi)
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        assert np.isclose(result[0], expected, rtol=1e-5)
    
    def test_normal_pdf_variance(self):
        """Test normal PDF with different variances."""
        x = np.array([0.0])
        mu = 0.0
        var_small = 0.25
        var_large = 4.0
        
        pdf_small = normal_pdf(x, mu, var_small)
        pdf_large = normal_pdf(x, mu, var_large)
        
        # Smaller variance should give higher PDF at mean
        assert pdf_small > pdf_large
    
    def test_normal_pdf_array(self):
        """Test normal PDF with array input."""
        x = np.linspace(-3, 3, 100)
        mu = 0.0
        var = 1.0
        result = normal_pdf(x, mu, var)
        
        assert len(result) == 100
        assert np.all(result > 0)
        # Check that PDF values are reasonable
        # Maximum should be at mean (x=0)
        mid = len(x) // 2
        assert result[mid] > result[0]  # Center > edge
        assert result[mid] > result[-1]  # Center > edge
        # Check that PDF decreases away from mean
        assert result[mid] > result[mid - 20]  # Center > left
        assert result[mid] > result[mid + 20]  # Center > right


class TestMaxPDFBivariateNormal:
    """Tests for max_pdf_bivariate_normal function."""
    
    def test_max_pdf_independent(self):
        """Test max PDF for independent variables (rho=0)."""
        z = np.linspace(-2, 2, 100)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 0.0
        var_y = 1.0
        rho = 0.0
        
        result = max_pdf_bivariate_normal(z, mu_x, var_x, mu_y, var_y, rho)
        
        assert len(result) == 100
        assert np.all(result >= 0)
    
    def test_max_pdf_correlated(self):
        """Test max PDF for correlated variables."""
        z = np.linspace(-2, 2, 100)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 0.0
        var_y = 1.0
        rho = 0.9
        
        result = max_pdf_bivariate_normal(z, mu_x, var_x, mu_y, var_y, rho)
        
        assert len(result) == 100
        assert np.all(result >= 0)
    
    def test_max_pdf_different_means(self):
        """Test max PDF with different means."""
        z = np.linspace(-2, 4, 100)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 1.0
        var_y = 1.0
        rho = 0.5
        
        result = max_pdf_bivariate_normal(z, mu_x, var_x, mu_y, var_y, rho)
        
        assert len(result) == 100
        assert np.all(result >= 0)
        # PDF should be higher for z > max(mu_x, mu_y)
        assert result[-1] > result[0]
    
    def test_max_pdf_normalization(self):
        """Test that max PDF integrates to approximately 1 after normalization."""
        z = np.linspace(-5, 5, 1000)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 0.0
        var_y = 1.0
        rho = 0.5
        
        pdf = max_pdf_bivariate_normal(z, mu_x, var_x, mu_y, var_y, rho)
        pdf_norm = normalize_pdf_on_grid(z, pdf)
        
        # Check normalization
        area = np.trapezoid(pdf_norm, z)
        assert np.isclose(area, 1.0, rtol=1e-3)


class TestMaxPDFDecomposed:
    """Tests for max_pdf_bivariate_normal_decomposed function."""
    
    def test_decomposed_basic(self):
        """Test decomposed PDF calculation."""
        z = np.linspace(-2, 2, 100)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 0.0
        var_y = 1.0
        rho = 0.5
        
        gx, gy = max_pdf_bivariate_normal_decomposed(z, mu_x, var_x, mu_y, var_y, rho)
        
        assert len(gx) == 100
        assert len(gy) == 100
        assert np.all(gx >= 0)
        assert np.all(gy >= 0)
        
        # Sum should equal max PDF
        pdf_max = max_pdf_bivariate_normal(z, mu_x, var_x, mu_y, var_y, rho)
        assert np.allclose(gx + gy, pdf_max, rtol=1e-5)
    
    def test_decomposed_symmetry(self):
        """Test symmetry when X and Y have same parameters."""
        z = np.linspace(-2, 2, 100)
        mu_x = 0.0
        var_x = 1.0
        mu_y = 0.0
        var_y = 1.0
        rho = 0.0
        
        gx, gy = max_pdf_bivariate_normal_decomposed(z, mu_x, var_x, mu_y, var_y, rho)
        
        # For symmetric case with rho=0, gx and gy should be equal
        assert np.allclose(gx, gy, rtol=1e-5)


class TestNormalizePDF:
    """Tests for normalize_pdf_on_grid function."""
    
    def test_normalize_basic(self):
        """Test basic PDF normalization."""
        z = np.linspace(0, 1, 100)
        f = np.ones(100)  # Uniform PDF
        
        f_norm = normalize_pdf_on_grid(z, f)
        
        # Check normalization
        area = np.trapezoid(f_norm, z)
        assert np.isclose(area, 1.0, rtol=1e-5)
    
    def test_normalize_normal_distribution(self):
        """Test normalization of normal distribution."""
        z = np.linspace(-5, 5, 1000)
        mu = 0.0
        var = 1.0
        f = normal_pdf(z, mu, var)
        
        f_norm = normalize_pdf_on_grid(z, f)
        
        # Should already be normalized, but check anyway
        area = np.trapezoid(f_norm, z)
        assert np.isclose(area, 1.0, rtol=1e-3)
    
    def test_normalize_negative_values(self):
        """Test normalization with negative values.
        
        Note: normalize_pdf_on_grid clips negative values only for area calculation,
        but returns normalized original values. This test verifies the function works
        correctly even with negative input values.
        """
        z = np.linspace(0, 1, 100)
        f = np.linspace(-1, 2, 100)  # Some negative values
        
        # normalize_pdf_on_grid uses np.maximum(f, 0.0) for area calculation only
        # The returned values are f / area, so negative values remain but are normalized
        f_norm = normalize_pdf_on_grid(z, f)
        
        # Check that normalization is correct (area = 1)
        # Note: The function uses clipped values for area calculation but returns
        # normalized original values, so negative values may remain
        area = np.trapezoid(np.maximum(f_norm, 0.0), z)
        assert np.isclose(area, 1.0, rtol=1e-5)
        
        # Verify that the function doesn't crash and produces valid output
        assert len(f_norm) == len(f)
        assert np.isfinite(f_norm).all()
    
    def test_normalize_zero_integral_error(self):
        """Test that zero integral raises ValueError."""
        z = np.linspace(0, 1, 100)
        f = np.zeros(100)  # Zero PDF
        
        with pytest.raises(ValueError, match="non-positive"):
            normalize_pdf_on_grid(z, f)
    
    def test_normalize_preserves_shape(self):
        """Test that normalization preserves shape."""
        z = np.linspace(-2, 2, 200)
        f = normal_pdf(z, 0.0, 1.0)
        
        f_norm = normalize_pdf_on_grid(z, f)
        
        assert len(f_norm) == len(f)
        assert f_norm.shape == f.shape

