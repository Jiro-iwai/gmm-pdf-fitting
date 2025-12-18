"""
Tests for PDF statistics computation functions.
"""
import numpy as np
import pytest
from gmm_fitting import (
    compute_pdf_statistics,
    normalize_pdf_on_grid,
    normal_pdf
)


class TestComputePDFStatistics:
    """Tests for compute_pdf_statistics function."""
    
    def test_statistics_normal_distribution(self):
        """Test statistics for standard normal distribution."""
        z = np.linspace(-5, 5, 10000)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        stats = compute_pdf_statistics(z, f)
        
        # Check mean
        assert np.isclose(stats['mean'], 0.0, rtol=1e-2)
        # Check standard deviation
        assert np.isclose(stats['std'], 1.0, rtol=1e-2)
        # Check skewness (should be close to 0 for normal)
        assert abs(stats['skewness']) < 0.1
        # Check kurtosis (should be close to 0 for normal, excess kurtosis)
        assert abs(stats['kurtosis']) < 0.1
    
    def test_statistics_shifted_normal(self):
        """Test statistics for shifted normal distribution."""
        z = np.linspace(-5, 10, 10000)
        mu_true = 2.0
        var_true = 1.0
        f = normal_pdf(z, mu_true, var_true)
        f = normalize_pdf_on_grid(z, f)
        
        stats = compute_pdf_statistics(z, f)
        
        assert np.isclose(stats['mean'], mu_true, rtol=1e-2)
        assert np.isclose(stats['std'], np.sqrt(var_true), rtol=1e-2)
    
    def test_statistics_uniform_distribution(self):
        """Test statistics for uniform distribution."""
        z = np.linspace(0, 1, 10000)
        f = np.ones_like(z)
        f = normalize_pdf_on_grid(z, f)
        
        stats = compute_pdf_statistics(z, f)
        
        # Mean should be 0.5
        assert np.isclose(stats['mean'], 0.5, rtol=1e-2)
        # Std should be sqrt(1/12) ≈ 0.289
        assert np.isclose(stats['std'], np.sqrt(1.0/12.0), rtol=1e-2)
    
    def test_statistics_skewed_distribution(self):
        """Test statistics for skewed distribution."""
        z = np.linspace(0, 10, 10000)
        # Create a right-skewed distribution (exponential-like)
        f = np.exp(-z)
        f = normalize_pdf_on_grid(z, f)
        
        stats = compute_pdf_statistics(z, f)
        
        # Should have positive skewness
        assert stats['skewness'] > 0
        # Mean should be positive
        assert stats['mean'] > 0
    
    def test_statistics_zero_variance(self):
        """Test statistics for degenerate distribution (zero variance)."""
        # Use a very narrow distribution instead of exact zero variance
        z = np.linspace(-0.01, 0.01, 1000)
        mu = 0.0
        var = 1e-6  # Very small variance
        f = normal_pdf(z, mu, var)
        f = normalize_pdf_on_grid(z, f)
        
        stats = compute_pdf_statistics(z, f)
        
        # Standard deviation should be very small
        assert stats['std'] < 0.1
        # Skewness and kurtosis should be close to 0 for symmetric distribution
        assert abs(stats['skewness']) < 0.1
        assert abs(stats['kurtosis']) < 0.1
    
    def test_statistics_consistency(self):
        """Test that statistics are consistent across different grid sizes."""
        mu_true = 1.0
        var_true = 2.0
        
        for npoints in [1000, 5000, 10000]:
            z = np.linspace(-5, 7, npoints)
            f = normal_pdf(z, mu_true, var_true)
            f = normalize_pdf_on_grid(z, f)
            
            stats = compute_pdf_statistics(z, f)
            
            # Mean and std should be consistent
            assert np.isclose(stats['mean'], mu_true, rtol=1e-1)
            assert np.isclose(stats['std'], np.sqrt(var_true), rtol=1e-1)

