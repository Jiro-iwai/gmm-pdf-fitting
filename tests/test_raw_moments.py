"""
Tests for raw moment computation functions.

Tests for:
- compute_pdf_raw_moments: Compute raw moments from PDF on grid
"""

import numpy as np
import pytest
from gmm_fitting import (
    compute_pdf_raw_moments,
    normal_pdf,
    normalize_pdf_on_grid,
)


class TestComputePDFRawMoments:
    """Tests for compute_pdf_raw_moments function."""
    
    def test_standard_normal(self):
        """Test raw moments for standard normal distribution."""
        z = np.linspace(-5, 5, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        
        M = compute_pdf_raw_moments(z, f, max_order=4)
        
        # M[0] = 1
        assert np.isclose(M[0], 1.0, rtol=1e-6)
        # M[1] = 0 (mean)
        assert np.abs(M[1]) < 1e-6
        # M[2] = 1 (variance = M[2] - M[1]^2 = 1)
        assert np.isclose(M[2], 1.0, rtol=1e-3)
        # M[3] = 0 (third moment)
        assert np.abs(M[3]) < 1e-3
        # M[4] = 3 (fourth moment for standard normal)
        assert np.isclose(M[4], 3.0, rtol=1e-2)
    
    def test_normalized_pdf(self):
        """Test that function normalizes PDF internally."""
        z = np.linspace(-3, 3, 100)
        # Unnormalized PDF (area != 1)
        f = normal_pdf(z, 0.0, 1.0) * 2.0
        
        M = compute_pdf_raw_moments(z, f, max_order=2)
        
        # Should still return M[0] = 1 after normalization
        assert np.isclose(M[0], 1.0, rtol=1e-6)
        # Mean should still be 0
        assert np.abs(M[1]) < 1e-5
    
    def test_max_order(self):
        """Test with different max_order values."""
        z = np.linspace(-3, 3, 100)
        f = normal_pdf(z, 0.0, 1.0)
        
        M2 = compute_pdf_raw_moments(z, f, max_order=2)
        M4 = compute_pdf_raw_moments(z, f, max_order=4)
        
        assert len(M2) == 3  # M[0], M[1], M[2]
        assert len(M4) == 5  # M[0], M[1], M[2], M[3], M[4]
        assert np.allclose(M2, M4[:3], rtol=1e-5)
    
    def test_shifted_normal(self):
        """Test raw moments for shifted normal distribution."""
        mu = 2.0
        var = 1.5
        z = np.linspace(mu - 5, mu + 5, 1000)
        f = normal_pdf(z, mu, var)
        
        M = compute_pdf_raw_moments(z, f, max_order=4)
        
        # M[0] = 1
        assert np.isclose(M[0], 1.0, rtol=1e-6)
        # M[1] = mu
        assert np.isclose(M[1], mu, rtol=1e-3)
        # M[2] = mu^2 + var
        expected_M2 = mu**2 + var
        assert np.isclose(M[2], expected_M2, rtol=1e-2)
    
    def test_non_negative_pdf(self):
        """Test that function handles non-negative PDF correctly."""
        z = np.linspace(0, 10, 100)
        # Simple exponential-like PDF (not normalized)
        f = np.exp(-z)
        
        M = compute_pdf_raw_moments(z, f, max_order=2)
        
        # Should normalize and compute moments
        assert M[0] > 0
        assert M[1] > 0  # Mean should be positive
        assert M[2] > 0  # Second moment should be positive
    
    def test_empty_array(self):
        """Test error handling for empty arrays."""
        z = np.array([])
        f = np.array([])
        
        with pytest.raises(ValueError):
            compute_pdf_raw_moments(z, f, max_order=4)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched array lengths."""
        z = np.linspace(-3, 3, 100)
        f = np.linspace(0, 1, 50)  # Different length
        
        with pytest.raises(ValueError):
            compute_pdf_raw_moments(z, f, max_order=4)
    
    def test_zero_pdf(self):
        """Test error handling for zero PDF."""
        z = np.linspace(-3, 3, 100)
        f = np.zeros(100)
        
        with pytest.raises(ValueError):
            compute_pdf_raw_moments(z, f, max_order=4)

