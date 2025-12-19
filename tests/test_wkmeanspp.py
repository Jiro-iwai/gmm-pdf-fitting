"""Tests for weighted k-means++ initialization."""
import numpy as np
import pytest
from src.ml_init.wkmeanspp import weighted_kmeanspp


def test_weighted_kmeanspp_basic():
    """Test basic weighted k-means++ functionality."""
    # Simple 1D case: 3 clusters
    z = np.linspace(-2, 2, 20)
    # Create a bimodal distribution
    f = np.exp(-(z + 1)**2) + np.exp(-(z - 1)**2)
    f = f / np.sum(f * (z[1] - z[0]))  # Normalize
    
    # Equal weights (trapezoidal rule)
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 3
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    # Check output shapes
    assert pi.shape == (K,)
    assert mu.shape == (K,)
    assert var.shape == (K,)
    
    # Check constraints
    assert np.allclose(np.sum(pi), 1.0), "Weights must sum to 1"
    assert np.all(pi >= 0), "Weights must be non-negative"
    assert np.all(var > 0), "Variances must be positive"
    assert np.all(var >= 1e-6), "Variances must be >= reg_var"
    
    # Check mu is sorted
    assert np.all(np.diff(mu) >= 0), "Means must be sorted"


def test_weighted_kmeanspp_empty_cluster():
    """Test handling of empty clusters."""
    # Create a very sparse distribution (most weight on one point)
    z = np.linspace(-2, 2, 20)
    f = np.zeros_like(z)
    f[10] = 1.0  # All weight on center point
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 5  # More clusters than data points with weight
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    # Should still return valid output
    assert pi.shape == (K,)
    assert mu.shape == (K,)
    assert var.shape == (K,)
    assert np.allclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)
    assert np.all(var > 0)


def test_weighted_kmeanspp_convergence():
    """Test that k-means converges."""
    z = np.linspace(-3, 3, 30)
    # Create a trimodal distribution
    f = (np.exp(-(z + 1.5)**2) + 
         np.exp(-z**2) + 
         np.exp(-(z - 1.5)**2))
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 3
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    # Check that means are near the modes
    assert np.min(np.abs(mu - (-1.5))) < 0.5, "Should find mode near -1.5"
    assert np.min(np.abs(mu - 0.0)) < 0.5, "Should find mode near 0"
    assert np.min(np.abs(mu - 1.5)) < 0.5, "Should find mode near 1.5"


def test_weighted_kmeanspp_reg_var():
    """Test that reg_var is applied correctly."""
    z = np.linspace(-2, 2, 20)
    f = np.exp(-z**2)
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 2
    reg_var = 1e-4
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=reg_var, max_iter=20)
    
    # All variances should be >= reg_var
    assert np.all(var >= reg_var), "Variances must be >= reg_var"


def test_weighted_kmeanspp_zero_weights():
    """Test handling of zero weights."""
    z = np.linspace(-2, 2, 20)
    f = np.zeros_like(z)
    f[5:15] = 1.0  # Weight only in middle
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 2
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    # Should still work
    assert np.allclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)
    assert np.all(var > 0)


def test_weighted_kmeanspp_k_equals_one():
    """Test edge case: K=1."""
    z = np.linspace(-2, 2, 20)
    f = np.exp(-z**2)
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 1
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    assert pi.shape == (1,)
    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert np.isclose(pi[0], 1.0)
    assert var[0] >= 1e-6


def test_weighted_kmeanspp_k_greater_than_n():
    """Test edge case: K > N (more clusters than data points)."""
    z = np.linspace(-2, 2, 5)  # Only 5 points
    f = np.exp(-z**2)
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 10  # More than N=5
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    # Should still return valid output
    assert pi.shape == (K,)
    assert mu.shape == (K,)
    assert var.shape == (K,)
    assert np.allclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)
    assert np.all(var > 0)


def test_weighted_kmeanspp_extreme_distribution():
    """Test with extreme distribution (very peaked)."""
    z = np.linspace(-2, 2, 100)
    f = np.zeros_like(z)
    f[50] = 1.0  # All weight on single point
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 3
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    assert np.allclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)
    assert np.all(var > 0)


def test_weighted_kmeanspp_negative_f():
    """Test handling of negative f values (should be clipped to 0)."""
    z = np.linspace(-2, 2, 20)
    f = np.exp(-z**2)
    f[0:5] = -0.1  # Some negative values
    f = f / np.sum(np.maximum(f, 0) * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 2
    # Should not raise error (negative values handled internally)
    pi, mu, var = weighted_kmeanspp(z, f, w, K, reg_var=1e-6, max_iter=20)
    
    assert np.allclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)
    assert np.all(var > 0)


def test_weighted_kmeanspp_shape_mismatch():
    """Test error handling for shape mismatches."""
    z = np.linspace(-2, 2, 20)
    f = np.exp(-z**2)
    f = f / np.sum(f * (z[1] - z[0]))
    
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    K = 3
    
    # Mismatched f shape
    with pytest.raises((ValueError, IndexError)):
        weighted_kmeanspp(z, f[:-1], w, K, reg_var=1e-6)
    
    # Mismatched w shape
    with pytest.raises((ValueError, IndexError)):
        weighted_kmeanspp(z, f, w[:-1], K, reg_var=1e-6)

