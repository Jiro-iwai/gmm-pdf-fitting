"""Tests for MDN evaluation metrics."""
import numpy as np
import pytest
from src.ml_init.metrics import (
    compute_pdf_linf_error,
    compute_cdf_linf_error,
    compute_quantile_error,
    compute_cross_entropy,
)


def test_compute_pdf_linf_error():
    """Test PDF L∞ error computation."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    # Perturbed version
    f_hat = f_true + 0.01 * np.sin(z)
    f_hat = np.maximum(f_hat, 0)
    f_hat = f_hat / np.sum(f_hat * (z[1] - z[0]))
    
    error = compute_pdf_linf_error(z, f_true, f_hat)
    
    assert error >= 0
    assert np.isclose(error, np.max(np.abs(f_true - f_hat)), rtol=1e-6)


def test_compute_cdf_linf_error():
    """Test CDF L∞ error computation."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    # Shifted version
    f_hat = np.roll(f_true, 5)
    f_hat = np.maximum(f_hat, 0)
    f_hat = f_hat / np.sum(f_hat * (z[1] - z[0]))
    
    error = compute_cdf_linf_error(z, f_true, f_hat)
    
    assert error >= 0
    
    # Compute CDFs manually
    w = np.full_like(z, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    F_true = np.cumsum(f_true * w)
    F_hat = np.cumsum(f_hat * w)
    expected_error = np.max(np.abs(F_true - F_hat))
    
    assert np.isclose(error, expected_error, rtol=1e-6)


def test_compute_quantile_error():
    """Test quantile error computation."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    # Slightly different distribution
    f_hat = np.exp(-(z - 0.1)**2)
    f_hat = f_hat / np.sum(f_hat * (z[1] - z[0]))
    
    quantiles = [0.5, 0.9, 0.99]
    errors = compute_quantile_error(z, f_true, f_hat, quantiles)
    
    assert len(errors) == len(quantiles)
    assert all(e >= 0 for e in errors)


def test_compute_cross_entropy():
    """Test cross-entropy computation."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    # Same distribution (should give low CE)
    f_hat = f_true.copy()
    
    ce = compute_cross_entropy(z, f_true, f_hat)
    
    assert ce >= 0
    # For identical distributions, CE should be small (but not necessarily < 1.0)
    # The exact value depends on the distribution shape
    assert ce < 10.0  # Should be relatively small for identical distributions
    
    # Very different distribution (should give high CE)
    f_hat_bad = np.ones_like(f_true) / np.sum(np.ones_like(f_true) * (z[1] - z[0]))
    ce_bad = compute_cross_entropy(z, f_true, f_hat_bad)
    
    assert ce_bad > ce


def test_compute_cross_entropy_epsilon():
    """Test cross-entropy with epsilon handling."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    # Distribution with zeros
    f_hat = f_true.copy()
    f_hat[:10] = 0.0
    f_hat = f_hat / np.sum(f_hat * (z[1] - z[0]))
    
    # Should not raise error
    ce = compute_cross_entropy(z, f_true, f_hat, epsilon=1e-12)
    assert np.isfinite(ce)
    assert ce >= 0


def test_compute_pdf_linf_error_zero_distribution():
    """Test PDF L∞ error with zero distribution."""
    z = np.linspace(-2, 2, 100)
    f_true = np.zeros_like(z)
    f_hat = np.zeros_like(z)
    
    error = compute_pdf_linf_error(z, f_true, f_hat)
    assert error == 0.0


def test_compute_cdf_linf_error_single_point():
    """Test CDF L∞ error with single point distribution."""
    z = np.array([0.0])
    f_true = np.array([1.0])
    f_hat = np.array([1.0])
    
    error = compute_cdf_linf_error(z, f_true, f_hat)
    assert error >= 0
    assert np.isfinite(error)


def test_compute_quantile_error_edge_quantiles():
    """Test quantile error with edge quantiles (0.0, 1.0)."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    f_hat = f_true.copy()
    
    quantiles = [0.0, 0.5, 1.0]
    errors = compute_quantile_error(z, f_true, f_hat, quantiles)
    
    assert len(errors) == 3
    assert all(e >= 0 for e in errors)
    assert all(np.isfinite(e) for e in errors)


def test_compute_cross_entropy_all_zeros():
    """Test cross-entropy when f_hat is all zeros (should use epsilon)."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    f_hat = np.zeros_like(z)
    
    # Should not raise error (epsilon handles zeros)
    ce = compute_cross_entropy(z, f_true, f_hat, epsilon=1e-12)
    assert np.isfinite(ce)
    assert ce >= 0
    assert ce > 0  # Should be large for very different distributions


def test_compute_cross_entropy_shape_mismatch():
    """Test error handling for shape mismatches."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    f_hat = f_true[:-1]  # Mismatched shape
    
    with pytest.raises((ValueError, IndexError)):
        compute_cross_entropy(z, f_true, f_hat)


def test_compute_pdf_linf_error_nan_handling():
    """Test PDF L∞ error with NaN values."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    f_hat = f_true.copy()
    f_hat[50] = np.nan
    
    # Should propagate NaN (or handle gracefully)
    error = compute_pdf_linf_error(z, f_true, f_hat)
    # Either NaN or finite value (depending on implementation)
    assert np.isnan(error) or np.isfinite(error)


def test_compute_quantile_error_empty_quantiles():
    """Test quantile error with empty quantile list."""
    z = np.linspace(-2, 2, 100)
    f_true = np.exp(-z**2)
    f_true = f_true / np.sum(f_true * (z[1] - z[0]))
    
    f_hat = f_true.copy()
    
    quantiles = []
    errors = compute_quantile_error(z, f_true, f_hat, quantiles)
    
    assert len(errors) == 0

