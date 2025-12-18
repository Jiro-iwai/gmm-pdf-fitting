"""
Tests for custom initialization in EM method.

Tests for:
- init="custom" option in fit_gmm1d_to_pdf_weighted_em
"""

import numpy as np
import pytest
from gmm_fitting import (
    fit_gmm1d_to_pdf_weighted_em,
    GMM1DParams,
)
from gmm_fitting import (
    normal_pdf,
    normalize_pdf_on_grid,
    VAR_FLOOR,
)


class TestCustomInit:
    """Tests for custom initialization."""
    
    def test_custom_init_basic(self):
        """Test basic custom initialization."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        K = 3
        pi_init = np.array([0.3, 0.4, 0.3])
        mu_init = np.array([-1.0, 0.0, 1.0])
        var_init = np.array([0.5, 1.0, 0.5])
        
        init_params = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f,
            K=K,
            max_iter=100,
            tol=1e-6,
            n_init=1,
            init="custom",
            init_params=init_params,
        )
        
        assert len(params.pi) == K
        assert len(params.mu) == K
        assert len(params.var) == K
        assert np.allclose(np.sum(params.pi), 1.0)
        assert np.all(params.pi >= 0)
        assert np.all(params.var >= VAR_FLOOR)
    
    def test_custom_init_normalization(self):
        """Test that custom init params are normalized."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        # Unnormalized weights
        pi_init = np.array([0.5, 0.5, 0.5])  # Sum = 1.5
        mu_init = np.array([-1.0, 0.0, 1.0])
        var_init = np.array([1.0, 1.0, 1.0])
        
        init_params = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f,
            K=3,
            max_iter=100,
            tol=1e-6,
            n_init=1,
            init="custom",
            init_params=init_params,
        )
        
        # Should be normalized
        assert np.isclose(np.sum(params.pi), 1.0, rtol=1e-6)
    
    def test_custom_init_var_clipping(self):
        """Test that var_init is clipped to VAR_FLOOR."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        pi_init = np.array([0.3, 0.4, 0.3])
        mu_init = np.array([-1.0, 0.0, 1.0])
        var_init = np.array([1e-15, 1.0, 1e-12])  # Very small values
        
        init_params = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f,
            K=3,
            max_iter=100,
            tol=1e-6,
            n_init=1,
            init="custom",
            init_params=init_params,
            reg_var=VAR_FLOOR,
        )
        
        # All variances should be >= VAR_FLOOR
        assert np.all(params.var >= VAR_FLOOR)
    
    def test_custom_init_missing_params(self):
        """Test error when init_params is missing required keys."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        # Missing "var"
        init_params = {
            "pi": np.array([0.3, 0.4, 0.3]),
            "mu": np.array([-1.0, 0.0, 1.0]),
        }
        
        with pytest.raises(ValueError, match="init_params must contain"):
            fit_gmm1d_to_pdf_weighted_em(
                z, f,
                K=3,
                init="custom",
                init_params=init_params,
            )
    
    def test_custom_init_shape_mismatch(self):
        """Test error when shapes don't match K."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        # Wrong shape
        init_params = {
            "pi": np.array([0.5, 0.5]),  # K=2 but K=3 specified
            "mu": np.array([-1.0, 0.0]),
            "var": np.array([1.0, 1.0]),
        }
        
        with pytest.raises(ValueError, match="must have length K"):
            fit_gmm1d_to_pdf_weighted_em(
                z, f,
                K=3,
                init="custom",
                init_params=init_params,
            )
    
    def test_custom_init_zero_sum_error(self):
        """Test error when sum of pi_init is zero."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        pi_init = np.array([0.0, 0.0, 0.0])  # Sum = 0
        mu_init = np.array([-1.0, 0.0, 1.0])
        var_init = np.array([1.0, 1.0, 1.0])
        
        init_params = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        with pytest.raises(ValueError, match="sum of pi_init must be > 0"):
            fit_gmm1d_to_pdf_weighted_em(
                z, f,
                K=3,
                init="custom",
                init_params=init_params,
            )
    
    def test_custom_init_multiple_trials(self):
        """Test custom init with n_init > 1 (should add perturbations)."""
        z = np.linspace(-3, 3, 200)
        f = normal_pdf(z, 0.0, 1.0)
        f = normalize_pdf_on_grid(z, f)
        
        pi_init = np.array([0.3, 0.4, 0.3])
        mu_init = np.array([-1.0, 0.0, 1.0])
        var_init = np.array([1.0, 1.0, 1.0])
        
        init_params = {
            "pi": pi_init,
            "mu": mu_init,
            "var": var_init,
        }
        
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f,
            K=3,
            max_iter=100,
            tol=1e-6,
            n_init=3,  # Multiple trials
            init="custom",
            init_params=init_params,
        )
        
        # Should still work
        assert len(params.pi) == 3
        assert np.isclose(np.sum(params.pi), 1.0)

