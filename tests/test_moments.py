"""
Tests for moment calculation functions.
"""
import numpy as np
import pytest
from gmm_fitting import (
    _compute_central_moments,
    _central_to_raw_moments,
    _project_moments_qp,
    GMM1DParams
)
from gmm_fitting import compute_component_raw_moments


class TestComputeCentralMoments:
    """Tests for _compute_central_moments function."""
    
    def test_central_moments_uniform(self):
        """Test central moments for uniform distribution."""
        z = np.linspace(0, 1, 1000)
        w = np.ones(1000) / 1000  # Uniform weights
        
        mu_star, v_star, mu3_star, mu4_star = _compute_central_moments(z, w)
        
        # For uniform [0,1], mean should be 0.5
        assert np.isclose(mu_star, 0.5, rtol=1e-2)
        # Variance should be 1/12 ≈ 0.0833
        assert np.isclose(v_star, 1.0/12.0, rtol=1e-2)
        # Third moment should be close to 0 (symmetric)
        assert abs(mu3_star) < 0.01
    
    def test_central_moments_normal(self):
        """Test central moments for normal distribution."""
        z = np.linspace(-5, 5, 10000)
        # Approximate normal distribution weights
        mu_true = 0.0
        var_true = 1.0
        w = np.exp(-0.5 * (z - mu_true)**2 / var_true)
        w = w / np.sum(w)
        
        mu_star, v_star, mu3_star, mu4_star = _compute_central_moments(z, w)
        
        assert np.isclose(mu_star, mu_true, rtol=1e-2)
        assert np.isclose(v_star, var_true, rtol=1e-2)
        # Third moment should be close to 0 (symmetric)
        assert abs(mu3_star) < 0.1


class TestCentralToRawMoments:
    """Tests for _central_to_raw_moments function."""
    
    def test_conversion_identity(self):
        """Test that conversion preserves known relationships."""
        mu_star = 1.0
        v_star = 2.0
        mu3_star = 0.0
        mu4_star = 3.0
        
        M0, M1, M2, M3, M4 = _central_to_raw_moments(mu_star, v_star, mu3_star, mu4_star)
        
        # Check known relationships
        assert M0 == 1.0
        assert np.isclose(M1, mu_star)
        assert np.isclose(M2, v_star + mu_star**2)
    
    def test_conversion_zero_mean(self):
        """Test conversion for zero mean distribution."""
        mu_star = 0.0
        v_star = 1.0
        mu3_star = 0.0
        mu4_star = 3.0
        
        M0, M1, M2, M3, M4 = _central_to_raw_moments(mu_star, v_star, mu3_star, mu4_star)
        
        assert M0 == 1.0
        assert np.isclose(M1, 0.0)
        assert np.isclose(M2, 1.0)
        assert np.isclose(M3, 0.0)
        assert np.isclose(M4, 3.0)


class TestComputeComponentRawMoments:
    """Tests for _compute_component_raw_moments function."""
    
    def test_component_moments_single(self):
        """Test raw moments for single component."""
        mu = np.array([0.0])
        var = np.array([1.0])
        
        A = compute_component_raw_moments(mu, var)
        
        assert A.shape == (5, 1)
        # Check known values for N(0,1)
        assert np.isclose(A[0, 0], 1.0)  # M0
        assert np.isclose(A[1, 0], 0.0)  # M1 = mean
        assert np.isclose(A[2, 0], 1.0)  # M2 = mean^2 + var
        assert np.isclose(A[3, 0], 0.0)  # M3 = mean^3 + 3*mean*var
    
    def test_component_moments_multiple(self):
        """Test raw moments for multiple components."""
        mu = np.array([-1.0, 0.0, 1.0])
        var = np.array([0.5, 1.0, 0.5])
        
        A = compute_component_raw_moments(mu, var)
        
        assert A.shape == (5, 3)
        # Check M0 (should be 1 for all)
        assert np.allclose(A[0, :], 1.0)
        # Check M1 (should equal means)
        assert np.allclose(A[1, :], mu)
        # Check M2 (should equal mean^2 + var)
        assert np.allclose(A[2, :], mu**2 + var)


class TestProjectMomentsQP:
    """Tests for _project_moments_qp function."""
    
    def test_qp_projection_hard_constraints(self):
        """Test QP projection with hard constraints."""
        pi_em = np.array([0.3, 0.3, 0.4])
        mu = np.array([-1.0, 0.0, 1.0])
        var = np.array([0.5, 1.0, 0.5])
        
        # Compute target moments from EM weights
        A = compute_component_raw_moments(mu, var)
        target_raw = A @ pi_em
        
        pi_projected, success, info = _project_moments_qp(
            pi_em, mu, var, tuple(target_raw),
            qp_mode="hard"
        )
        
        assert len(pi_projected) == 3
        assert np.all(pi_projected >= 0)
        assert np.isclose(np.sum(pi_projected), 1.0)
        assert info['method'] in ['hard', 'soft', 'none']
    
    def test_qp_projection_soft_constraints(self):
        """Test QP projection with soft constraints."""
        pi_em = np.array([0.25, 0.25, 0.25, 0.25])
        mu = np.array([-1.5, -0.5, 0.5, 1.5])
        var = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Compute target moments from EM weights
        A = compute_component_raw_moments(mu, var)
        target_raw = A @ pi_em
        
        pi_projected, success, info = _project_moments_qp(
            pi_em, mu, var, tuple(target_raw),
            qp_mode="soft"
        )
        
        assert len(pi_projected) == 4
        assert np.all(pi_projected >= 0)
        assert np.isclose(np.sum(pi_projected), 1.0)
        assert info['method'] == 'soft'
    
    def test_qp_projection_preserves_sum(self):
        """Test that QP projection preserves sum of weights."""
        pi_em = np.array([0.2, 0.3, 0.5])
        mu = np.array([-1.0, 0.0, 1.0])
        var = np.array([1.0, 1.0, 1.0])
        
        A = compute_component_raw_moments(mu, var)
        target_raw = A @ pi_em
        
        pi_projected, success, info = _project_moments_qp(
            pi_em, mu, var, tuple(target_raw),
            qp_mode="soft"
        )
        
        assert np.isclose(np.sum(pi_projected), 1.0, rtol=1e-6)
    
    def test_qp_projection_non_negative(self):
        """Test that QP projection produces non-negative weights."""
        pi_em = np.array([0.1, 0.2, 0.3, 0.4])
        mu = np.array([-2.0, -1.0, 1.0, 2.0])
        var = np.array([0.5, 0.5, 0.5, 0.5])
        
        A = compute_component_raw_moments(mu, var)
        target_raw = A @ pi_em
        
        pi_projected, success, info = _project_moments_qp(
            pi_em, mu, var, tuple(target_raw),
            qp_mode="soft"
        )
        
        assert np.all(pi_projected >= 0)

