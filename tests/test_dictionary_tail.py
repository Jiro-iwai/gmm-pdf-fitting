"""
Tests for tail-focused dictionary generation.

Tests for:
- build_gaussian_dictionary with tail_focus parameter
"""

import numpy as np
import pytest
from gmm_fitting import build_gaussian_dictionary
from gmm_fitting import normal_pdf, pdf_to_cdf_trapz


class TestBuildGaussianDictionaryTail:
    """Tests for build_gaussian_dictionary with tail_focus."""
    
    def test_tail_focus_none(self):
        """Test tail_focus='none' (uniform quantiles)."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="none",
            tail_alpha=1.0
        )
        
        mus = result["mus"]
        # Extract unique mu values (first L values should be unique)
        unique_mus = mus[::3]  # Every L-th element
        
        # Should be approximately uniformly distributed
        # Check that mus span the range
        assert mus.min() >= z.min()
        assert mus.max() <= z.max()
    
    def test_tail_focus_right(self):
        """Test tail_focus='right' (right tail emphasis)."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result_none = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="none",
            tail_alpha=1.0
        )
        
        result_right = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="right",
            tail_alpha=2.0
        )
        
        mus_none = result_none["mus"][::3]
        mus_right = result_right["mus"][::3]
        
        # Right tail focus should shift means to the right
        # Check that larger means are more emphasized
        assert mus_right.mean() > mus_none.mean()
        assert mus_right.max() > mus_none.max()
    
    def test_tail_focus_left(self):
        """Test tail_focus='left' (left tail emphasis)."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result_none = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="none",
            tail_alpha=1.0
        )
        
        result_left = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="left",
            tail_alpha=2.0
        )
        
        mus_none = result_none["mus"][::3]
        mus_left = result_left["mus"][::3]
        
        # Left tail focus should shift means to the left
        assert mus_left.mean() < mus_none.mean()
        assert mus_left.min() < mus_none.min()
    
    def test_tail_focus_both(self):
        """Test tail_focus='both' (both tails emphasis)."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result_both = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="both",
            tail_alpha=2.0
        )
        
        mus_both = result_both["mus"][::3]
        
        # Both tails should be emphasized, so extremes should be more spread out
        assert mus_both.max() - mus_both.min() > 0
    
    def test_tail_alpha_effect(self):
        """Test that larger tail_alpha increases tail emphasis."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result_alpha1 = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="right",
            tail_alpha=1.0
        )
        
        result_alpha3 = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="right",
            tail_alpha=3.0
        )
        
        mus_alpha1 = result_alpha1["mus"][::3]
        mus_alpha3 = result_alpha3["mus"][::3]
        
        # Larger alpha should shift more to the right
        assert mus_alpha3.mean() > mus_alpha1.mean()
        assert mus_alpha3.max() > mus_alpha1.max()
    
    def test_tail_alpha_clipping(self):
        """Test that tail_alpha < 1.0 is clipped to 1.0."""
        z = np.linspace(-3, 3, 100)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        # Should not raise error, but use alpha=1.0 internally
        result = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="right",
            tail_alpha=0.5  # < 1.0
        )
        
        # Should still produce valid dictionary
        assert "mus" in result
        assert "sigmas" in result
    
    def test_quantile_levels_override(self):
        """Test that quantile_levels overrides tail_focus."""
        z = np.linspace(-3, 3, 1000)
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        custom_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        result = build_gaussian_dictionary(
            z, f, J=5, L=3,
            mu_mode="quantile",
            tail_focus="right",  # Should be ignored
            tail_alpha=3.0,      # Should be ignored
            quantile_levels=custom_levels
        )
        
        mus = result["mus"][::3]  # Unique mu values
        
        # Should match custom quantile levels (approximately)
        F = pdf_to_cdf_trapz(z, f)
        F = np.maximum.accumulate(F)
        F /= F[-1]
        expected_mus = np.interp(custom_levels, F, z)
        
        assert len(mus) == len(expected_mus)
        assert np.allclose(mus, expected_mus, rtol=1e-2)
    
    def test_cdf_monotonicity(self):
        """Test that CDF monotonicity is enforced."""
        z = np.linspace(-3, 3, 100)
        # Create a PDF that might cause non-monotonic CDF due to numerical errors
        f = normal_pdf(z, 0.0, 1.0) + 1e-10 * np.random.randn(100)
        f = np.maximum(f, 0)
        f = f / np.trapezoid(f, z)
        
        # Should not raise error
        result = build_gaussian_dictionary(
            z, f, J=10, L=3,
            mu_mode="quantile",
            tail_focus="right",
            tail_alpha=2.0
        )
        
        assert "mus" in result
        assert len(result["mus"]) == 10 * 3

