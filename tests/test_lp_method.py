"""
Tests for LP-based GMM fitting functions.
"""
import numpy as np
import pytest
from gmm_fitting import (
    build_gaussian_dictionary,
    build_gaussian_dictionary_simple,
    compute_basis_matrices,
    pdf_to_cdf_trapz,
    solve_lp_pdf_linf,
    solve_lp_pdf_moments_linf,
    fit_gmm_lp_simple
)
from gmm_fitting import (
    compute_gmm_moments_from_weights,
    normal_pdf,
    normalize_pdf_on_grid,
    compute_pdf_statistics
)


class TestPDFToCDFTrapz:
    """Tests for pdf_to_cdf_trapz function."""
    
    def test_cdf_basic(self):
        """Test basic CDF calculation."""
        z = np.linspace(0, 1, 100)
        f = np.ones(100)  # Uniform PDF
        f = f / np.trapezoid(f, z)  # Normalize
        
        F = pdf_to_cdf_trapz(z, f)
        
        assert len(F) == len(z)
        assert F[0] == 0.0
        assert np.isclose(F[-1], 1.0, rtol=1e-5)
        assert np.all(np.diff(F) >= 0)  # Monotonic increasing
    
    def test_cdf_normal_distribution(self):
        """Test CDF calculation for normal distribution."""
        z = np.linspace(-5, 5, 1000)
        mu = 0.0
        var = 1.0
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, mu, var)
        f = f / np.trapezoid(f, z)  # Normalize
        
        F = pdf_to_cdf_trapz(z, f)
        
        assert len(F) == len(z)
        assert F[0] == 0.0
        assert np.isclose(F[-1], 1.0, rtol=1e-3)
        # At mean, CDF should be approximately 0.5
        mid = len(z) // 2
        assert np.isclose(F[mid], 0.5, rtol=1e-2)
    
    def test_cdf_monotonic(self):
        """Test that CDF is monotonic increasing."""
        z = np.linspace(-3, 3, 200)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        F = pdf_to_cdf_trapz(z, f)
        
        # Check monotonicity
        assert np.all(np.diff(F) >= -1e-10)  # Allow small numerical errors


class TestBuildGaussianDictionary:
    """Tests for build_gaussian_dictionary function."""
    
    def test_dictionary_uniform_mode(self):
        """Test dictionary generation with uniform mode."""
        z = np.linspace(-3, 3, 100)
        f = np.ones(100)
        f = f / np.trapezoid(f, z)
        
        dict_params = {
            "J": 10,
            "L": 3,
            "mu_mode": "uniform",
            "sigma_min_scale": 0.1,
            "sigma_max_scale": 2.0
        }
        
        result = build_gaussian_dictionary(z, f, **dict_params)
        
        assert "mus" in result
        assert "sigmas" in result
        assert len(result["mus"]) == 10 * 3  # J * L
        assert len(result["sigmas"]) == 10 * 3
        assert np.all(result["mus"] >= z.min())
        assert np.all(result["mus"] <= z.max())
        assert np.all(result["sigmas"] > 0)
    
    def test_dictionary_quantile_mode(self):
        """Test dictionary generation with quantile mode."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        dict_params = {
            "J": 10,
            "L": 3,
            "mu_mode": "quantile",
            "sigma_min_scale": 0.1,
            "sigma_max_scale": 2.0
        }
        
        result = build_gaussian_dictionary(z, f, **dict_params)
        
        assert len(result["mus"]) == 10 * 3
        assert len(result["sigmas"]) == 10 * 3
        # In quantile mode, means should be distributed according to CDF
        assert np.all(result["mus"] >= z.min())
        assert np.all(result["mus"] <= z.max())
    
    def test_dictionary_sigma_scales(self):
        """Test that sigma scales are applied correctly."""
        z = np.linspace(-3, 3, 100)
        f = np.ones(100)
        f = f / np.trapezoid(f, z)
        
        # Compute true variance
        mean_true = np.trapezoid(z * f, z)
        var_true = np.trapezoid((z - mean_true)**2 * f, z)
        sigma_z = np.sqrt(var_true)
        
        dict_params = {
            "J": 5,
            "L": 4,
            "mu_mode": "uniform",
            "sigma_min_scale": 0.5,
            "sigma_max_scale": 2.0
        }
        
        result = build_gaussian_dictionary(z, f, **dict_params)
        
        # Check that sigmas are in the expected range
        sigma_min_expected = 0.5 * sigma_z
        sigma_max_expected = 2.0 * sigma_z
        assert np.min(result["sigmas"]) >= sigma_min_expected * 0.9  # Allow some tolerance
        assert np.max(result["sigmas"]) <= sigma_max_expected * 1.1


class TestComputeBasisMatrices:
    """Tests for compute_basis_matrices function."""
    
    def test_basis_matrices_shape(self):
        """Test that basis matrices have correct shapes."""
        z = np.linspace(-3, 3, 100)
        mus = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.5, 1.0, 1.5])
        
        result = compute_basis_matrices(z, mus, sigmas)
        
        assert "Phi_pdf" in result
        assert "Phi_cdf" in result
        assert result["Phi_pdf"].shape == (len(z), len(mus))
        assert result["Phi_cdf"].shape == (len(z), len(mus))
    
    def test_basis_matrices_pdf_values(self):
        """Test that PDF basis matrix contains correct values."""
        z = np.linspace(-2, 2, 100)
        mus = np.array([0.0])
        sigmas = np.array([1.0])
        
        result = compute_basis_matrices(z, mus, sigmas)
        
        from gmm_fitting import normal_pdf
        expected = normal_pdf(z, 0.0, 1.0)
        
        assert np.allclose(result["Phi_pdf"][:, 0], expected, rtol=1e-5)
    
    def test_basis_matrices_cdf_values(self):
        """Test that CDF basis matrix contains correct values."""
        z = np.linspace(-2, 2, 100)
        mus = np.array([0.0])
        sigmas = np.array([1.0])
        
        result = compute_basis_matrices(z, mus, sigmas)
        
        from scipy.special import ndtr
        expected = ndtr((z - 0.0) / 1.0)
        
        assert np.allclose(result["Phi_cdf"][:, 0], expected, rtol=1e-5)
    
    def test_basis_matrices_non_negative(self):
        """Test that PDF basis matrix is non-negative."""
        z = np.linspace(-3, 3, 100)
        mus = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.5, 1.0, 1.5])
        
        result = compute_basis_matrices(z, mus, sigmas)
        
        assert np.all(result["Phi_pdf"] >= 0)
        assert np.all(result["Phi_cdf"] >= 0)
        assert np.all(result["Phi_cdf"] <= 1)


class TestBuildGaussianDictionarySimple:
    """Tests for build_gaussian_dictionary_simple function."""
    
    def test_dictionary_simple_basic(self):
        """Test basic dictionary generation."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        result = build_gaussian_dictionary_simple(
            z, f,
            K=5,
            L=3,
            sigma_min_scale=0.1,
            sigma_max_scale=2.0
        )
        
        assert "mus" in result
        assert "sigmas" in result
        assert len(result["mus"]) == 5 * 3  # K * L
        assert len(result["sigmas"]) == 5 * 3
        assert np.all(result["sigmas"] > 0)
    
    def test_dictionary_simple_range(self):
        """Test that dictionary covers [μ-3σ, μ+3σ] range."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        # Compute true statistics
        mean_true = np.trapezoid(z * f, z)
        var_true = np.trapezoid((z - mean_true)**2 * f, z)
        sigma_z = np.sqrt(var_true)
        
        result = build_gaussian_dictionary_simple(
            z, f,
            K=10,
            L=2,
            sigma_min_scale=0.1,
            sigma_max_scale=3.0
        )
        
        # Check that means are within [μ-3σ, μ+3σ]
        z_min_expected = mean_true - 3 * sigma_z
        z_max_expected = mean_true + 3 * sigma_z
        
        assert np.min(result["mus"]) >= z_min_expected - 1e-6
        assert np.max(result["mus"]) <= z_max_expected + 1e-6
    
    def test_dictionary_simple_segment_centers(self):
        """Test that means are at segment centers."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        K = 4
        result = build_gaussian_dictionary_simple(
            z, f,
            K=K,
            L=1,
            sigma_min_scale=0.1,
            sigma_max_scale=3.0
        )
        
        # Extract unique means (should be K values)
        unique_mus = np.unique(result["mus"])
        assert len(unique_mus) == K
        
        # Check that means are evenly spaced
        mean_true = np.trapezoid(z * f, z)
        var_true = np.trapezoid((z - mean_true)**2 * f, z)
        sigma_z = np.sqrt(var_true)
        z_min = mean_true - 3 * sigma_z
        z_max = mean_true + 3 * sigma_z
        segment_width = (z_max - z_min) / K
        
        for i, mu in enumerate(sorted(unique_mus)):
            expected = z_min + (i + 0.5) * segment_width
            assert np.isclose(mu, expected, rtol=1e-5)
    
    def test_dictionary_simple_sigma_scales(self):
        """Test that sigma scales are applied correctly."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        f = normal_pdf(z, 0.0, 1.0)
        f = f / np.trapezoid(f, z)
        
        # Compute true variance
        mean_true = np.trapezoid(z * f, z)
        var_true = np.trapezoid((z - mean_true)**2 * f, z)
        sigma_z = np.sqrt(var_true)
        
        result = build_gaussian_dictionary_simple(
            z, f,
            K=3,
            L=4,
            sigma_min_scale=0.5,
            sigma_max_scale=2.0
        )
        
        # Extract unique sigmas (should be L values)
        unique_sigmas = np.unique(result["sigmas"])
        assert len(unique_sigmas) == 4
        
        # Check that sigmas are in expected range
        sigma_min_expected = 0.5 * sigma_z
        sigma_max_expected = 2.0 * sigma_z
        assert np.min(unique_sigmas) >= sigma_min_expected * 0.9
        assert np.max(unique_sigmas) <= sigma_max_expected * 1.1


class TestSolveLPPDFLinf:
    """Tests for solve_lp_pdf_linf function."""
    
    def test_lp_solve_pdf_only_simple(self):
        """Test LP solving for PDF only (simple case)."""
        z = np.linspace(-2, 2, 50)
        from gmm_fitting import normal_pdf
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = f_true / np.trapezoid(f_true, z)
        
        # Create basis with exact match
        mus = np.array([0.0])
        sigmas = np.array([1.0])
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_linf(
            basis["Phi_pdf"],
            f_true,
            solver="highs"
        )
        
        assert result["status"] == 0  # Success
        assert len(result["w"]) == 1
        assert result["w"][0] >= 0
        assert np.isclose(np.sum(result["w"]), 1.0, rtol=1e-6)
        assert result["t_pdf"] >= 0
        assert "objective" in result
    
    def test_lp_solve_pdf_only_two_components(self):
        """Test LP solving with two components."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf
        # Create mixture PDF
        f1 = normal_pdf(z, -1.0, 0.5)
        f2 = normal_pdf(z, 1.0, 0.5)
        f_true = 0.5 * f1 + 0.5 * f2
        f_true = f_true / np.trapezoid(f_true, z)
        
        mus = np.array([-1.0, 1.0])
        sigmas = np.array([0.5, 0.5])
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_linf(
            basis["Phi_pdf"],
            f_true,
            solver="highs"
        )
        
        assert result["status"] == 0
        assert len(result["w"]) == 2
        assert np.all(result["w"] >= 0)
        assert np.isclose(np.sum(result["w"]), 1.0, rtol=1e-6)
    
    def test_lp_solve_pdf_constraints_satisfied(self):
        """Test that LP solution satisfies PDF constraints."""
        z = np.linspace(-2, 2, 50)
        from gmm_fitting import normal_pdf
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = f_true / np.trapezoid(f_true, z)
        
        mus = np.array([0.0])
        sigmas = np.array([1.0])
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_linf(
            basis["Phi_pdf"],
            f_true,
            solver="highs"
        )
        
        # Check that constraints are satisfied
        f_hat = basis["Phi_pdf"] @ result["w"]
        pdf_error = np.abs(f_hat - f_true)
        
        assert np.max(pdf_error) <= result["t_pdf"] + 1e-6  # Allow numerical error


class TestComputeGMMMomentsFromWeights:
    """Tests for _compute_gmm_moments_from_weights function."""
    
    def test_moments_single_component(self):
        """Test moment calculation for single component."""
        weights = np.array([1.0])
        mus = np.array([0.0])
        sigmas = np.array([1.0])
        
        mean, var, skew, kurt = compute_gmm_moments_from_weights(weights, mus, sigmas)
        
        assert np.isclose(mean, 0.0, rtol=1e-6)
        assert np.isclose(var, 1.0, rtol=1e-6)
        assert np.isclose(skew, 0.0, rtol=1e-6)
        assert np.isclose(kurt, 0.0, rtol=1e-6)  # Excess kurtosis for normal is 0
    
    def test_moments_two_components(self):
        """Test moment calculation for two components."""
        weights = np.array([0.5, 0.5])
        mus = np.array([-1.0, 1.0])
        sigmas = np.array([1.0, 1.0])
        
        mean, var, skew, kurt = compute_gmm_moments_from_weights(weights, mus, sigmas)
        
        # Mean should be 0 (symmetric)
        assert np.isclose(mean, 0.0, rtol=1e-6)
        # Variance should be > 1 (mixture has larger variance)
        assert var > 1.0
        # Should be symmetric, so skewness should be close to 0
        assert abs(skew) < 0.1
    
    def test_moments_weights_sum_to_one(self):
        """Test that weights are normalized."""
        weights = np.array([0.3, 0.7])
        mus = np.array([0.0, 1.0])
        sigmas = np.array([1.0, 1.0])
        
        mean, var, skew, kurt = compute_gmm_moments_from_weights(weights, mus, sigmas)
        
        # Mean should be weighted average
        expected_mean = 0.3 * 0.0 + 0.7 * 1.0
        assert np.isclose(mean, expected_mean, rtol=1e-6)


class TestSolveLPPDFMomentsLinf:
    """Tests for solve_lp_pdf_moments_linf function."""
    
    def test_moments_mode_simple(self):
        """Test moments mode with simple Gaussian."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, compute_pdf_statistics
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = f_true / np.trapezoid(f_true, z)
        
        # Compute target moments
        stats = compute_pdf_statistics(z, f_true)
        target_mean = stats['mean']
        target_variance = stats['std']**2
        target_skewness = stats['skewness']
        target_kurtosis = stats['kurtosis']
        
        # Create basis
        mus = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.8, 1.0, 0.8])
        basis = compute_basis_matrices(z, mus, sigmas)
        
        result = solve_lp_pdf_moments_linf(
            basis["Phi_pdf"],
            mus,
            sigmas,
            f_true,
            target_mean,
            target_variance,
            target_skewness,
            target_kurtosis,
            solver="highs",
            pdf_tolerance=0.1,
            max_moment_iter=5
        )
        
        assert result["status"] == 0  # Success
        assert len(result["w"]) == 3
        assert np.all(result["w"] >= 0)
        assert np.isclose(np.sum(result["w"]), 1.0, rtol=1e-6)
        assert "t_pdf" in result
        assert "t_mean" in result
        assert "t_var" in result
        assert "t_skew" in result
        assert "t_kurt" in result
        assert "moment_errors" in result
    
    def test_moments_mode_pdf_constraint(self):
        """Test that PDF constraint is satisfied in moments mode."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, compute_pdf_statistics
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = f_true / np.trapezoid(f_true, z)
        
        stats = compute_pdf_statistics(z, f_true)
        target_mean = stats['mean']
        target_variance = stats['std']**2
        target_skewness = stats['skewness']
        target_kurtosis = stats['kurtosis']
        
        mus = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.8, 1.0, 0.8])
        basis = compute_basis_matrices(z, mus, sigmas)
        
        pdf_tolerance = 0.05
        result = solve_lp_pdf_moments_linf(
            basis["Phi_pdf"],
            mus,
            sigmas,
            f_true,
            target_mean,
            target_variance,
            target_skewness,
            target_kurtosis,
            solver="highs",
            pdf_tolerance=pdf_tolerance,
            max_moment_iter=5
        )
        
        # Check PDF constraint
        assert result["t_pdf"] <= pdf_tolerance + 1e-6
        
        # Check that PDF error is within tolerance
        f_hat = basis["Phi_pdf"] @ result["w"]
        pdf_error = np.max(np.abs(f_hat - f_true))
        assert pdf_error <= pdf_tolerance + 1e-6


class TestFitGMMLPSimple:
    """Tests for fit_gmm_lp_simple function."""
    
    def test_fit_simple_gaussian(self):
        """Test fitting a simple Gaussian."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, normalize_pdf_on_grid
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        lp_params = {
            "solver": "highs",
            "sigma_min_scale": 0.5,
            "sigma_max_scale": 2.0
        }
        
        result, timing = fit_gmm_lp_simple(
            z, f_true,
            K=5,
            L=3,
            lp_params=lp_params
        )
        
        assert "weights" in result
        assert "mus" in result
        assert "sigmas" in result
        assert len(result["weights"]) == 5 * 3  # K * L
        assert len(result["mus"]) == 5 * 3
        assert len(result["sigmas"]) == 5 * 3
        assert np.all(result["weights"] >= 0)
        assert np.isclose(np.sum(result["weights"]), 1.0, rtol=1e-6)
        
        # Check timing dictionary
        assert "dict_generation" in timing
        assert "basis_computation" in timing
        assert "lp_solving" in timing
        assert "total" in timing
        
        # Check diagnostics
        assert "diagnostics" in result
        assert "t_pdf" in result["diagnostics"]
        assert "n_bases" in result["diagnostics"]
        assert "n_nonzero" in result["diagnostics"]
        assert "L" in result["diagnostics"]
        assert result["diagnostics"]["objective_mode"] == "pdf"
    
    def test_fit_k_components(self):
        """Test fitting with different K and L values."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, normalize_pdf_on_grid
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        lp_params = {
            "solver": "highs",
            "sigma_min_scale": 0.3,
            "sigma_max_scale": 3.0
        }
        
        result, timing = fit_gmm_lp_simple(
            z, f_true,
            K=8,
            L=4,
            lp_params=lp_params
        )
        
        assert len(result["weights"]) == 8 * 4
        assert len(result["mus"]) == 8 * 4
        assert len(result["sigmas"]) == 8 * 4
        assert np.all(result["weights"] >= 0)
        assert np.isclose(np.sum(result["weights"]), 1.0, rtol=1e-6)
        assert result["diagnostics"]["n_bases"] == 8 * 4
    
    def test_fit_max_pdf(self):
        """Test fitting to max(X,Y) PDF."""
        z = np.linspace(-3, 5, 200)
        from gmm_fitting import max_pdf_bivariate_normal
        from gmm_fitting import normalize_pdf_on_grid
        f_true = max_pdf_bivariate_normal(z, 0.0, 1.0, 0.0, 1.0, 0.5)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        lp_params = {
            "solver": "highs",
            "sigma_min_scale": 0.2,
            "sigma_max_scale": 3.0
        }
        
        result, timing = fit_gmm_lp_simple(
            z, f_true,
            K=6,
            L=5,
            lp_params=lp_params
        )
        
        assert len(result["weights"]) == 6 * 5
        assert np.all(result["weights"] >= 0)
        assert np.isclose(np.sum(result["weights"]), 1.0, rtol=1e-6)
        assert "lp_objective" in result
        assert result["lp_objective"] >= 0  # Should be non-negative
        assert result["diagnostics"]["t_pdf"] >= 0
    
    def test_fit_moments_mode(self):
        """Test fitting with objective_mode='moments'."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, normalize_pdf_on_grid, compute_pdf_statistics
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        # Compute target moments
        stats = compute_pdf_statistics(z, f_true)
        
        lp_params = {
            "solver": "highs",
            "sigma_min_scale": 0.5,
            "sigma_max_scale": 2.0,
            "pdf_tolerance": 0.1,
            "max_moment_iter": 5,
            "lambda_mean": 1.0,
            "lambda_variance": 1.0,
            "lambda_skewness": 1.0,
            "lambda_kurtosis": 1.0
        }
        
        result, timing = fit_gmm_lp_simple(
            z, f_true,
            K=5,
            L=3,
            lp_params=lp_params,
            objective_mode="moments"
        )
        
        assert "weights" in result
        assert len(result["weights"]) == 5 * 3
        assert np.all(result["weights"] >= 0)
        assert np.isclose(np.sum(result["weights"]), 1.0, rtol=1e-6)
        assert result["diagnostics"]["objective_mode"] == "moments"
        assert "t_mean" in result["diagnostics"]
        assert "t_var" in result["diagnostics"]
        assert "t_skew" in result["diagnostics"]
        assert "t_kurt" in result["diagnostics"]
        assert "moment_errors" in result["diagnostics"]
    
    def test_fit_moments_mode_pdf_constraint(self):
        """Test that PDF constraint is satisfied in moments mode."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, normalize_pdf_on_grid
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        lp_params = {
            "solver": "highs",
            "sigma_min_scale": 0.5,
            "sigma_max_scale": 2.0,
            "pdf_tolerance": 0.05,
            "max_moment_iter": 5
        }
        
        result, timing = fit_gmm_lp_simple(
            z, f_true,
            K=5,
            L=3,
            lp_params=lp_params,
            objective_mode="moments"
        )
        
        # Check that PDF error is within tolerance
        assert result["diagnostics"]["t_pdf"] <= lp_params["pdf_tolerance"] + 1e-6
    
    def test_fit_error_handling(self):
        """Test error handling for invalid parameters."""
        z = np.linspace(-3, 3, 100)
        from gmm_fitting import normal_pdf, normalize_pdf_on_grid
        f_true = normal_pdf(z, 0.0, 1.0)
        f_true = normalize_pdf_on_grid(z, f_true)
        
        lp_params = {"solver": "highs"}
        
        # Test K < 1
        with pytest.raises(ValueError, match="K and L must be >= 1"):
            fit_gmm_lp_simple(z, f_true, K=0, L=3, lp_params=lp_params)
        
        # Test L < 1
        with pytest.raises(ValueError, match="K and L must be >= 1"):
            fit_gmm_lp_simple(z, f_true, K=3, L=0, lp_params=lp_params)
        
        # Test invalid objective_mode
        with pytest.raises(ValueError, match="objective_mode must be 'pdf', 'moments', or 'raw_moments'"):
            fit_gmm_lp_simple(z, f_true, K=3, L=3, lp_params=lp_params, objective_mode="invalid")
