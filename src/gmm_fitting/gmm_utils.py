"""
Common utilities for GMM fitting methods.

This module provides shared utility functions and constants used by both
EM and LP methods for Gaussian Mixture Model fitting.
"""

import numpy as np
from typing import Tuple


# ============================================================
# Numerical Constants
# ============================================================

EPSILON = 1e-10
MASS_FLOOR = 1e-15
SIGMA_FLOOR = 1e-12
MIN_PDF_VALUE = 1e-10  # For log scale plotting
VAR_FLOOR = 1e-10  # Minimum variance for numerical stability


# ============================================================
# PDF and Statistical Functions
# ============================================================

def normal_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    Compute the probability density function of a univariate normal distribution.
    
    Parameters:
    -----------
    x : np.ndarray
        Points at which to evaluate the PDF
    mu : float
        Mean of the normal distribution
    var : float
        Variance of the normal distribution (must be positive)
    
    Returns:
    --------
    np.ndarray
        PDF values at x: N(x; mu, var) = (1/sqrt(2πσ²)) * exp(-(x-μ)²/(2σ²))
    """
    s = np.sqrt(var)
    u = (x - mu) / s
    return np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * s)


def normalize_pdf_on_grid(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Normalize a PDF on a grid so that its integral equals 1.
    
    Uses the trapezoidal rule for numerical integration to compute the area,
    then normalizes the PDF values by dividing by this area.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points (must be sorted)
    f : np.ndarray
        PDF values at grid points
    
    Returns:
    --------
    np.ndarray
        Normalized PDF values such that ∫ f(z) dz = 1
    
    Raises:
    ------
    ValueError
        If the computed integral is non-positive
    """
    z = np.asarray(z)
    f = np.asarray(f)
    area = np.trapezoid(np.maximum(f, 0.0), z)
    if area <= 0:
        raise ValueError("PDF integral is non-positive.")
    return f / area


def compute_pdf_statistics(z: np.ndarray, f: np.ndarray) -> dict:
    """
    Compute statistical moments from a PDF on a grid.
    
    Computes the first four moments: mean, standard deviation, skewness, and kurtosis.
    Uses numerical integration (trapezoidal rule) to compute moments from the PDF.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points (must be sorted)
    f : np.ndarray
        PDF values at grid points
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'mean': Mean (first moment) E[X]
        - 'std': Standard deviation sqrt(Var[X])
        - 'skewness': Skewness E[((X-μ)/σ)³] (third standardized moment)
        - 'kurtosis': Excess kurtosis E[((X-μ)/σ)⁴] - 3 (fourth standardized moment)
    
    Note:
    -----
    Excess kurtosis is used (subtracts 3), so normal distribution has kurtosis = 0.
    Positive kurtosis indicates heavier tails than normal, negative indicates lighter tails.
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    # Normalize PDF (ensure it integrates to 1)
    f_norm = normalize_pdf_on_grid(z, f)
    
    # Compute moments using numerical integration (trapezoidal rule)
    # First moment (mean): E[X] = ∫ x * f(x) dx
    mean = np.trapezoid(z * f_norm, z)
    
    # Second moment: E[X²] = ∫ x² * f(x) dx
    mean2 = np.trapezoid(z * z * f_norm, z)
    # Variance: Var[X] = E[X²] - E[X]²
    variance = mean2 - mean * mean
    
    # Handle degenerate case (zero variance)
    if variance <= 0:
        return {
            'mean': mean,
            'std': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    std = np.sqrt(variance)
    
    # Standardized variable: (X - μ) / σ
    z_std = (z - mean) / std
    
    # Third standardized moment (skewness): E[((X-μ)/σ)³]
    # Positive skewness: right tail is longer, distribution is right-skewed
    # Negative skewness: left tail is longer, distribution is left-skewed
    skewness = np.trapezoid(z_std**3 * f_norm, z)
    
    # Fourth standardized moment (excess kurtosis): E[((X-μ)/σ)⁴] - 3
    # Positive kurtosis: heavier tails than normal distribution
    # Negative kurtosis: lighter tails than normal distribution
    kurtosis = np.trapezoid(z_std**4 * f_norm, z) - 3.0
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def compute_errors(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    quantile_ps: list[float] = [0.9, 0.99, 0.999],
    tail_weight_p0: float = 0.9,
) -> dict:
    """
    Compute error metrics between true and estimated PDFs.
    
    Computes:
    - PDF L∞ error: max_i |f_true(z_i) - f_hat(z_i)|
    - CDF L∞ error: max_i |F_true(z_i) - F_hat(z_i)|
    - Quantile errors: |q_p^true - q_p^hat| for each p in quantile_ps
    - Tail-weighted L1 error: ∫_{q_p0}^∞ |f_true(z) - f_hat(z)| dz
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f_true : np.ndarray
        True PDF values, shape (N,)
    f_hat : np.ndarray
        Estimated PDF values, shape (N,)
    quantile_ps : list[float]
        Quantile probability levels (default: [0.9, 0.99, 0.999])
    tail_weight_p0 : float
        Probability level for tail-weighted L1 error (default: 0.9)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "linf_pdf": float - PDF L∞ error
        - "linf_cdf": float - CDF L∞ error
        - "quantiles_true": dict - True quantiles {p: q_true}
        - "quantiles_hat": dict - Estimated quantiles {p: q_hat}
        - "quantile_abs_errors": dict - Quantile absolute errors {p: |q_true - q_hat|}
        - "tail_l1_error": float - Tail-weighted L1 error
    """
    z = np.asarray(z)
    f_true = np.asarray(f_true)
    f_hat = np.asarray(f_hat)
    
    if len(z) != len(f_true) or len(z) != len(f_hat):
        raise ValueError("z, f_true, and f_hat must have the same length")
    
    # Normalize PDFs
    f_true_norm = normalize_pdf_on_grid(z, f_true)
    f_hat_norm = normalize_pdf_on_grid(z, f_hat)
    
    # PDF L∞ error
    linf_pdf = np.max(np.abs(f_true_norm - f_hat_norm))
    
    # Compute CDFs with monotonicity guarantee
    F_true = pdf_to_cdf_trapz(z, f_true_norm)
    F_true = np.maximum.accumulate(F_true)
    if F_true[-1] > 0:
        F_true /= F_true[-1]
    
    F_hat = pdf_to_cdf_trapz(z, f_hat_norm)
    F_hat = np.maximum.accumulate(F_hat)
    if F_hat[-1] > 0:
        F_hat /= F_hat[-1]
    
    # CDF L∞ error
    linf_cdf = np.max(np.abs(F_true - F_hat))
    
    # Quantile errors
    quantiles_true = {}
    quantiles_hat = {}
    quantile_abs_errors = {}
    
    for p in quantile_ps:
        q_true = np.interp(p, F_true, z)
        q_hat = np.interp(p, F_hat, z)
        quantiles_true[p] = q_true
        quantiles_hat[p] = q_hat
        quantile_abs_errors[p] = abs(q_true - q_hat)
    
    # Tail-weighted L1 error
    q_true_p0 = np.interp(tail_weight_p0, F_true, z)
    mask = z >= q_true_p0
    if np.any(mask):
        tail_l1_error = np.trapezoid(
            np.abs(f_true_norm[mask] - f_hat_norm[mask]),
            z[mask]
        )
    else:
        tail_l1_error = 0.0
    
    return {
        "linf_pdf": linf_pdf,
        "linf_cdf": linf_cdf,
        "quantiles_true": quantiles_true,
        "quantiles_hat": quantiles_hat,
        "quantile_abs_errors": quantile_abs_errors,
        "tail_l1_error": tail_l1_error,
    }


def pdf_to_cdf_trapz(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute CDF from PDF using trapezoidal rule.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,)
    
    Returns:
    --------
    np.ndarray
        CDF values, shape (N,), with F[0] = 0 and F[-1] ≈ 1
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    if len(z) != len(f):
        raise ValueError("z and f must have the same length")
    
    if len(z) == 0:
        return np.array([])
    if len(z) == 1:
        return np.array([0.0])
    
    # Compute CDF using trapezoidal rule (vectorized)
    # dz[i] = z[i+1] - z[i] for i = 0, ..., N-2
    dz = np.diff(z)
    
    # Trapezoidal rule: F[i+1] = F[i] + 0.5 * (f[i] + f[i+1]) * dz[i]
    # This is equivalent to cumulative sum of trapezoidal areas
    trapezoidal_areas = 0.5 * (f[:-1] + f[1:]) * dz
    F = np.zeros(len(z))
    F[1:] = np.cumsum(trapezoidal_areas)
    
    return F


# ============================================================
# Moment Computation Functions
# ============================================================

def compute_gmm_moments_from_weights(
    weights: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute mean, variance, skewness, and kurtosis from GMM weights and parameters.
    
    Parameters:
    -----------
    weights : np.ndarray
        Mixing weights, shape (s,)
    mus : np.ndarray
        Component means, shape (s,)
    sigmas : np.ndarray
        Component standard deviations, shape (s,)
    
    Returns:
    --------
    mean : float
        Mean of the mixture
    variance : float
        Variance of the mixture
    skewness : float
        Skewness of the mixture
    kurtosis : float
        Excess kurtosis of the mixture
    """
    weights = np.asarray(weights)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    
    if len(weights) != len(mus) or len(weights) != len(sigmas):
        raise ValueError("weights, mus, and sigmas must have the same length")
    
    vars_component = sigmas**2
    
    # Mean: E[X] = sum(w_j * mu_j)
    mean = np.sum(weights * mus)
    
    # Variance: Var[X] = E[X²] - E[X]²
    # E[X²] = sum(w_j * (mu_j² + sigma_j²))
    mean2 = np.sum(weights * (mus**2 + vars_component))
    variance = mean2 - mean**2
    
    # Handle degenerate case (zero variance)
    if variance <= 0:
        return mean, 0.0, 0.0, 0.0
    
    std = np.sqrt(variance)
    
    # Third central moment: E[(X-μ)³] = sum(w_j * E[(X_j-μ)³])
    # For normal component: E[(X_j-μ)³] = 3*sigma_j²*(mu_j-μ) + (mu_j-μ)³
    mu_centered = mus - mean
    mu3_central = np.sum(weights * (mu_centered**3 + 3 * vars_component * mu_centered))
    
    # Fourth central moment: E[(X-μ)⁴] = sum(w_j * E[(X_j-μ)⁴])
    # For normal component: E[(X_j-μ)⁴] = 3*sigma_j⁴ + 6*sigma_j²*(mu_j-μ)² + (mu_j-μ)⁴
    mu4_central = np.sum(weights * (
        mu_centered**4 + 6 * vars_component * mu_centered**2 + 3 * vars_component**2
    ))
    
    # Skewness: E[((X-μ)/σ)³] = E[(X-μ)³] / σ³
    skewness = mu3_central / (std**3) if std > 0 else 0.0
    
    # Excess kurtosis: E[((X-μ)/σ)⁴] - 3 = E[(X-μ)⁴] / σ⁴ - 3
    kurtosis = (mu4_central / (std**4) - 3.0) if std > 0 else 0.0
    
    return mean, variance, skewness, kurtosis


def compute_pdf_raw_moments(
    z: np.ndarray,
    f: np.ndarray,
    max_order: int = 4
) -> np.ndarray:
    """
    Compute raw moments from PDF on grid.
    
    Computes raw moments M[n] = E[Z^n] = ∫ z^n f(z) dz for n = 0..max_order.
    The PDF is normalized internally before computing moments.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,) (may be unnormalized)
    max_order : int
        Maximum order of moments to compute (default: 4)
    
    Returns:
    --------
    np.ndarray
        Raw moments, shape (max_order+1,)
        M[0] = 1.0 (normalized)
        M[n] = ∫ z^n f(z) dz for n = 1..max_order
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    if len(z) == 0 or len(f) == 0:
        raise ValueError("z and f must be non-empty")
    if len(z) != len(f):
        raise ValueError("z and f must have the same length")
    
    # Normalize PDF
    f_norm = normalize_pdf_on_grid(z, f)
    
    # Compute raw moments using trapezoidal rule
    M = np.zeros(max_order + 1)
    M[0] = 1.0  # Always 1 for normalized PDF
    
    for n in range(1, max_order + 1):
        M[n] = np.trapezoid(z**n * f_norm, z)
    
    return M


def compute_component_raw_moments(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Compute raw moments (0-4th order) for each GMM component.
    
    For a normal distribution N(μ, σ²), the raw moments are:
    - m_0 = 1
    - m_1 = μ
    - m_2 = μ² + σ²
    - m_3 = μ³ + 3μσ²
    - m_4 = μ⁴ + 6μ²σ² + 3σ⁴
    
    Parameters:
    -----------
    mu : np.ndarray
        Component means, shape (K,)
    var : np.ndarray
        Component variances, shape (K,)
    
    Returns:
    --------
    A : np.ndarray
        Matrix of raw moments, shape (5, K)
        A[n, k] = n-th raw moment of component k
    """
    K = len(mu)
    A = np.zeros((5, K))
    
    # Order 0: m_0k = 1
    A[0, :] = 1.0
    
    # Order 1: m_1k = μ_k
    A[1, :] = mu
    
    # Order 2: m_2k = μ_k² + σ_k²
    A[2, :] = mu**2 + var
    
    # Order 3: m_3k = μ_k³ + 3μ_kσ_k²
    A[3, :] = mu**3 + 3 * mu * var
    
    # Order 4: m_4k = μ_k⁴ + 6μ_k²σ_k² + 3σ_k⁴
    A[4, :] = mu**4 + 6 * mu**2 * var + 3 * var**2
    
    return A

