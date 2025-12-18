"""
EM Method - Gaussian Mixture Model (GMM) Approximation for Maximum of Bivariate Normal PDF

This module implements a weighted EM algorithm to fit a 1D Gaussian Mixture Model (GMM)
to the probability density function (PDF) of the maximum of two correlated normal random variables.

Main components:
1. PDF calculation for max(X, Y) where (X, Y) is bivariate normal
2. Weighted EM algorithm for GMM fitting with multiple initialization methods
3. PDF statistics computation (mean, std, skewness, kurtosis)
4. Visualization of PDF comparison including individual GMM components
"""

import numpy as np
import json
import time
import warnings
import matplotlib
matplotlib.use('Agg')  # Set backend (no GUI required)
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.special import ndtr, logsumexp
from scipy.optimize import minimize

from .gmm_utils import (
    normal_pdf,
    normalize_pdf_on_grid,
    compute_pdf_statistics,
    pdf_to_cdf_trapz,
    compute_gmm_moments_from_weights,
    compute_component_raw_moments,
    EPSILON,
    MASS_FLOOR,
    SIGMA_FLOOR,
    MIN_PDF_VALUE,
)

# ============================================================
# Constants
# ============================================================

# Default values
DEFAULT_MU_X = 0.0
DEFAULT_SIGMA_X = 0.8
DEFAULT_MU_Y = 0.0
DEFAULT_SIGMA_Y = 1.6
DEFAULT_RHO = 0.9
DEFAULT_Z_RANGE = [-6.0, 8.0]
DEFAULT_Z_NPOINTS = 2500
DEFAULT_K = 3
DEFAULT_MAX_ITER = 400
DEFAULT_TOL = 1e-10
DEFAULT_REG_VAR = 1e-6
DEFAULT_N_INIT = 8
DEFAULT_SEED = 1
DEFAULT_INIT = "quantile"
DEFAULT_OUTPUT_PATH = "pdf_comparison"
DEFAULT_SHOW_GRID_POINTS = True
DEFAULT_MAX_GRID_POINTS_DISPLAY = 200
DEFAULT_USE_MOMENT_MATCHING = False
DEFAULT_QP_MODE = "hard"
DEFAULT_SOFT_LAMBDA = 1e4

# Numerical constants are imported from gmm_utils

# Output formatting
SECTION_WIDTH = 70
COL_STAT_WIDTH = 15
COL_NUM_WIDTH = 18
COL_REL_WIDTH = 20

# ============================================================
# 1) True PDF for max(X, Y) where (X, Y) is bivariate normal
#    Formula: f_Z(z) = f_X(z) * P(Y<=z | X=z) + f_Y(z) * P(X<=z | Y=z)
# ============================================================

# Use normal_pdf from gmm_utils instead

def max_pdf_bivariate_normal(
    z: np.ndarray,
    mu_x: float, var_x: float,
    mu_y: float, var_y: float,
    rho: float
) -> np.ndarray:
    """
    Compute the PDF of Z = max(X, Y) where (X, Y) follows a bivariate normal distribution.
    
    The PDF is computed using the formula:
    f_Z(z) = f_X(z) * P(Y ≤ z | X = z) + f_Y(z) * P(X ≤ z | Y = z)
    
    This uses conditional distributions:
    - Y | X=z ~ N(μ_Y + ρ*(σ_Y/σ_X)*(z-μ_X), σ²_Y*(1-ρ²))
    - X | Y=z ~ N(μ_X + ρ*(σ_X/σ_Y)*(z-μ_Y), σ²_X*(1-ρ²))
    
    Parameters:
    -----------
    z : np.ndarray
        Points at which to evaluate the PDF
    mu_x : float
        Mean of X
    var_x : float
        Variance of X (must be positive)
    mu_y : float
        Mean of Y
    var_y : float
        Variance of Y (must be positive)
    rho : float
        Correlation coefficient between X and Y (-1 ≤ rho ≤ 1)
    
    Returns:
    --------
    np.ndarray
        PDF values f_Z(z) for each point in z
    """
    z = np.asarray(z)
    sx = np.sqrt(var_x)
    sy = np.sqrt(var_y)

    fx = normal_pdf(z, mu_x, var_x)
    fy = normal_pdf(z, mu_y, var_y)

    # Numerical stability: handle rho near ±1
    eps_rho = 1e-12
    rho2 = rho * rho
    delta = np.maximum(1.0 - rho2, eps_rho)
    
    # Y | X=z ~ N( mu_y + rho*(sy/sx)*(z-mu_x),  sy^2*(1-rho^2) )
    mu_y_given_x = mu_y + rho * (sy / sx) * (z - mu_x)
    sy_given_x = sy * np.sqrt(delta)
    sy_given_x = np.maximum(sy_given_x, SIGMA_FLOOR)
    p_y_le = ndtr((z - mu_y_given_x) / sy_given_x)

    # X | Y=z ~ N( mu_x + rho*(sx/sy)*(z-mu_y),  sx^2*(1-rho^2) )
    mu_x_given_y = mu_x + rho * (sx / sy) * (z - mu_y)
    sx_given_y = sx * np.sqrt(delta)
    sx_given_y = np.maximum(sx_given_y, SIGMA_FLOOR)
    p_x_le = ndtr((z - mu_x_given_y) / sx_given_y)

    fz = fx * p_y_le + fy * p_x_le
    return fz

def max_pdf_bivariate_normal_decomposed(
    z: np.ndarray,
    mu_x: float, var_x: float,
    mu_y: float, var_y: float,
    rho: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the decomposed PDF of Z = max(X, Y) into g_X(z) and g_Y(z) components.
    
    Returns:
    --------
    gx : np.ndarray
        Contribution from X winning: g_X(z) = f_X(z) * P(Y ≤ z | X = z)
    gy : np.ndarray
        Contribution from Y winning: g_Y(z) = f_Y(z) * P(X ≤ z | Y = z)
    """
    z = np.asarray(z)
    sx = np.sqrt(var_x)
    sy = np.sqrt(var_y)

    fx = normal_pdf(z, mu_x, var_x)
    fy = normal_pdf(z, mu_y, var_y)

    # Numerical stability: handle rho near ±1
    eps_rho = 1e-12
    rho2 = rho * rho
    delta = np.maximum(1.0 - rho2, eps_rho)

    # Y | X=z ~ N( mu_y + rho*(sy/sx)*(z-mu_x),  sy^2*(1-rho^2) )
    mu_y_given_x = mu_y + rho * (sy / sx) * (z - mu_x)
    sy_given_x = sy * np.sqrt(delta)
    sy_given_x = np.maximum(sy_given_x, SIGMA_FLOOR)
    p_y_le = ndtr((z - mu_y_given_x) / sy_given_x)

    # X | Y=z ~ N( mu_x + rho*(sx/sy)*(z-mu_y),  sx^2*(1-rho^2) )
    mu_x_given_y = mu_x + rho * (sx / sy) * (z - mu_y)
    sx_given_y = sx * np.sqrt(delta)
    sx_given_y = np.maximum(sx_given_y, SIGMA_FLOOR)
    p_x_le = ndtr((z - mu_x_given_y) / sx_given_y)

    gx = fx * p_y_le  # X wins: g_X(z)
    gy = fy * p_x_le  # Y wins: g_Y(z)
    return gx, gy
    """
    Compute the decomposed PDF of Z = max(X, Y) into g_X(z) and g_Y(z) components.
    
    Returns:
    --------
    gx : np.ndarray
        Contribution from X winning: g_X(z) = f_X(z) * P(Y ≤ z | X = z)
    gy : np.ndarray
        Contribution from Y winning: g_Y(z) = f_Y(z) * P(X ≤ z | Y = z)
    """
    z = np.asarray(z)
    sx = np.sqrt(var_x)
    sy = np.sqrt(var_y)

    fx = normal_pdf(z, mu_x, var_x)
    fy = normal_pdf(z, mu_y, var_y)

    # Y | X=z ~ N( mu_y + rho*(sy/sx)*(z-mu_x),  sy^2*(1-rho^2) )
    mu_y_given_x = mu_y + rho * (sy / sx) * (z - mu_x)
    sy_given_x = sy * np.sqrt(1.0 - rho * rho)
    p_y_le = ndtr((z - mu_y_given_x) / sy_given_x)

    # X | Y=z ~ N( mu_x + rho*(sx/sy)*(z-mu_y),  sx^2*(1-rho^2) )
    mu_x_given_y = mu_x + rho * (sx / sy) * (z - mu_y)
    sx_given_y = sx * np.sqrt(1.0 - rho * rho)
    p_x_le = ndtr((z - mu_x_given_y) / sx_given_y)

    gx = fx * p_y_le  # X wins: g_X(z)
    gy = fy * p_x_le  # Y wins: g_Y(z)
    return gx, gy

# Use normalize_pdf_on_grid from gmm_utils instead

# ============================================================
# 2) Weighted EM Algorithm for Fitting 1D GMM to PDF Grid
# ============================================================

@dataclass
class GMM1DParams:
    """
    Parameters for a 1D Gaussian Mixture Model (GMM).
    
    Attributes:
    -----------
    pi : np.ndarray
        Mixing weights (probabilities) for each component, shape (K,)
        Must satisfy: sum(pi) = 1, pi[k] >= 0 for all k
    mu : np.ndarray
        Mean values for each component, shape (K,)
    var : np.ndarray
        Variance values for each component, shape (K,)
        Must be positive: var[k] > 0 for all k
    """
    pi: np.ndarray   # shape (K,)
    mu: np.ndarray   # shape (K,)
    var: np.ndarray  # shape (K,)

def _log_normal_pdf_1d(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Compute log PDF of univariate normal distributions for all components.
    
    Computes log N(x_i; μ_k, σ²_k) for all combinations of data points x_i
    and GMM components k. Uses broadcasting for efficient computation.
    
    Parameters:
    -----------
    x : np.ndarray
        Data points, shape (N,)
    mu : np.ndarray
        Mean values for each component, shape (K,)
    var : np.ndarray
        Variance values for each component, shape (K,)
    
    Returns:
    --------
    np.ndarray
        Log PDF values, shape (N, K)
        Element [i, k] contains log N(x[i]; mu[k], var[k])
    """
    x = x[:, None]
    mu = mu[None, :]
    var = var[None, :]
    return -0.5 * (np.log(2.0*np.pi*var) + (x - mu)**2 / var)

def _init_gmm_qmi(
    z: np.ndarray,
    f: np.ndarray,
    K: int,
    sigma_floor: float = 1e-12,
    mass_floor: float = 1e-15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize GMM parameters using QMI (Quantile-based Moment Initialization).
    
    This implements Method 1 from initial_guess_spec.md:
    Divides PDF into K quantile bins and computes local moments for each bin.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points (must be strictly increasing)
    f : np.ndarray
        PDF values at grid points
    K : int
        Number of components
    sigma_floor : float
        Minimum variance value
    mass_floor : float
        Minimum bin mass threshold
    
    Returns:
    --------
    pi : np.ndarray
        Mixing weights (K,)
    mu : np.ndarray
        Means (K,)
    var : np.ndarray
        Variances (K,)
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    # Check input
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing")
    if K > len(z) / 2:
        raise ValueError(f"K ({K}) is too large relative to grid size ({len(z)})")
    
    # Step 0: Normalize PDF
    f = np.maximum(f, 0)
    area = np.trapezoid(f, z)
    if area <= 0:
        raise ValueError("PDF integral is non-positive")
    f = f / area
    
    # Step 1: Compute CDF
    dz = np.diff(z)
    f_avg = (f[:-1] + f[1:]) / 2  # Average PDF values for trapezoidal rule
    cdf = np.concatenate([[0], np.cumsum(f_avg * dz)])
    # Normalize CDF to ensure it ends at 1 (handle numerical errors)
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    
    # Step 2: Find quantile edges
    targets = np.linspace(0, 1, K + 1)
    edges = np.interp(targets, cdf, z)
    edges[0] = z[0]  # Clip to grid boundaries
    edges[-1] = z[-1]
    
    # Step 3: Compute local moments for each bin
    pi = np.zeros(K)
    mu = np.zeros(K)
    var = np.zeros(K)
    
    for k in range(K):
        z_low = edges[k]
        z_high = edges[k + 1]
        
        # Find indices within this bin
        idx_low = np.searchsorted(z, z_low, side='left')
        idx_high = np.searchsorted(z, z_high, side='right')
        
        if idx_low >= idx_high:
            # Empty bin: use bin center and small variance
            mu[k] = (z_low + z_high) / 2
            var[k] = sigma_floor
            pi[k] = 1.0 / K  # Will be normalized later
            continue
        
        # Extract bin region
        z_bin = z[idx_low:idx_high]
        f_bin = f[idx_low:idx_high]
        
        # Add boundary points if needed
        if idx_low > 0 and z[idx_low] > z_low:
            # Interpolate at left boundary
            f_left = np.interp(z_low, z[idx_low-1:idx_low+1], f[idx_low-1:idx_low+1])
            z_bin = np.concatenate([[z_low], z_bin])
            f_bin = np.concatenate([[f_left], f_bin])
        
        if idx_high < len(z) and z[idx_high - 1] < z_high:
            # Interpolate at right boundary
            if idx_high < len(z):
                f_right = np.interp(z_high, z[idx_high-1:idx_high+1], f[idx_high-1:idx_high+1])
            else:
                f_right = f[-1]
            z_bin = np.concatenate([z_bin, [z_high]])
            f_bin = np.concatenate([f_bin, [f_right]])
        
        # Compute bin mass (pi_k)
        if len(z_bin) > 1:
            dz_bin = np.diff(z_bin)
            f_avg_bin = (f_bin[:-1] + f_bin[1:]) / 2
            pi[k] = np.sum(f_avg_bin * dz_bin)
        else:
            pi[k] = 0
        
        # Handle very small mass
        if pi[k] < mass_floor:
            mu[k] = (z_low + z_high) / 2
            var[k] = sigma_floor
            pi[k] = mass_floor  # Will be normalized
            continue
        
        # Compute mean (mu_k)
        if len(z_bin) > 1:
            zf_avg = (z_bin[:-1] * f_bin[:-1] + z_bin[1:] * f_bin[1:]) / 2
            mu[k] = np.sum(zf_avg * dz_bin) / pi[k]
        else:
            mu[k] = z_bin[0]
        
        # Compute variance (sigma_k^2)
        if len(z_bin) > 1:
            diff2_avg = ((z_bin[:-1] - mu[k])**2 * f_bin[:-1] + 
                        (z_bin[1:] - mu[k])**2 * f_bin[1:]) / 2
            var[k] = np.sum(diff2_avg * dz_bin) / pi[k]
        else:
            var[k] = sigma_floor
    
    # Step 4: Normalize and stabilize
    pi = np.maximum(pi, 0)
    pi_sum = np.sum(pi)
    if pi_sum <= 0:
        raise ValueError("All bin masses are zero")
    pi = pi / pi_sum
    
    var = np.maximum(var, sigma_floor)
    
    # Sort by mean (optional but recommended)
    sort_idx = np.argsort(mu)
    pi = pi[sort_idx]
    mu = mu[sort_idx]
    var = var[sort_idx]
    
    return pi, mu, var

def _init_gmm_wqmi(
    z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    K: int,
    sigma_floor: float = 1e-12,
    mass_floor: float = 1e-15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize GMM parameters using WQMI (Winner-decomposed Quantile-based Moment Initialization).
    
    This implements Method 2 from initial_guess_spec.md:
    Decomposes MAX(X,Y) PDF into g_X and g_Y components, then applies QMI to each.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points (must be strictly increasing)
    gx : np.ndarray
        g_X(z) values (contribution from X winning)
    gy : np.ndarray
        g_Y(z) values (contribution from Y winning)
    K : int
        Number of components (must be >= 2)
    sigma_floor : float
        Minimum variance value
    mass_floor : float
        Minimum bin mass threshold
    
    Returns:
    --------
    pi : np.ndarray
        Mixing weights (K,)
    mu : np.ndarray
        Means (K,)
    var : np.ndarray
        Variances (K,)
    """
    if K < 2:
        raise ValueError("WQMI requires K >= 2 (need at least one component for each winner)")
    
    z = np.asarray(z)
    gx = np.asarray(gx)
    gy = np.asarray(gy)
    
    # Step 0: Normalize gx and gy
    gx = np.maximum(gx, 0)
    gy = np.maximum(gy, 0)
    total = np.trapezoid(gx + gy, z)
    if total <= 0:
        raise ValueError("Total mass of gx + gy is non-positive")
    gx = gx / total
    gy = gy / total
    
    # Step 1: Compute win probabilities
    p_x = np.trapezoid(gx, z)
    p_y = np.trapezoid(gy, z)
    
    # Step 2: Allocate components
    K_x = max(1, min(K - 1, int(np.round(K * p_x))))
    K_y = K - K_x
    
    # Step 3: Apply QMI to each side
    if K_x > 0 and p_x > mass_floor:
        h_x = gx / p_x  # Normalized PDF for X side
        pi_x, mu_x, var_x = _init_gmm_qmi(z, h_x, K_x, sigma_floor, mass_floor)
        pi_x = pi_x * p_x  # Scale by win probability
    else:
        # Fallback: use bin center
        pi_x = np.array([p_x])
        mu_x = np.array([np.sum(z * gx) / p_x if p_x > 0 else z[len(z)//2]])
        var_x = np.array([sigma_floor])
        K_x = 1
    
    if K_y > 0 and p_y > mass_floor:
        h_y = gy / p_y  # Normalized PDF for Y side
        pi_y, mu_y, var_y = _init_gmm_qmi(z, h_y, K_y, sigma_floor, mass_floor)
        pi_y = pi_y * p_y  # Scale by win probability
    else:
        # Fallback: use bin center
        pi_y = np.array([p_y])
        mu_y = np.array([np.sum(z * gy) / p_y if p_y > 0 else z[len(z)//2]])
        var_y = np.array([sigma_floor])
        K_y = 1
    
    # Step 4: Combine and normalize
    pi = np.concatenate([pi_x, pi_y])
    mu = np.concatenate([mu_x, mu_y])
    var = np.concatenate([var_x, var_y])
    
    # Final normalization
    pi_sum = np.sum(pi)
    if pi_sum > 0:
        pi = pi / pi_sum
    else:
        pi = np.ones(K) / K
    
    var = np.maximum(var, sigma_floor)
    
    # Sort by mean (recommended)
    sort_idx = np.argsort(mu)
    pi = pi[sort_idx]
    mu = mu[sort_idx]
    var = var[sort_idx]
    
    return pi, mu, var

def _grid_weights_from_pdf(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute grid weights proportional to PDF values and grid spacing.
    
    Creates weights w_i ∝ f(z_i) * Δz_i where Δz_i is the grid spacing.
    The weights are normalized so that sum(w_i) = 1.
    
    These weights are used in the weighted EM algorithm to approximate
    the continuous integral ∫ f(z) log g(z) dz ≈ Σ w_i log g(z_i).
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points (must be sorted)
    f : np.ndarray
        PDF values at grid points
    
    Returns:
    --------
    np.ndarray
        Normalized weights, shape (N,), where sum(weights) = 1
    
    Raises:
    ------
    ValueError
        If the total weight is non-positive
    """
    z = np.asarray(z)
    f = np.asarray(f)
    # 台形則の区間幅に対応する簡易Δz（端は最後の幅で埋める）
    dz = np.diff(z)
    dz = np.concatenate([dz, dz[-1:]])
    w = np.maximum(f, 0.0) * dz
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("Non-positive total weight from pdf.")
    return w / s

def fit_gmm1d_to_pdf_weighted_em(
    z: np.ndarray,
    f: np.ndarray,
    K: int = 3,
    max_iter: int = 300,
    tol: float = 1e-9,
    reg_var: float = 1e-6,
    n_init: int = 5,
    seed: int = 0,
    init: str = "quantile",  # "quantile", "random", "qmi", "wqmi", or "custom"
    init_params: Optional[Dict] = None,  # Additional parameters for initialization
    use_moment_matching: bool = False,  # Whether to apply moment matching QP projection
    qp_mode: str = "hard",  # "hard" or "soft"
    soft_lambda: float = 1e4  # Penalty coefficient for soft constraints
) -> Tuple[GMM1DParams, float, int]:
    """
    Fit a 1D Gaussian Mixture Model (GMM) to a PDF grid using weighted EM algorithm.
    
    This function approximates a continuous PDF f(z) with a GMM by maximizing
    the weighted log-likelihood:
        L = Σ_i w_i log(Σ_k π_k N(z_i; μ_k, σ²_k))
    where w_i ∝ f(z_i) * Δz_i are weights proportional to the PDF and grid spacing.
    
    The algorithm uses multiple random initializations (n_init) and returns the
    best result based on log-likelihood.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points at which PDF is evaluated, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,)
    K : int, optional
        Number of GMM components (default: 3)
    max_iter : int, optional
        Maximum number of EM iterations (default: 300)
    tol : float, optional
        Convergence tolerance for log-likelihood change (default: 1e-9)
    reg_var : float, optional
        Regularization term for variance (minimum variance value) (default: 1e-6)
    n_init : int, optional
        Number of random initializations to try (default: 5)
    seed : int, optional
        Random seed for reproducibility (default: 0)
    init : str, optional
        Initialization method (default: "quantile")
        - "quantile": Initialize means at quantiles of the PDF (simple)
        - "random": Initialize means randomly from grid points
        - "qmi": Quantile-based Moment Initialization (from initial_guess_spec.md)
        - "wqmi": Winner-decomposed QMI (for MAX(X,Y), uses gx/gy decomposition)
    init_params : dict, optional
        Additional parameters for initialization methods:
        - "sigma_floor": Minimum variance (default: 1e-12 or reg_var)
        - "mass_floor": Minimum bin mass (default: 1e-15)
        - For "wqmi": "gx" and "gy" arrays, or bivariate normal params
    use_moment_matching : bool, optional
        Whether to apply moment matching QP projection after EM (default: False)
        If True, mixing weights are adjusted to match target moments (mean, variance, skewness, kurtosis)
    qp_mode : str, optional
        QP projection mode: "hard" (try hard constraints first) or "soft" (use soft constraints directly)
        Default: "hard"
    soft_lambda : float, optional
        Penalty coefficient for soft constraints (default: 1e4)
        Larger values enforce stricter moment matching
    
    Returns:
    --------
    params : GMM1DParams
        Estimated GMM parameters (mixing weights, means, variances)
    best_ll : float
        Best weighted log-likelihood achieved
    best_iter : int
        Number of iterations for the best trial
    
    Raises:
    ------
    ValueError
        If z and f have incompatible shapes or K < 1
    """
    z = np.asarray(z)
    f = np.asarray(f)
    if z.ndim != 1 or f.ndim != 1 or len(z) != len(f):
        raise ValueError("z and f must be 1D arrays of same length.")
    if K < 1:
        raise ValueError("K must be >= 1")

    # Normalize PDF and compute grid weights
    # Weights w_i are proportional to f(z_i) * Δz_i and sum to 1
    f_norm = normalize_pdf_on_grid(z, f)
    w = _grid_weights_from_pdf(z, f_norm)  # ∑w_i=1

    rng = np.random.default_rng(seed)

    # Compute overall mean and variance for initialization
    # These are used as baseline values for component initialization
    m1 = float(np.sum(w * z))  # Weighted mean: E[Z]
    m2 = float(np.sum(w * z * z))  # Weighted second moment: E[Z²]
    var0 = max(m2 - m1*m1, reg_var)  # Variance: Var[Z] = E[Z²] - E[Z]²

    best_params: Optional[GMM1DParams] = None
    best_ll = -np.inf
    best_iter = 0

    # Prepare initialization parameters
    if init_params is None:
        init_params = {}
    sigma_floor_init = init_params.get("sigma_floor", max(reg_var, 1e-12))
    mass_floor_init = init_params.get("mass_floor", 1e-15)
    
    # Try multiple random initializations to avoid local optima
    for trial in range(n_init):
        # -------- Initialization --------
        if init == "quantile":
            # Initialize means at quantiles of the PDF (more stable)
            # This spreads components across the support of the PDF
            cdf = np.cumsum(w)  # Cumulative distribution function
            qs = (np.arange(K) + 0.5) / K  # Quantile positions
            mu = np.interp(qs, cdf, z)  # Interpolate to find quantile values
            # Add small noise to prevent identical means
            mu = mu + rng.normal(scale=0.05*np.sqrt(var0), size=K)
            # Initialize mixing weights uniformly and variances to overall variance
            pi = np.ones(K) / K  # Equal mixing weights: 1/K
            var = np.ones(K) * var0  # All components start with same variance
            
        elif init == "random":
            # Initialize means randomly from grid points
            mu = rng.choice(z, size=K, replace=False)
            # Initialize mixing weights uniformly and variances to overall variance
            pi = np.ones(K) / K  # Equal mixing weights: 1/K
            var = np.ones(K) * var0  # All components start with same variance
            
        elif init == "qmi":
            # QMI: Quantile-based Moment Initialization
            # Add small noise to PDF for randomization across trials
            if trial > 0:
                f_noisy = f_norm * (1.0 + rng.normal(scale=0.01, size=len(f_norm)))
                f_noisy = np.maximum(f_noisy, 0)
                f_noisy = normalize_pdf_on_grid(z, f_noisy)
            else:
                f_noisy = f_norm
            pi, mu, var = _init_gmm_qmi(z, f_noisy, K, sigma_floor_init, mass_floor_init)
            
        elif init == "wqmi":
            # WQMI: Winner-decomposed QMI
            # Get gx and gy from init_params or compute from bivariate normal params
            if "gx" in init_params and "gy" in init_params:
                gx = init_params["gx"]
                gy = init_params["gy"]
            elif all(key in init_params for key in ["mu_x", "var_x", "mu_y", "var_y", "rho"]):
                # Compute from bivariate normal parameters
                gx, gy = max_pdf_bivariate_normal_decomposed(
                    z,
                    init_params["mu_x"], init_params["var_x"],
                    init_params["mu_y"], init_params["var_y"],
                    init_params["rho"]
                )
            else:
                raise ValueError("WQMI requires 'gx'/'gy' or bivariate normal params in init_params")
            
            # Add small noise for randomization across trials
            if trial > 0:
                noise_scale = 0.01
                gx_noisy = gx * (1.0 + rng.normal(scale=noise_scale, size=len(gx)))
                gy_noisy = gy * (1.0 + rng.normal(scale=noise_scale, size=len(gy)))
                gx_noisy = np.maximum(gx_noisy, 0)
                gy_noisy = np.maximum(gy_noisy, 0)
            else:
                gx_noisy = gx
                gy_noisy = gy
            
            pi, mu, var = _init_gmm_wqmi(z, gx_noisy, gy_noisy, K, sigma_floor_init, mass_floor_init)
            
        elif init == "custom":
            # Custom initialization: use provided pi, mu, var
            if init_params is None:
                raise ValueError("init='custom' requires init_params with 'pi', 'mu', 'var'")
            
            required_keys = ["pi", "mu", "var"]
            for key in required_keys:
                if key not in init_params:
                    raise ValueError(f"init_params must contain '{key}' for init='custom'")
            
            pi_init = np.asarray(init_params["pi"])
            mu_init = np.asarray(init_params["mu"])
            var_init = np.asarray(init_params["var"])
            
            # Validate shapes
            if len(pi_init) != K or len(mu_init) != K or len(var_init) != K:
                raise ValueError(f"init_params['pi'], ['mu'], ['var'] must have length K={K}")
            
            # Validate and normalize pi
            if np.sum(pi_init) <= 0:
                raise ValueError("sum of pi_init must be > 0")
            pi = pi_init / np.sum(pi_init)
            
            # Use provided mu
            mu = mu_init.copy()
            
            # Clip var to reg_var
            var = np.maximum(var_init, reg_var)
            
            # Add perturbations for trial > 0
            if trial > 0:
                mu = mu * (1.0 + rng.normal(scale=0.01, size=K))
                var = var * (1.0 + rng.normal(scale=0.02, size=K))
                var = np.maximum(var, reg_var)
            
        else:
            raise ValueError(f"init must be 'quantile', 'random', 'qmi', 'wqmi', or 'custom', got '{init}'")

        prev_ll = -np.inf

        # EM algorithm iterations
        for it in range(max_iter):
            # -------- E-step: Compute responsibilities --------
            # Responsibility r_ik = P(component k | data point i)
            # Computed in log space for numerical stability
            log_comp = _log_normal_pdf_1d(z, mu, var)  # log N(z_i; μ_k, σ²_k), shape (N,K)
            log_pi = np.log(np.maximum(pi, 1e-300))[None, :]  # log π_k, shape (1,K)
            log_num = log_pi + log_comp  # log(π_k * N(z_i; μ_k, σ²_k)), shape (N,K)
            log_den = logsumexp(log_num, axis=1, keepdims=True)  # log(Σ_k π_k N(...)), shape (N,1)
            log_r = log_num - log_den  # log r_ik, shape (N,K)
            r = np.exp(log_r)  # Responsibilities r_ik, shape (N,K)

            # Compute weighted log-likelihood (approximation of ∫ f log ĝ)
            # This is the objective function we're maximizing
            ll = float(np.sum(w * log_den[:, 0]))

            # Check for convergence
            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # -------- M-step: Update parameters using weighted responsibilities --------
            # Effective number of points assigned to each component
            Nk = np.sum((w[:, None] * r), axis=0)  # N_k = Σ_i w_i r_ik, shape (K,)
            Nk = np.maximum(Nk, 1e-300)  # Prevent division by zero

            # Update mixing weights: π_k = N_k / Σ_k' N_k'
            pi = Nk / np.sum(Nk)

            # Update means: μ_k = (1/N_k) Σ_i w_i r_ik z_i
            mu = (w[:, None] * r * z[:, None]).sum(axis=0) / Nk

            # Update variances: σ²_k = (1/N_k) Σ_i w_i r_ik (z_i - μ_k)² + regularization
            diff2 = (z[:, None] - mu[None, :])**2  # (z_i - μ_k)², shape (N,K)
            var = (w[:, None] * r * diff2).sum(axis=0) / Nk  # Weighted variance
            var = np.maximum(var, reg_var)  # Apply regularization (minimum variance)

        # Save best result across all initialization trials
        if prev_ll > best_ll:
            best_ll = prev_ll
            best_params = GMM1DParams(pi=pi.copy(), mu=mu.copy(), var=var.copy())
            best_iter = it + 1  # it is 0-indexed, so add 1

    assert best_params is not None
    
    # Apply moment matching QP projection if requested
    if use_moment_matching:
        # Warn if K is too small for accurate moment matching
        if K < 5:
            warnings.warn(
                f"Moment matching with K={K} components may result in large errors, "
                f"especially for higher moments (skewness, kurtosis). "
                f"Recommendation: Use K≥5 for accurate moment matching. "
                f"Current constraint error may be large due to insufficient degrees of freedom "
                f"(5 moment constraints vs {K} weight parameters).",
                UserWarning,
                stacklevel=2
            )
        
        # Compute target central moments from true PDF
        mu_star, v_star, mu3_star, mu4_star = _compute_central_moments(z, w)
        
        # Convert to raw moments
        target_raw = _central_to_raw_moments(mu_star, v_star, mu3_star, mu4_star)
        
        # Project mixing weights to match moments
        pi_projected, qp_success, qp_info = _project_moments_qp(
            best_params.pi,
            best_params.mu,
            best_params.var,
            target_raw,
            qp_mode=qp_mode,
            soft_lambda=soft_lambda,
            var_floor=reg_var
        )
        
        # Warn if constraint error is large (indicates infeasibility)
        constraint_error = qp_info.get('constraint_error', float('inf'))
        if constraint_error > 1e-4:
            warnings.warn(
                f"Large moment constraint error ({constraint_error:.6e}) detected. "
                f"This indicates that exact moment matching may be infeasible with the current "
                f"GMM parameters (K={K}). Higher moments (especially kurtosis) may have significant errors. "
                f"Consider increasing K or adjusting soft_lambda (current: {soft_lambda:.0e}).",
                UserWarning,
                stacklevel=2
            )
        
        # Update best_params with projected weights
        best_params = GMM1DParams(
            pi=pi_projected,
            mu=best_params.mu.copy(),
            var=best_params.var.copy()
        )
        
        # Store QP info in a custom attribute (for debugging/output)
        best_params._qp_info = qp_info
    
    return best_params, best_ll, best_iter

def _compute_central_moments(z: np.ndarray, w: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute central moments (1-4th order) from weighted grid points.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    w : np.ndarray
        Weights (must sum to 1)
    
    Returns:
    --------
    mu_star : float
        Mean (1st moment)
    v_star : float
        Variance (2nd central moment)
    mu3_star : float
        3rd central moment
    mu4_star : float
        4th central moment
    """
    mu_star = np.sum(w * z)
    v_star = np.sum(w * (z - mu_star)**2)
    mu3_star = np.sum(w * (z - mu_star)**3)
    mu4_star = np.sum(w * (z - mu_star)**4)
    return mu_star, v_star, mu3_star, mu4_star

def _central_to_raw_moments(mu_star: float, v_star: float, mu3_star: float, mu4_star: float) -> Tuple[float, float, float, float, float]:
    """
    Convert central moments to raw moments (0-4th order).
    
    Parameters:
    -----------
    mu_star : float
        Mean (1st moment)
    v_star : float
        Variance (2nd central moment)
    mu3_star : float
        3rd central moment
    mu4_star : float
        4th central moment
    
    Returns:
    --------
    M0, M1, M2, M3, M4 : float
        Raw moments of orders 0-4
    """
    M0 = 1.0
    M1 = mu_star
    M2 = v_star + mu_star**2
    M3 = mu3_star + 3 * mu_star * v_star + mu_star**3
    M4 = mu4_star + 4 * mu_star * mu3_star + 6 * mu_star**2 * v_star + mu_star**4
    return M0, M1, M2, M3, M4

# Use compute_component_raw_moments from gmm_utils instead

def _project_moments_qp(
    pi_em: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    target_raw_moments: Tuple[float, float, float, float, float],
    qp_mode: str = "hard",
    soft_lambda: float = 1e4,
    var_floor: float = 1e-12
) -> Tuple[np.ndarray, bool, dict]:
    """
    Project GMM mixing weights to match target raw moments using QP.
    
    Implements moment matching projection from moment_em.md.
    
    Parameters:
    -----------
    pi_em : np.ndarray
        EM-learned mixing weights, shape (K,)
    mu : np.ndarray
        Component means, shape (K,)
    var : np.ndarray
        Component variances, shape (K,)
    target_raw_moments : tuple
        Target raw moments (M0, M1, M2, M3, M4)
    qp_mode : str
        "hard": Try hard constraints first, fallback to soft if fails
        "soft": Use soft constraints directly
    soft_lambda : float
        Penalty coefficient for soft constraints
    var_floor : float
        Minimum variance (for validation)
    
    Returns:
    --------
    pi_projected : np.ndarray
        Projected mixing weights
    success : bool
        True if hard constraints succeeded, False if soft fallback was used
    info : dict
        Additional information (moment errors, etc.)
    """
    import time
    qp_start_time = time.time()
    
    K = len(pi_em)
    M0, M1, M2, M3, M4 = target_raw_moments
    
    # Compute component raw moments matrix A (5 x K)
    A = compute_component_raw_moments(mu, var)
    b = np.array([M0, M1, M2, M3, M4])
    
    # Try hard constraints first if requested
    if qp_mode == "hard":
        try:
            # Hard constraint QP: minimize ||π - π_em||² subject to Aπ = b, π ≥ 0
            def objective(pi):
                return 0.5 * np.sum((pi - pi_em)**2)
            
            # Equality constraints: A @ pi = b
            constraints = {
                'type': 'eq',
                'fun': lambda pi: A @ pi - b
            }
            
            # Bounds: π_k ≥ 0
            bounds = [(0, None)] * K
            
            # Initial guess: EM weights
            x0 = pi_em.copy()
            
            # Solve QP (maxiter=100 to avoid long computation when infeasible)
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if result.success:
                pi_projected = result.x
                # Check constraint satisfaction
                constraint_error = np.max(np.abs(A @ pi_projected - b))
                if constraint_error < 1e-8:
                    # Normalize (numerical error correction)
                    pi_projected = np.maximum(pi_projected, 0)
                    pi_projected = pi_projected / np.sum(pi_projected)
                    
                    qp_elapsed_time = time.time() - qp_start_time
                    info = {
                        'method': 'hard',
                        'constraint_error': constraint_error,
                        'moment_errors': A @ pi_projected - b,
                        'qp_time': qp_elapsed_time
                    }
                    return pi_projected, True, info
        except Exception:
            # Hard constraints failed, fallback to soft
            pass
    
    # Soft constraint QP (fallback or direct)
    # Minimize: 0.5||π - π_em||² + (λ/2)||Aπ - b||²
    # Subject to: π ≥ 0, Σπ = 1
    def objective_soft(pi):
        diff = pi - pi_em
        constraint_violation = A @ pi - b
        return 0.5 * np.sum(diff**2) + 0.5 * soft_lambda * np.sum(constraint_violation**2)
    
    # Equality constraint: Σπ = 1
    constraints_soft = {
        'type': 'eq',
        'fun': lambda pi: np.sum(pi) - 1.0
    }
    
    # Bounds: π_k ≥ 0
    bounds_soft = [(0, None)] * K
    
    # Initial guess: EM weights
    x0_soft = pi_em.copy()
    
    # Solve soft QP
    result_soft = minimize(
        objective_soft,
        x0_soft,
        method='SLSQP',
        bounds=bounds_soft,
        constraints=constraints_soft,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result_soft.success:
        pi_projected = result_soft.x
        # Normalize (numerical error correction)
        pi_projected = np.maximum(pi_projected, 0)
        pi_projected = pi_projected / np.sum(pi_projected)
        
        qp_elapsed_time = time.time() - qp_start_time
        constraint_error = np.linalg.norm(A @ pi_projected - b)
        info = {
            'method': 'soft',
            'constraint_error': constraint_error,
            'moment_errors': A @ pi_projected - b,
            'qp_time': qp_elapsed_time
        }
        return pi_projected, False, info
    else:
        # If even soft fails, return EM weights
        qp_elapsed_time = time.time() - qp_start_time
        info = {
            'method': 'none',
            'constraint_error': np.linalg.norm(A @ pi_em - b),
            'moment_errors': A @ pi_em - b,
            'error': 'QP optimization failed',
            'qp_time': qp_elapsed_time
        }
        return pi_em.copy(), False, info

def gmm1d_pdf(z: np.ndarray, params: GMM1DParams) -> np.ndarray:
    """
    Evaluate the PDF of a 1D Gaussian Mixture Model at given points.
    
    Computes the PDF as a weighted sum of component PDFs:
        f(z) = Σ_k π_k N(z; μ_k, σ²_k)
    
    Parameters:
    -----------
    z : np.ndarray
        Points at which to evaluate the PDF
    params : GMM1DParams
        GMM parameters (mixing weights, means, variances)
    
    Returns:
    --------
    np.ndarray
        PDF values at z
    """
    z = np.asarray(z)
    K = len(params.pi)
    out = np.zeros_like(z, dtype=float)
    for k in range(K):
        out += params.pi[k] * normal_pdf(z, params.mu[k], params.var[k])
    return out

# Use compute_pdf_statistics from gmm_utils instead

def plot_pdf_comparison(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    output_path: str,
    mu_x: float, sigma_x: float,
    mu_y: float, sigma_y: float,
    rho: float,
    ll: float,
    show_grid_points: bool = True,
    max_grid_points_display: int = 200,
    gmm_params = None,
    component_threshold: float = 1e-8
):
    """
    Create comparison plots of true PDF and GMM-approximated PDF.
    
    Generates a single PNG file with two subplots:
    - Top: Linear scale plot (shows main features clearly)
    - Bottom: Logarithmic scale plot (shows tail behavior)
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points
    f_true : np.ndarray
        True PDF values (max of bivariate normal)
    f_hat : np.ndarray
        GMM-approximated PDF values
    output_path : str
        Output file path without extension (will add .png)
    mu_x : float
        Mean of X
    sigma_x : float
        Standard deviation of X
    mu_y : float
        Mean of Y
    sigma_y : float
        Standard deviation of Y
    rho : float
        Correlation coefficient between X and Y
    ll : float
        Log-likelihood value (displayed in title)
    show_grid_points : bool, optional
        Whether to display grid points on the plot (default: True)
    max_grid_points_display : int, optional
        Maximum number of grid points to display (default: 200)
        If grid has more points, they will be downsampled for display
    gmm_params : GMM1DParams, optional
        GMM parameters for plotting individual components
    component_threshold : float, optional
        Minimum weight threshold for displaying components (default: 1e-8)
        Components with weight below this threshold are not displayed
    
    Returns:
    --------
    None
        Saves plot to {output_path}.png
    """
    # Set font family for plot labels
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Clip PDF values to positive values for log scale plotting
    # Values below MIN_PDF_VALUE are set to MIN_PDF_VALUE to avoid log(0) issues
    f_true_pos = np.maximum(f_true, MIN_PDF_VALUE)
    f_hat_pos = np.maximum(f_hat, MIN_PDF_VALUE)
    
    # Use actual grid points for plotting to show the true discretization
    # This accurately represents what was used for fitting
    z_plot = z  # Use actual grid points
    f_true_plot = f_true  # Use actual PDF values at grid points
    f_hat_plot = f_hat  # Use actual GMM values at grid points
    f_true_plot_pos = np.maximum(f_true_plot, MIN_PDF_VALUE)
    f_hat_plot_pos = np.maximum(f_hat_plot, MIN_PDF_VALUE)
    
    # Create figure with two subplots (vertical layout)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Prepare grid points for display (as scatter points)
    # These are shown as markers on top of the line plots to indicate the actual grid points used
    if show_grid_points:
        if len(z) > max_grid_points_display:
            # Downsample grid points for display (too many points make plot cluttered)
            indices = np.linspace(0, len(z) - 1, max_grid_points_display, dtype=int)
            z_display = z[indices]
            f_true_display = f_true[indices]
        else:
            # Show all grid points (important for adaptive grids to see density variations)
            z_display = z
            f_true_display = f_true
    else:
        z_display = None
    
    # Compute individual component PDFs if gmm_params is provided
    component_pdfs = None
    component_pdfs_pos = None
    component_indices_to_plot = None
    if gmm_params is not None:
        K = len(gmm_params.pi)
        # Filter components by weight threshold
        component_indices_to_plot = np.where(gmm_params.pi >= component_threshold)[0]
        n_components_to_plot = len(component_indices_to_plot)
        
        if n_components_to_plot > 0:
            component_pdfs = np.zeros((len(z_plot), n_components_to_plot))
            for idx, k in enumerate(component_indices_to_plot):
                # Component PDF: π_k * N(z; μ_k, σ²_k)
                component_pdfs[:, idx] = gmm_params.pi[k] * normal_pdf(z_plot, gmm_params.mu[k], gmm_params.var[k])
            component_pdfs_pos = np.maximum(component_pdfs, MIN_PDF_VALUE)
    
    # -------- Top subplot: Linear scale --------
    # Shows main features of the PDF clearly
    # Plot using actual grid points (connected by lines, no markers on lines)
    ax1.plot(z_plot, f_true_plot, 'b-', linewidth=2, label='True PDF (max of bivariate normal)', alpha=0.8)
    ax1.plot(z_plot, f_hat_plot, 'r--', linewidth=2, label='GMM approximation', alpha=0.8)
    
    # Plot individual GMM components
    if component_pdfs is not None and component_indices_to_plot is not None:
        # Use different colors for components (cycle through a color palette)
        n_components_to_plot = len(component_indices_to_plot)
        colors = plt.cm.tab10(np.linspace(0, 1, n_components_to_plot))
        for idx, k in enumerate(component_indices_to_plot):
            ax1.plot(z_plot, component_pdfs[:, idx], ':', linewidth=1.5, 
                    color=colors[idx], alpha=0.6, 
                    label=f'Component {k+1} (π={gmm_params.pi[k]:.3f})')
    
    # Display grid points if requested
    # For adaptive grids, use varying marker sizes to show density variations
    if show_grid_points and z_display is not None:
        # Calculate spacing to visualize adaptive grid density
        spacing = np.diff(z)
        if len(spacing) > 0 and np.max(spacing) > 0:
            spacing_normalized = spacing / np.max(spacing)  # Normalize to [0, 1]
            # Use larger markers for denser regions (smaller spacing)
            marker_sizes = 10 + 20 * (1 - spacing_normalized)  # Size range: 10-30
            # Map spacing to marker sizes for each point
            # Each point gets the size based on spacing to its right neighbor
            if len(z_display) == len(z):
                # Show all points with adaptive sizing
                # First point uses first spacing, last point uses last spacing
                marker_sizes_display = np.concatenate([[marker_sizes[0]], marker_sizes])
            else:
                # Downsampled display: use uniform size
                marker_sizes_display = 15
        else:
            # Uniform spacing or single point: use uniform size
            marker_sizes_display = 15
        
        ax1.scatter(z_display, f_true_display, c='blue', s=marker_sizes_display, alpha=0.6, 
                   marker='o', edgecolors='darkblue', linewidths=0.5, 
                   label=f'Grid points (n={len(z)})', zorder=5)
    
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(
        f'PDF Comparison (Linear Scale)',
        fontsize=12
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(z_plot.min(), z_plot.max())
    
    # -------- Bottom subplot: Logarithmic scale --------
    # Shows tail behavior and small PDF values more clearly
    # Plot using actual grid points (connected by lines, no markers on lines)
    ax2.semilogy(z_plot, f_true_plot_pos, 'b-', linewidth=2, label='True PDF (max of bivariate normal)', alpha=0.8)
    ax2.semilogy(z_plot, f_hat_plot_pos, 'r--', linewidth=2, label='GMM approximation', alpha=0.8)
    
    # Plot individual GMM components on log scale
    if component_pdfs_pos is not None and component_indices_to_plot is not None:
        # Use same colors as linear scale
        n_components_to_plot = len(component_indices_to_plot)
        colors = plt.cm.tab10(np.linspace(0, 1, n_components_to_plot))
        for idx, k in enumerate(component_indices_to_plot):
            ax2.semilogy(z_plot, component_pdfs_pos[:, idx], ':', linewidth=1.5,
                        color=colors[idx], alpha=0.6,
                        label=f'Component {k+1} (π={gmm_params.pi[k]:.3f})')
    
    # Display grid points if requested (use clipped values for log scale)
    if show_grid_points and z_display is not None:
        f_true_display_pos = np.maximum(f_true_display, MIN_PDF_VALUE)
        # Use same adaptive marker sizing as linear scale
        spacing = np.diff(z)
        if len(spacing) > 0 and np.max(spacing) > 0:
            spacing_normalized = spacing / np.max(spacing)
            marker_sizes = 10 + 20 * (1 - spacing_normalized)
            if len(z_display) == len(z):
                marker_sizes_display = np.concatenate([[marker_sizes[0]], marker_sizes])
            else:
                marker_sizes_display = 15
        else:
            marker_sizes_display = 15
        
        ax2.scatter(z_display, f_true_display_pos, c='blue', s=marker_sizes_display, alpha=0.6,
                   marker='o', edgecolors='darkblue', linewidths=0.5,
                   label=f'Grid points (n={len(z)})', zorder=5)
    
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('Probability Density (log scale)', fontsize=12)
    ax2.set_title(
        f'PDF Comparison (Log Scale)',
        fontsize=12
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines
    ax2.set_xlim(z_plot.min(), z_plot.max())
    
    # Overall title with parameter information and log-likelihood
    fig.suptitle(
        f'PDF Comparison: μ_X={mu_x:.2f}, σ_X={sigma_x:.2f}, μ_Y={mu_y:.2f}, σ_Y={sigma_y:.2f}, ρ={rho:.2f} | Log-likelihood: {ll:.6f}',
        fontsize=13,
        y=0.995
    )
    
    # Save figure
    # Note: bbox_inches='tight' handles layout automatically, so tight_layout() is not needed
    # and can cause warnings when axes decorations are too large
    plt.savefig(f'{output_path}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 2) Output formatting functions
# ============================================================

def print_section_header(title: str, width: int = SECTION_WIDTH) -> None:
    """Print a section header with separator lines."""
    print("\n" + "="*width)
    print(title)
    print("="*width)

def print_subsection_header(title: str, width: int = SECTION_WIDTH) -> None:
    """Print a subsection header with separator lines."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)

def print_em_results(ll: float, n_iter: int, max_iter: int) -> None:
    """Print EM algorithm results."""
    print(f"Best weighted log-likelihood: {ll:.10f}")
    print(f"Iterations: {n_iter} / {max_iter}")
    print(f"Convergence: {'Yes' if n_iter < max_iter else 'No (max iterations reached)'}")

def print_execution_time(em_time: float, qp_time: float = 0.0, total_time: float = 0.0, 
                        use_moment_matching: bool = False, method: str = "em",
                        lp_timing: Optional[Dict] = None) -> None:
    """Print execution time summary."""
    print_subsection_header("EXECUTION TIME")
    
    if method == "em":
        print(f"EM algorithm:          {em_time:>10.6f} seconds")
        if use_moment_matching and qp_time > 0:
            print(f"QP projection:         {qp_time:>10.6f} seconds")
            print(f"Total (EM + QP):       {total_time:>10.6f} seconds")
        else:
            print(f"Total:                 {total_time:>10.6f} seconds")
    elif method == "lp" and lp_timing is not None:
        print(f"Dictionary generation: {lp_timing.get('dict_generation', 0):>10.6f} seconds")
        print(f"Basis computation:     {lp_timing.get('basis_computation', 0):>10.6f} seconds")
        greedy_time = lp_timing.get('greedy_selection', 0)
        if greedy_time > 0:
            # Greedy LP method
            print(f"Greedy selection:      {greedy_time:>10.6f} seconds")
            n_lp_calls = lp_timing.get('n_lp_calls', 0)
            lp_total = lp_timing.get('lp_solving', 0)
            print(f"LP solving ({n_lp_calls} calls): {lp_total:>10.6f} seconds")
            if n_lp_calls > 0:
                avg_lp = lp_total / n_lp_calls
                print(f"  Average per LP call: {avg_lp:>10.6f} seconds")
            print(f"Total (LP+Greedy):     {lp_timing.get('total', total_time):>10.6f} seconds")
        else:
            # Simple LP method (no greedy selection)
            lp_total = lp_timing.get('lp_solving', 0)
            print(f"LP solving:            {lp_total:>10.6f} seconds")
            print(f"Total:                 {lp_timing.get('total', total_time):>10.6f} seconds")
    else:
        print(f"Total:                 {total_time:>10.6f} seconds")

def print_moment_matching_info(qp_info: Dict) -> None:
    """Print moment matching QP projection information."""
    print_subsection_header("MOMENT MATCHING QP PROJECTION")
    method_str = qp_info['method'].upper()
    if qp_info['method'] == 'hard':
        print(f"Method: {method_str} (moments matched exactly)")
    elif qp_info['method'] == 'soft':
        print(f"Method: {method_str} (moments approximately matched)")
        constraint_error = qp_info.get('constraint_error', 0.0)
        if constraint_error > 1e-4:
            print(f"  ⚠️  Warning: Large constraint error indicates moment matching may be infeasible")
    else:
        print(f"Method: {method_str}")
    constraint_error = qp_info.get('constraint_error', 0.0)
    print(f"Constraint error: {constraint_error:.6e}")
    if constraint_error > 1e-4:
        print(f"  ⚠️  Note: Constraint error > 1e-4 suggests infeasibility. "
              f"Higher moments (especially kurtosis) may have significant errors.")
    moment_names = ['M0', 'M1', 'M2', 'M3', 'M4']
    print("Moment errors:")
    for i, name in enumerate(moment_names):
        err = qp_info['moment_errors'][i]
        print(f"  {name}: {err:+.6e}")
        # Highlight M4 (kurtosis-related) error if large
        if i == 4 and abs(err) > 1e-3:
            print(f"       ⚠️  Large M4 error may result in significant kurtosis error")

def print_moment_matching_info_lp(lp_info: Dict) -> None:
    """Print moment matching information for LP method."""
    print_subsection_header("MOMENT MATCHING (LP CONSTRAINTS)")
    method_str = lp_info.get('moment_method', 'none').upper()
    if method_str == 'INEQUALITY':
        tolerance = lp_info.get('moment_tolerance', 0.0)
        print(f"Method: {method_str} (moments within relative tolerance: ±{tolerance*100:.2f}%)")
    elif method_str == 'HARD':
        print(f"Method: {method_str} (moments matched exactly)")
    elif method_str == 'SOFT':
        print(f"Method: {method_str} (moments approximately matched)")
    else:
        print(f"Method: {method_str}")
    if 'moment_constraint_error' in lp_info:
        print(f"Max absolute error: {lp_info['moment_constraint_error']:.6e}")
    if 'moment_tolerance' in lp_info:
        tolerance = lp_info['moment_tolerance']
        print(f"Relative tolerance: ±{tolerance*100:.2f}%")
    if 'moment_errors' in lp_info:
        moment_names = ['M0', 'M1', 'M2', 'M3', 'M4']
        print("Moment errors:")
        for i, name in enumerate(moment_names):
            err = lp_info['moment_errors'][i]
            print(f"  {name}: {err:+.6e}")

def calc_relative_error(true_val: float, approx_val: float) -> float:
    """
    Calculate relative error in percentage.
    
    Returns:
    --------
    float
        Relative error as percentage: 100 * (approx - true) / |true|
        Returns inf if true_val is near zero and approx_val is not
    """
    if abs(true_val) < EPSILON:
        return float('inf') if abs(approx_val) > EPSILON else 0.0
    return 100.0 * (approx_val - true_val) / abs(true_val)

def print_statistics_comparison(stats_true: Dict, stats_hat: Dict) -> None:
    """Print PDF statistics comparison table."""
    print_section_header("PDF STATISTICS COMPARISON")
    
    rel_mean = calc_relative_error(stats_true['mean'], stats_hat['mean'])
    rel_std = calc_relative_error(stats_true['std'], stats_hat['std'])
    rel_skewness = calc_relative_error(stats_true['skewness'], stats_hat['skewness'])
    rel_kurtosis = calc_relative_error(stats_true['kurtosis'], stats_hat['kurtosis'])
    
    total_width = COL_STAT_WIDTH + COL_NUM_WIDTH * 2 + COL_REL_WIDTH
    rel_format = lambda x: f"{x:.4f}%" if not np.isinf(x) else "inf"
    
    # Header
    print(f"{'Statistic':<{COL_STAT_WIDTH}} {'True PDF':>{COL_NUM_WIDTH}} {'GMM Approx PDF':>{COL_NUM_WIDTH}} {'Rel Error (%)':>{COL_REL_WIDTH}}")
    print("-" * total_width)
    
    # Data rows
    print(f"{'Mean':<{COL_STAT_WIDTH}} {stats_true['mean']:>{COL_NUM_WIDTH}.6f} {stats_hat['mean']:>{COL_NUM_WIDTH}.6f} {rel_format(rel_mean):>{COL_REL_WIDTH}}")
    print(f"{'Std Dev':<{COL_STAT_WIDTH}} {stats_true['std']:>{COL_NUM_WIDTH}.6f} {stats_hat['std']:>{COL_NUM_WIDTH}.6f} {rel_format(rel_std):>{COL_REL_WIDTH}}")
    print(f"{'Skewness':<{COL_STAT_WIDTH}} {stats_true['skewness']:>{COL_NUM_WIDTH}.6f} {stats_hat['skewness']:>{COL_NUM_WIDTH}.6f} {rel_format(rel_skewness):>{COL_REL_WIDTH}}")
    print(f"{'Kurtosis':<{COL_STAT_WIDTH}} {stats_true['kurtosis']:>{COL_NUM_WIDTH}.6f} {stats_hat['kurtosis']:>{COL_NUM_WIDTH}.6f} {rel_format(rel_kurtosis):>{COL_REL_WIDTH}}")

def print_gmm_parameters(params: 'GMM1DParams', threshold: float = 1e-8) -> None:
    """
    Print GMM component parameters.
    
    Parameters:
    -----------
    params : GMM1DParams
        GMM parameters
    threshold : float
        Minimum weight threshold for display (default: 1e-8)
        Components with weight below this threshold are not displayed
    """
    print_section_header("GMM PARAMETERS")
    K = len(params.pi)
    
    # Filter non-zero components
    nonzero_mask = params.pi > threshold
    n_nonzero = np.sum(nonzero_mask)
    
    if n_nonzero < K:
        print(f"Number of components: {K} (showing {n_nonzero} non-zero components)")
    else:
        print(f"Number of components: {K}")
    
    print("\nComponent details:")
    component_idx = 0
    for k in range(K):
        if params.pi[k] > threshold:
            component_idx += 1
            print(f"  Component {component_idx}: π={params.pi[k]:.8f}, μ={params.mu[k]:.8f}, σ={np.sqrt(params.var[k]):.8f}")

def print_plot_output(output_path: str) -> None:
    """Print plot output information."""
    print_section_header("PLOT OUTPUT")
    print(f"Plot saved: {output_path}.png")
    print("="*SECTION_WIDTH)

# ============================================================
# 3) Configuration and setup functions
# ============================================================

def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file
    
    Returns:
    --------
    dict
        Configuration dictionary with default values applied
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Using default parameters.")
        config = {}
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        raise
    
    # Apply defaults
    return {
        "mu_x": config.get("mu_x", DEFAULT_MU_X),
        "sigma_x": config.get("sigma_x", DEFAULT_SIGMA_X),
        "mu_y": config.get("mu_y", DEFAULT_MU_Y),
        "sigma_y": config.get("sigma_y", DEFAULT_SIGMA_Y),
        "rho": config.get("rho", DEFAULT_RHO),
        "z_range": config.get("z_range", DEFAULT_Z_RANGE),
        "z_npoints": config.get("z_npoints", DEFAULT_Z_NPOINTS),
        "K": config.get("K", DEFAULT_K),
        "L": config.get("L"),  # LP method parameter (optional, defaults handled in main.py)
        "max_iter": config.get("max_iter", DEFAULT_MAX_ITER),
        "tol": config.get("tol", DEFAULT_TOL),
        "reg_var": config.get("reg_var", DEFAULT_REG_VAR),
        "n_init": config.get("n_init", DEFAULT_N_INIT),
        "seed": config.get("seed", DEFAULT_SEED),
        "init": config.get("init", DEFAULT_INIT),
        "use_moment_matching": config.get("use_moment_matching", DEFAULT_USE_MOMENT_MATCHING),
        "qp_mode": config.get("qp_mode", DEFAULT_QP_MODE),
        "soft_lambda": config.get("soft_lambda", DEFAULT_SOFT_LAMBDA),
        "method": config.get("method", "em"),  # "em" or "lp"
        "objective_mode": config.get("objective_mode", "pdf"),  # "pdf" or "moments" (for LP method)
        "lp_params": config.get("lp_params", {}),
        "output_path": config.get("output_path", DEFAULT_OUTPUT_PATH),
        "show_grid_points": config.get("show_grid_points", DEFAULT_SHOW_GRID_POINTS),
        "max_grid_points_display": config.get("max_grid_points_display", DEFAULT_MAX_GRID_POINTS_DISPLAY),
        "_raw_config": config  # Keep original for init_params override
    }

def prepare_init_params(config: Dict, init: str, mu_x: float, sigma_x: float, 
                       mu_y: float, sigma_y: float, rho: float) -> Optional[Dict]:
    """
    Prepare initialization parameters for WQMI method.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    init : str
        Initialization method name
    mu_x, sigma_x, mu_y, sigma_y, rho : float
        Bivariate normal distribution parameters
    
    Returns:
    --------
    dict or None
        Initialization parameters if init == "wqmi", None otherwise
    """
    if init != "wqmi":
        return None
    
    init_params = {
        "mu_x": mu_x,
        "var_x": sigma_x**2,
        "mu_y": mu_y,
        "var_y": sigma_y**2,
        "rho": rho
    }
    
    # Allow override from config
    raw_config = config.get("_raw_config", {})
    if "init_params" in raw_config:
        init_params.update(raw_config["init_params"])
    
    return init_params

