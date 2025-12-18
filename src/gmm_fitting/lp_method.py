"""
LP Method - Simple Linear Programming for GMM Fitting

This module implements a simple linear programming (LP) based approach to fit a 1D Gaussian
Mixture Model (GMM) to a PDF, using L∞ norm minimization for PDF error.

Main components:
1. Dictionary generation (Gaussian basis functions)
2. Basis matrix computation (PDF and CDF)
3. LP solver for L∞ minimization
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, Literal
from scipy.special import ndtr
from scipy.optimize import linprog
from scipy import sparse

from .gmm_utils import (
    normal_pdf,
    normalize_pdf_on_grid,
    compute_pdf_statistics,
    pdf_to_cdf_trapz,
    compute_gmm_moments_from_weights,
    compute_component_raw_moments,
    compute_pdf_raw_moments,
    VAR_FLOOR,
)


def build_gaussian_dictionary_simple(
    z: np.ndarray,
    f: np.ndarray,
    K: int,
    L: int,
    sigma_min_scale: float = 0.1,
    sigma_max_scale: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Build a simple dictionary of Gaussian basis functions.
    
    Divides the range [μ - 3σ, μ + 3σ] into K equal segments,
    uses the center of each segment as μ, and generates L different
    sigma values for each segment.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,)
    K : int
        Number of segments (also number of mean locations)
    L : int
        Number of sigma levels per segment
    sigma_min_scale : float
        Minimum sigma scale relative to true std dev (default: 0.1)
    sigma_max_scale : float
        Maximum sigma scale relative to true std dev (default: 3.0)
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "mus": np.ndarray (K * L,) - mean locations
        - "sigmas": np.ndarray (K * L,) - standard deviations
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    if K < 1 or L < 1:
        raise ValueError("K and L must be >= 1")
    
    # Compute true PDF statistics
    mean_true = np.trapezoid(z * f, z)
    var_true = np.trapezoid((z - mean_true)**2 * f, z)
    sigma_z = np.sqrt(max(var_true, 1e-10))  # Avoid zero variance
    
    # Define range: [μ - 3σ, μ + 3σ]
    z_min = mean_true - 3 * sigma_z
    z_max = mean_true + 3 * sigma_z
    
    # Divide range into K equal segments
    # Segment boundaries: z_min, z_min + Δ, z_min + 2Δ, ..., z_max
    # where Δ = (z_max - z_min) / K
    segment_width = (z_max - z_min) / K
    
    # Mean locations: center of each segment
    mu_candidates = np.array([
        z_min + (i + 0.5) * segment_width
        for i in range(K)
    ])
    
    # Generate sigma candidates (logarithmic spacing)
    sigma_candidates = _generate_sigma_candidates(
        sigma_z, L, sigma_min_scale, sigma_max_scale
    )
    
    # Create Cartesian product: all combinations of mu and sigma
    # For each segment (K segments), generate L different sigmas
    mus = np.repeat(mu_candidates, L)
    sigmas = np.tile(sigma_candidates, K)
    
    return {
        "mus": mus,
        "sigmas": sigmas
    }


def build_gaussian_dictionary(
    z: np.ndarray,
    f: np.ndarray,
    J: int,
    L: int,
    mu_mode: str = "quantile",
    sigma_min_scale: float = 0.1,
    sigma_max_scale: float = 3.0,
    tail_focus: str = "none",
    tail_alpha: float = 1.0,
    quantile_levels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a dictionary of Gaussian basis functions.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,)
    J : int
        Number of mean locations
    L : int
        Number of sigma levels per mean location
    mu_mode : str
        Mode for placing means: "uniform" or "quantile" (default: "quantile")
    sigma_min_scale : float
        Minimum sigma scale relative to true std dev (default: 0.1)
    sigma_max_scale : float
        Maximum sigma scale relative to true std dev (default: 3.0)
    tail_focus : str
        Tail emphasis mode: "none", "right", "left", or "both" (default: "none")
    tail_alpha : float
        Tail emphasis strength (>= 1.0, default: 1.0). Larger values emphasize tails more.
    quantile_levels : Optional[np.ndarray]
        Custom quantile levels. If provided, tail_focus and tail_alpha are ignored.
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "mus": np.ndarray (J * L,) - mean locations
        - "sigmas": np.ndarray (J * L,) - standard deviations
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    if J < 1 or L < 1:
        raise ValueError("J and L must be >= 1")
    if mu_mode not in ["uniform", "quantile"]:
        raise ValueError(f"mu_mode must be 'uniform' or 'quantile', got '{mu_mode}'")
    if tail_focus not in ["none", "right", "left", "both"]:
        raise ValueError(f"tail_focus must be 'none', 'right', 'left', or 'both', got '{tail_focus}'")
    
    # Clip tail_alpha to >= 1.0
    tail_alpha = max(tail_alpha, 1.0)
    
    # Compute true PDF statistics
    mean_true = np.trapezoid(z * f, z)
    var_true = np.trapezoid((z - mean_true)**2 * f, z)
    sigma_z = np.sqrt(max(var_true, 1e-10))  # Avoid zero variance
    
    # Generate mean locations
    if mu_mode == "uniform":
        # Uniform spacing in [z_min, z_max]
        z_min = z.min()
        z_max = z.max()
        mu_candidates = np.linspace(z_min, z_max, J)
    else:  # quantile mode
        # Compute CDF with monotonicity guarantee
        F = pdf_to_cdf_trapz(z, f)
        F = np.maximum.accumulate(F)
        if F[-1] <= 0:
            raise ValueError("CDF integral is non-positive")
        F /= F[-1]
        
        # Generate quantile levels
        if quantile_levels is not None:
            # Use custom quantile levels
            p = np.asarray(quantile_levels)
        else:
            # Generate uniform base levels
            u = np.array([(j + 0.5) / J for j in range(J)])
            
            # Apply tail focus transformation
            if tail_focus == "none":
                p = u
            elif tail_focus == "right":
                p = 1.0 - (1.0 - u)**tail_alpha
            elif tail_focus == "left":
                p = u**tail_alpha
            else:  # both
                p = 0.5 + np.sign(u - 0.5) * np.abs(u - 0.5)**tail_alpha
        
        # Clip quantile levels to (eps, 1-eps)
        eps = 1e-6
        p = np.clip(p, eps, 1.0 - eps)
        
        # Interpolate to find mu_j such that F(mu_j) = p[j]
        mu_candidates = np.interp(p, F, z)
    
    # Generate sigma candidates (logarithmic spacing)
    sigma_candidates = _generate_sigma_candidates(
        sigma_z, L, sigma_min_scale, sigma_max_scale
    )
    
    # Create Cartesian product: all combinations of mu and sigma
    # For each mean (J means), generate L different sigmas
    mus = np.repeat(mu_candidates, L)
    sigmas = np.tile(sigma_candidates, J)
    
    return {
        "mus": mus,
        "sigmas": sigmas
    }


def compute_basis_matrices(
    z: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute basis matrices for PDF and CDF.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    mus : np.ndarray
        Mean values for each basis, shape (m,)
    sigmas : np.ndarray
        Standard deviations for each basis, shape (m,)
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "Phi_pdf": np.ndarray (N, m) where Phi_pdf[i,j] = N(z_i; mu_j, sigma_j^2)
        - "Phi_cdf": np.ndarray (N, m) where Phi_cdf[i,j] = Φ((z_i-mu_j)/sigma_j)
    """
    z = np.asarray(z)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    
    if len(mus) != len(sigmas):
        raise ValueError("mus and sigmas must have the same length")
    
    N = len(z)
    m = len(mus)
    
    # Compute PDF basis matrix
    Phi_pdf = np.zeros((N, m))
    for j in range(m):
        var_j = sigmas[j]**2
        Phi_pdf[:, j] = normal_pdf(z, mus[j], var_j)
    
    # Compute CDF basis matrix
    Phi_cdf = np.zeros((N, m))
    for j in range(m):
        Phi_cdf[:, j] = ndtr((z - mus[j]) / sigmas[j])
    
    return {
        "Phi_pdf": Phi_pdf,
        "Phi_cdf": Phi_cdf
    }


def _generate_sigma_candidates(
    sigma_z: float,
    L: int,
    sigma_min_scale: float = 0.1,
    sigma_max_scale: float = 3.0,
) -> np.ndarray:
    """
    Generate sigma candidates using logarithmic spacing.
    
    Parameters:
    -----------
    sigma_z : float
        True standard deviation of the PDF
    L : int
        Number of sigma levels
    sigma_min_scale : float
        Minimum sigma scale relative to sigma_z (default: 0.1)
    sigma_max_scale : float
        Maximum sigma scale relative to sigma_z (default: 3.0)
    
    Returns:
    --------
    np.ndarray
        Sigma candidates, shape (L,)
    """
    sigma_min = sigma_min_scale * sigma_z
    sigma_max = sigma_max_scale * sigma_z
    return np.logspace(
        np.log10(max(sigma_min, 1e-10)),
        np.log10(sigma_max),
        L
    )


# Use normal_pdf from gmm_utils instead


def solve_lp_pdf_linf(
    Phi_pdf_sub: np.ndarray,
    f: np.ndarray,
    solver: str = "highs",
) -> Dict:
    """
    Solve LP problem for L∞ minimization of PDF error only.
    
    Minimizes: t_pdf
    Subject to:
        -t_pdf <= sum(w_j * Phi_pdf[:,j]) - f <= t_pdf
        w_j >= 0, sum(w_j) = 1
        t_pdf >= 0
    
    Parameters:
    -----------
    Phi_pdf_sub : np.ndarray
        PDF basis matrix, shape (N, s)
    f : np.ndarray
        True PDF values, shape (N,)
    solver : str
        LP solver method (default: "highs")
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "w": np.ndarray (s,) - weights
        - "t_pdf": float - PDF error bound
        - "objective": float - objective value
        - "status": int - solver status (0 = success)
        - "message": str - solver message
    """
    N, s = Phi_pdf_sub.shape
    
    if len(f) != N:
        raise ValueError("f must have length N")
    
    # Variables: [w_1, ..., w_s, t_pdf]
    n_vars = s + 1
    
    # Objective: minimize t_pdf
    c = np.zeros(n_vars)
    c[s] = 1.0  # coefficient for t_pdf
    
    # Constraints:
    # 1. PDF constraints: sum(w_j * Phi_pdf[i,j]) - t_pdf <= f[i]
    #    and: -sum(w_j * Phi_pdf[i,j]) - t_pdf <= -f[i]
    # 2. Sum constraint: sum(w_j) = 1
    # 3. Non-negativity: w_j >= 0, t_pdf >= 0
    
    # Build constraint matrix A_ub and b_ub for inequalities
    # Total: 2*N (PDF) inequality constraints
    n_ineq = 2 * N
    
    # Use sparse matrix if large (original condition)
    # Use sparse matrices only for very large problems where density is low
    # For dense matrices (100% density), dense format is faster
    # Condition: N * s > 100000 ensures sparse matrices are only used for large problems
    use_sparse = N * s > 100000
    
    if use_sparse:
        # Build sparse matrix efficiently using COO format
        # For rows 0 to N-1: Phi_pdf @ w - t_pdf <= f
        row_indices = np.repeat(np.arange(N), s + 1)
        col_indices = np.concatenate([np.tile(np.arange(s), N), np.repeat([s], N)])
        data = np.concatenate([Phi_pdf_sub.flatten(), np.full(N, -1.0)])
        
        # For rows N to 2*N-1: -Phi_pdf @ w - t_pdf <= -f
        row_indices2 = np.repeat(np.arange(N, 2*N), s + 1)
        col_indices2 = np.concatenate([np.tile(np.arange(s), N), np.repeat([s], N)])
        data2 = np.concatenate([-Phi_pdf_sub.flatten(), np.full(N, -1.0)])
        
        # Combine all constraints
        row_indices_all = np.concatenate([row_indices, row_indices2])
        col_indices_all = np.concatenate([col_indices, col_indices2])
        data_all = np.concatenate([data, data2])
        A_ub = sparse.coo_matrix((data_all, (row_indices_all, col_indices_all)), shape=(n_ineq, n_vars))
        A_ub = A_ub.tocsr()
        
        # Equality constraint: sum(w_j) = 1
        A_eq = sparse.coo_matrix((np.ones(s), (np.zeros(s), np.arange(s))), shape=(1, n_vars))
        A_eq = A_eq.tocsr()
        
        b_ub = np.zeros(n_ineq)
        b_ub[:N] = f
        b_ub[N:2*N] = -f
        b_eq = np.array([1.0])
    else:
        A_ub = np.zeros((n_ineq, n_vars))
        # PDF constraints: Phi_pdf @ w - t_pdf <= f
        A_ub[:N, :s] = Phi_pdf_sub
        A_ub[:N, s] = -1.0
        
        # PDF constraints: -Phi_pdf @ w - t_pdf <= -f
        A_ub[N:2*N, :s] = -Phi_pdf_sub
        A_ub[N:2*N, s] = -1.0
        
        # Equality constraint: sum(w_j) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :s] = 1.0
        
        b_ub = np.zeros(n_ineq)
        b_ub[:N] = f
        b_ub[N:2*N] = -f
        b_eq = np.array([1.0])
    
    # Bounds: w_j >= 0, t_pdf >= 0
    bounds = [(0, None)] * n_vars
    
    # Solve LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=solver
    )
    
    if result.success:
        w = result.x[:s]
        t_pdf = result.x[s]
        objective = result.fun
        
        return {
            "w": w,
            "t_pdf": t_pdf,
            "objective": objective,
            "status": 0,
            "message": result.message if hasattr(result, 'message') else ""
        }
    else:
        # Solver failed
        w = np.zeros(s)
        t_pdf = np.inf
        objective = np.inf
    
    return {
        "w": w,
        "t_pdf": t_pdf,
        "objective": objective,
            "status": -1,
            "message": result.message if hasattr(result, 'message') else "LP solver failed"
    }


# Use pdf_to_cdf_trapz and compute_gmm_moments_from_weights from gmm_utils instead


def solve_lp_pdf_moments_linf(
    Phi_pdf_sub: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    f: np.ndarray,
    target_mean: float,
    target_variance: float,
    target_skewness: float,
    target_kurtosis: float,
    lambda_pdf: float = 1.0,
    lambda_mean: float = 1.0,
    lambda_variance: float = 1.0,
    lambda_skewness: float = 1.0,
    lambda_kurtosis: float = 1.0,
    solver: str = "highs",
    pdf_tolerance: float = 1e-6,
    max_moment_iter: int = 5,
    moment_tolerance: float = 1e-6,
) -> Dict:
    """
    Solve LP problem for L∞ minimization of moment relative errors with PDF constraint.
    
    Minimizes: λ_mean * t_mean + λ_variance * t_var + λ_skewness * t_skew + λ_kurtosis * t_kurt
    Subject to:
        -t_pdf <= sum(w_j * Phi_pdf[:,j]) - f <= t_pdf  (PDF constraint)
        -t_mean * |mean_target| <= mean_mixture - mean_target <= t_mean * |mean_target|
        -t_var * |var_target| <= var_mixture - var_target <= t_var * |var_target|
        -t_skew * |skew_target| <= skew_mixture - skew_target <= t_skew * |skew_target|
        -t_kurt * |kurt_target| <= kurt_mixture - kurt_target <= t_kurt * |kurt_target|
        w_j >= 0, sum(w_j) = 1
        t_pdf <= pdf_tolerance (hard constraint)
        t_mean >= 0, t_var >= 0, t_skew >= 0, t_kurt >= 0
    
    Parameters:
    -----------
    Phi_pdf_sub : np.ndarray
        PDF basis matrix for selected subset, shape (N, s)
    mus : np.ndarray
        Component means, shape (s,)
    sigmas : np.ndarray
        Component standard deviations, shape (s,)
    f : np.ndarray
        True PDF values, shape (N,)
    target_mean : float
        Target mean value
    target_variance : float
        Target variance value
    target_skewness : float
        Target skewness value
    target_kurtosis : float
        Target kurtosis value
    lambda_pdf : float
        Weight for PDF error term (not used in objective, but for constraint)
    lambda_mean : float
        Weight for mean error term (default: 1.0)
    lambda_variance : float
        Weight for variance error term (default: 1.0)
    lambda_skewness : float
        Weight for skewness error term (default: 1.0)
    lambda_kurtosis : float
        Weight for kurtosis error term (default: 1.0)
    solver : str
        LP solver method (default: "highs")
    pdf_tolerance : float
        Maximum allowed PDF error (hard constraint, default: 1e-6)
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "w": np.ndarray (s,) - weights
        - "t_pdf": float - PDF error bound
        - "t_mean": float - mean relative error bound
        - "t_var": float - variance relative error bound
        - "t_skew": float - skewness relative error bound
        - "t_kurt": float - kurtosis relative error bound
        - "objective": float - objective value
        - "status": int - solver status (0 = success)
        - "message": str - solver message
        - "moment_errors": dict - actual moment errors
    """
    N, s = Phi_pdf_sub.shape
    
    if len(mus) != s or len(sigmas) != s:
        raise ValueError("mus and sigmas must have length s")
    if len(f) != N:
        raise ValueError("f must have length N")
    
    # Variables: [w_1, ..., w_s, t_pdf, t_mean, t_var, t_skew, t_kurt]
    n_vars = s + 5
    
    # Objective: minimize λ_mean * t_mean + λ_var * t_var + λ_skew * t_skew + λ_kurt * t_kurt
    c = np.zeros(n_vars)
    c[s + 1] = lambda_mean      # coefficient for t_mean
    c[s + 2] = lambda_variance   # coefficient for t_var
    c[s + 3] = lambda_skewness   # coefficient for t_skew
    c[s + 4] = lambda_kurtosis   # coefficient for t_kurt
    
    # Compute component raw moments for linearization
    vars_component = sigmas**2
    moment_matrix = compute_component_raw_moments(mus, vars_component)  # shape (5, s)
    
    # Compute normalization factors for relative errors
    abs_mean = abs(target_mean) if abs(target_mean) > 1e-10 else 1.0
    abs_var = abs(target_variance) if abs(target_variance) > 1e-10 else 1.0
    abs_skew = abs(target_skewness) if abs(target_skewness) > 1e-10 else 1.0
    abs_kurt = abs(target_kurtosis) if abs(target_kurtosis) > 1e-10 else 1.0
    
    # Build constraint matrix
    # PDF constraints: 2*N
    # Mean constraints: 2 (upper and lower bounds)
    # Variance constraints: 2 (using E[X²] - E[X]² approximation)
    # Skewness constraints: 2 (using linear approximation with current mean/std)
    # Kurtosis constraints: 2 (using linear approximation with current mean/std)
    # Note: Skewness and kurtosis are highly nonlinear, so we use iterative approximation
    
    # PDF constraints: 2*N
    n_pdf_ineq = 2 * N
    # Mean constraints: 2
    n_mean_ineq = 2
    # Variance constraints: 2 (linear approximation using E[X²] and E[X])
    n_var_ineq = 2
    # Skewness constraints: 2 (will be updated iteratively)
    n_skew_ineq = 2
    # Kurtosis constraints: 2 (will be updated iteratively)
    n_kurt_ineq = 2
    # Total inequality constraints
    n_ineq = n_pdf_ineq + n_mean_ineq + n_var_ineq + n_skew_ineq + n_kurt_ineq
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # PDF constraints: Phi_pdf @ w - t_pdf <= f
    A_ub[:N, :s] = Phi_pdf_sub
    A_ub[:N, s] = -1.0
    b_ub[:N] = f
    
    # PDF constraints: -Phi_pdf @ w - t_pdf <= -f
    A_ub[N:2*N, :s] = -Phi_pdf_sub
    A_ub[N:2*N, s] = -1.0
    b_ub[N:2*N] = -f
    
    # Mean constraints: moment_matrix[1] @ w - t_mean * abs_mean <= target_mean
    # Upper bound: moment_matrix[1] @ w <= target_mean + t_mean * abs_mean
    A_ub[2*N, :s] = moment_matrix[1, :]  # E[X] = sum(w_j * mu_j)
    A_ub[2*N, s + 1] = -abs_mean
    b_ub[2*N] = target_mean
    
    # Lower bound: moment_matrix[1] @ w >= target_mean - t_mean * abs_mean
    # Standard form: -moment_matrix[1] @ w <= -target_mean + t_mean * abs_mean
    A_ub[2*N + 1, :s] = -moment_matrix[1, :]
    A_ub[2*N + 1, s + 1] = -abs_mean
    b_ub[2*N + 1] = -target_mean
    
    # Variance constraints: Var[X] = E[X²] - E[X]² ≈ E[X²] - target_mean² (linear approximation)
    # Upper bound: moment_matrix[2] @ w - target_mean² <= target_variance + t_var * abs_var
    A_ub[2*N + 2, :s] = moment_matrix[2, :]  # E[X²] = sum(w_j * (mu_j² + sigma_j²))
    A_ub[2*N + 2, s + 2] = -abs_var
    b_ub[2*N + 2] = target_variance + target_mean**2
    
    # Lower bound: moment_matrix[2] @ w - target_mean² >= target_variance - t_var * abs_var
    # Standard form: -moment_matrix[2] @ w <= -target_variance + target_mean² - t_var * abs_var
    A_ub[2*N + 3, :s] = -moment_matrix[2, :]
    A_ub[2*N + 3, s + 2] = -abs_var
    b_ub[2*N + 3] = -target_variance + target_mean**2
    
    # Skewness constraints: will be updated iteratively with current mean/std
    # Placeholder constraints (will be updated in iteration loop)
    # Upper bound: skew_mixture <= target_skewness + t_skew * abs_skew
    A_ub[2*N + 4, :s] = 0.0  # Will be updated iteratively
    A_ub[2*N + 4, s + 3] = -abs_skew
    b_ub[2*N + 4] = target_skewness
    
    # Lower bound: skew_mixture >= target_skewness - t_skew * abs_skew
    # Standard form: -skew_mixture <= -target_skewness + t_skew * abs_skew
    A_ub[2*N + 5, :s] = 0.0  # Will be updated iteratively
    A_ub[2*N + 5, s + 3] = -abs_skew
    b_ub[2*N + 5] = -target_skewness
    
    # Kurtosis constraints: will be updated iteratively with current mean/std
    # Placeholder constraints (will be updated in iteration loop)
    # Upper bound: kurt_mixture <= target_kurtosis + t_kurt * abs_kurt
    A_ub[2*N + 6, :s] = 0.0  # Will be updated iteratively
    A_ub[2*N + 6, s + 4] = -abs_kurt
    b_ub[2*N + 6] = target_kurtosis
    
    # Lower bound: kurt_mixture >= target_kurtosis - t_kurt * abs_kurt
    # Standard form: -kurt_mixture <= -target_kurtosis + t_kurt * abs_kurt
    A_ub[2*N + 7, :s] = 0.0  # Will be updated iteratively
    A_ub[2*N + 7, s + 4] = -abs_kurt
    b_ub[2*N + 7] = -target_kurtosis
    
    # PDF tolerance constraint: t_pdf <= pdf_tolerance
    # This is a bound constraint, handled separately
    
    # Equality constraint: sum(w_j) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :s] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_j >= 0, t_pdf >= 0, t_mean >= 0, t_var >= 0, t_skew >= 0, t_kurt >= 0
    # t_pdf <= pdf_tolerance
    bounds = [(0, None)] * n_vars
    bounds[s] = (0, pdf_tolerance)  # t_pdf bound
    
    # Solve LP iteratively to handle nonlinear constraints (skewness and kurtosis)
    max_iter = max_moment_iter
    w_current = None
    tolerance = moment_tolerance
    iter_info = None  # Will be set if solver fails during iteration
    mean_current = None
    var_current = None
    std_current = None
    skew_current = None
    kurt_current = None
    
    for iter_num in range(max_iter):
        # Update constraints if we have a current solution
        if w_current is not None:
            # Compute current moments for linearization
            mean_current, var_current, skew_current, kurt_current = compute_gmm_moments_from_weights(
                w_current, mus, sigmas
            )
            std_current = np.sqrt(max(var_current, VAR_FLOOR))
            
            # Update variance constraint with current mean
            b_ub[2*N + 2] = target_variance + mean_current**2
            b_ub[2*N + 3] = -target_variance + mean_current**2
            
            # Update skewness constraints using linear approximation
            # Skewness = E[(X-μ)³] / σ³
            # Linearize around current mean and std
            # For each component j: contribution to skewness depends on (mu_j - mean_current) and sigma_j
            mu_centered = mus - mean_current
            vars_component = sigmas**2
            
            # Third central moment for each component: E[(X_j - μ)³] = (mu_j - μ)³ + 3*sigma_j²*(mu_j - μ)
            mu3_component = mu_centered**3 + 3 * vars_component * mu_centered
            # Skewness contribution: mu3_component / std_current^3
            skew_coeffs = mu3_component / (std_current**3) if std_current > VAR_FLOOR else np.zeros(s)
            
            # Upper bound: skew_mixture <= target_skewness + t_skew * abs_skew
            A_ub[2*N + 4, :s] = skew_coeffs
            b_ub[2*N + 4] = target_skewness
            
            # Lower bound: skew_mixture >= target_skewness - t_skew * abs_skew
            A_ub[2*N + 5, :s] = -skew_coeffs
            b_ub[2*N + 5] = -target_skewness
            
            # Update kurtosis constraints using linear approximation
            # Kurtosis = E[(X-μ)⁴] / σ⁴ - 3
            # Fourth central moment for each component: E[(X_j - μ)⁴] = (mu_j - μ)⁴ + 6*sigma_j²*(mu_j - μ)² + 3*sigma_j⁴
            mu4_component = mu_centered**4 + 6 * vars_component * mu_centered**2 + 3 * vars_component**2
            # Kurtosis contribution: mu4_component / std_current^4 - 3 (but we linearize around current)
            kurt_coeffs = mu4_component / (std_current**4) if std_current > VAR_FLOOR else np.zeros(s)
            
            # Upper bound: kurt_mixture <= target_kurtosis + t_kurt * abs_kurt
            A_ub[2*N + 6, :s] = kurt_coeffs
            b_ub[2*N + 6] = target_kurtosis + 3.0  # Add 3 because we use excess kurtosis
            
            # Lower bound: kurt_mixture >= target_kurtosis - t_kurt * abs_kurt
            A_ub[2*N + 7, :s] = -kurt_coeffs
            b_ub[2*N + 7] = -target_kurtosis - 3.0  # Subtract 3 because we use excess kurtosis
        else:
            # First iteration: use initial approximation
            # For skewness and kurtosis, use raw moments as initial approximation
            # This is a rough approximation, will be refined in subsequent iterations
            mu_centered_init = mus - target_mean
            vars_component = sigmas**2
            std_init = np.sqrt(max(target_variance, VAR_FLOOR))
            
            # Initial skewness approximation
            mu3_component_init = mu_centered_init**3 + 3 * vars_component * mu_centered_init
            skew_coeffs_init = mu3_component_init / (std_init**3) if std_init > VAR_FLOOR else np.zeros(s)
            A_ub[2*N + 4, :s] = skew_coeffs_init
            b_ub[2*N + 4] = target_skewness
            A_ub[2*N + 5, :s] = -skew_coeffs_init
            b_ub[2*N + 5] = -target_skewness
            
            # Initial kurtosis approximation
            mu4_component_init = mu_centered_init**4 + 6 * vars_component * mu_centered_init**2 + 3 * vars_component**2
            kurt_coeffs_init = mu4_component_init / (std_init**4) if std_init > VAR_FLOOR else np.zeros(s)
            A_ub[2*N + 6, :s] = kurt_coeffs_init
            b_ub[2*N + 6] = target_kurtosis + 3.0
            A_ub[2*N + 7, :s] = -kurt_coeffs_init
            b_ub[2*N + 7] = -target_kurtosis - 3.0
        
        # Solve LP with timeout protection
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=solver
        )
        
        if not result.success:
            # Store iteration info for error message
            iter_info = {
                "iteration": iter_num + 1,
                "max_iterations": max_iter,
                "solver_message": result.message if hasattr(result, 'message') else "Unknown error"
            }
            break
        
        w_new = result.x[:s]
        
        # Compute moments for convergence check and next iteration
        mean_new, var_new, skew_new, kurt_new = compute_gmm_moments_from_weights(
            w_new, mus, sigmas
        )
        std_new = np.sqrt(max(var_new, VAR_FLOOR))
        
        # Check convergence
        if w_current is not None:
            weight_change = np.max(np.abs(w_new - w_current))
            # Also check moment convergence for better accuracy (especially kurtosis)
            kurt_change = abs(kurt_new - kurt_current)
            
            if weight_change < tolerance:
                # Additional check: if kurtosis change is small relative to target, consider converged
                if kurt_change < abs_kurt * 1e-3:  # Relative tolerance for kurtosis
                    break
            
            # Update current moments for next iteration
            mean_current = mean_new
            var_current = var_new
            std_current = std_new
            skew_current = skew_new
            kurt_current = kurt_new
        else:
            # First iteration: initialize moments
            mean_current = mean_new
            var_current = var_new
            std_current = std_new
            skew_current = skew_new
            kurt_current = kurt_new
        
        w_current = w_new.copy()
        
        if result.success:
            w = result.x[:s]
            t_pdf = result.x[s]
            t_mean = result.x[s + 1]
            t_var = result.x[s + 2]
            t_skew = result.x[s + 3]
            t_kurt = result.x[s + 4]
            objective = result.fun
            
            # Compute actual moment errors
            mean_actual, var_actual, skew_actual, kurt_actual = compute_gmm_moments_from_weights(
                w, mus, sigmas
            )
            
            moment_errors = {
                "mean": mean_actual - target_mean,
                "variance": var_actual - target_variance,
                "skewness": skew_actual - target_skewness,
                "kurtosis": kurt_actual - target_kurtosis,
                "mean_relative": (mean_actual - target_mean) / abs_mean if abs_mean > 1e-10 else 0.0,
                "variance_relative": (var_actual - target_variance) / abs_var if abs_var > 1e-10 else 0.0,
                "skewness_relative": (skew_actual - target_skewness) / abs_skew if abs_skew > 1e-10 else 0.0,
                "kurtosis_relative": (kurt_actual - target_kurtosis) / abs_kurt if abs_kurt > 1e-10 else 0.0,
            }
            
            return {
                "w": w,
                "t_pdf": t_pdf,
                "t_mean": t_mean,
                "t_var": t_var,
                "t_skew": t_skew,
                "t_kurt": t_kurt,
                "objective": objective,
                "status": 0,
                "message": result.message if hasattr(result, 'message') else "",
                "moment_errors": moment_errors
            }
        else:
            # Solver failed
            w = np.zeros(s)
            t_pdf = np.inf
            t_mean = np.inf
            t_var = np.inf
            t_skew = np.inf
            t_kurt = np.inf
            objective = np.inf
            
            # Build detailed error message
            error_msg = result.message if hasattr(result, 'message') else "LP solver failed"
            # Check if iter_info was set (only if loop broke due to failure)
            if 'iter_info' in locals() and iter_info is not None:
                error_msg += f" (failed at iteration {iter_info['iteration']}/{iter_info['max_iterations']})"
            
            return {
                "w": w,
                "t_pdf": t_pdf,
                "t_mean": t_mean,
                "t_var": t_var,
                "t_skew": t_skew,
                "t_kurt": t_kurt,
                "objective": objective,
                "status": -1,
                "message": error_msg,
                "moment_errors": {
                "mean": 0.0,
                "variance": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "mean_relative": 0.0,
                "variance_relative": 0.0,
                "skewness_relative": 0.0,
                "kurtosis_relative": 0.0,
            }
        }


def solve_lp_pdf_rawmoments_linf(
    Phi_pdf: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    z: np.ndarray,
    f: np.ndarray,
    pdf_tolerance: Optional[float],
    lambda_pdf: float,
    lambda_raw: Tuple[float, float, float, float],
    solver: str = "highs",
    objective_form: Literal["A", "B"] = "A",
) -> Dict:
    """
    Solve LP problem for L∞ minimization of PDF and raw moment errors.
    
    Variables: [w_1, ..., w_m, t_pdf, t_1, t_2, t_3, t_4]
    
    Objective form A: minimize λ_1*t_1 + λ_2*t_2 + λ_3*t_3 + λ_4*t_4
                      subject to t_pdf <= pdf_tolerance
    
    Objective form B: minimize λ_pdf*t_pdf + Σ λ_n*t_n
    
    Parameters:
    -----------
    Phi_pdf : np.ndarray
        PDF basis matrix, shape (N, m)
    mus : np.ndarray
        Component means, shape (m,)
    sigmas : np.ndarray
        Component standard deviations, shape (m,)
    z : np.ndarray
        Grid points, shape (N,) (for computing target raw moments)
    f : np.ndarray
        True PDF values, shape (N,)
    pdf_tolerance : Optional[float]
        Maximum PDF error (for form A). None means no upper bound.
    lambda_pdf : float
        Weight for PDF error term (for form B)
    lambda_raw : Tuple[float, float, float, float]
        Weights for raw moment errors (λ_1, λ_2, λ_3, λ_4)
    solver : str
        LP solver method (default: "highs")
    objective_form : Literal["A", "B"]
        Objective function form (default: "A")
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - "w": np.ndarray (m,) - weights
        - "t_pdf": float - PDF error bound
        - "t_raw": np.ndarray (4,) - raw moment error bounds
        - "objective": float - objective value
        - "status": int - solver status (0 = success)
        - "message": str - solver message
        - "diagnostics": dict - diagnostic information
    """
    N, m = Phi_pdf.shape
    
    if len(mus) != m or len(sigmas) != m:
        raise ValueError("mus and sigmas must have length m")
    if len(z) != N or len(f) != N:
        raise ValueError("z and f must have length N")
    
    # Compute target raw moments
    M_target = compute_pdf_raw_moments(z, f, max_order=4)
    
    # Compute component raw moments
    vars_component = sigmas**2
    A_raw = compute_component_raw_moments(mus, vars_component)  # shape (5, m)
    
    # Variable indices
    idx_t_pdf = m
    idx_t1 = m + 1
    idx_t2 = m + 2
    idx_t3 = m + 3
    idx_t4 = m + 4
    n_vars = m + 5
    
    # Fallback for pdf_tolerance (form A only)
    if objective_form == "A" and pdf_tolerance is not None:
        taus = [pdf_tolerance, pdf_tolerance * 10, pdf_tolerance * 100]
    else:
        taus = [None]
    
    result = None
    last_error = None
    
    for tau in taus:
        # Objective function
        c = np.zeros(n_vars)
        if objective_form == "A":
            # Minimize Σ λ_n * t_n
            c[idx_t1] = lambda_raw[0]
            c[idx_t2] = lambda_raw[1]
            c[idx_t3] = lambda_raw[2]
            c[idx_t4] = lambda_raw[3]
        else:  # form B
            # Minimize λ_pdf * t_pdf + Σ λ_n * t_n
            c[idx_t_pdf] = lambda_pdf
            c[idx_t1] = lambda_raw[0]
            c[idx_t2] = lambda_raw[1]
            c[idx_t3] = lambda_raw[2]
            c[idx_t4] = lambda_raw[3]
        
        # Build constraint matrices
        # (A) PDF L∞ constraints: 2N rows
        # (B) Raw moment constraints: 8 rows (2 per moment)
        # Total: 2N + 8 inequality constraints
        n_ineq = 2 * N + 8
        
        # Use sparse matrix if large (original condition)
        # Use sparse matrices only for very large problems where density is low
        # For dense matrices (100% density), dense format is faster
        # Condition: N * m > 100000 ensures sparse matrices are only used for large problems
        use_sparse = N * m > 100000
        
        if use_sparse:
            A_ub = sparse.lil_matrix((n_ineq, n_vars))
        else:
            A_ub = np.zeros((n_ineq, n_vars))
        
        b_ub = np.zeros(n_ineq)
        
        # (A) PDF constraints: Phi_pdf @ w - t_pdf <= f
        if use_sparse:
            # Build sparse matrix more efficiently using COO format
            # For rows 0 to N-1: Phi_pdf @ w - t_pdf <= f
            row_indices = np.repeat(np.arange(N), m + 1)
            col_indices = np.concatenate([np.tile(np.arange(m), N), np.repeat([idx_t_pdf], N)])
            data = np.concatenate([Phi_pdf.flatten(), np.full(N, -1.0)])
            
            # For rows N to 2*N-1: -Phi_pdf @ w - t_pdf <= -f
            row_indices2 = np.repeat(np.arange(N, 2*N), m + 1)
            col_indices2 = np.concatenate([np.tile(np.arange(m), N), np.repeat([idx_t_pdf], N)])
            data2 = np.concatenate([-Phi_pdf.flatten(), np.full(N, -1.0)])
            
            # Combine all PDF constraints into one COO matrix
            row_indices_all = np.concatenate([row_indices, row_indices2])
            col_indices_all = np.concatenate([col_indices, col_indices2])
            data_all = np.concatenate([data, data2])
            A_ub = sparse.coo_matrix((data_all, (row_indices_all, col_indices_all)), shape=(2*N, n_vars))
            
            b_ub[:N] = f
            b_ub[N:2*N] = -f
        else:
            # Dense matrix assignment
            A_ub[:N, :m] = Phi_pdf
            A_ub[:N, idx_t_pdf] = -1.0
            b_ub[:N] = f
            A_ub[N:2*N, :m] = -Phi_pdf
            A_ub[N:2*N, idx_t_pdf] = -1.0
            b_ub[N:2*N] = -f
        
        # (B) Raw moment constraints (for M_1..M_4)
        row_base = 2 * N
        if use_sparse:
            # Build raw moment constraints efficiently
            row_indices_raw = []
            col_indices_raw = []
            data_raw = []
            for n in range(1, 5):  # n = 1, 2, 3, 4
                idx_tn = m + n
                # Use relative row indices (0-7) for A_ub_raw, then offset when combining
                rel_row = (n - 1) * 2
                # A_raw[n, :] @ w - t_n <= M_target[n]
                row_indices_raw.extend([rel_row] * (m + 1))
                col_indices_raw.extend(list(range(m)) + [idx_tn])
                data_raw.extend(A_raw[n, :].tolist() + [-1.0])
                # -A_raw[n, :] @ w - t_n <= -M_target[n]
                row_indices_raw.extend([rel_row + 1] * (m + 1))
                col_indices_raw.extend(list(range(m)) + [idx_tn])
                data_raw.extend((-A_raw[n, :]).tolist() + [-1.0])
                b_ub[row_base] = M_target[n]
                b_ub[row_base + 1] = -M_target[n]
                row_base += 2
            
            # Create sparse matrix for raw moment constraints (8 rows)
            A_ub_raw = sparse.coo_matrix((data_raw, (row_indices_raw, col_indices_raw)), shape=(8, n_vars))
            # Combine with PDF constraints (A_ub is already COO, so convert to CSR first for vstack)
            A_ub = sparse.vstack([A_ub.tocsr(), A_ub_raw.tocsr()])
        else:
            # Dense matrix assignment (more efficient)
            for n in range(1, 5):  # n = 1, 2, 3, 4
                idx_tn = m + n
                A_ub[row_base, :m] = A_raw[n, :]
                A_ub[row_base, idx_tn] = -1.0
                b_ub[row_base] = M_target[n]
                A_ub[row_base + 1, :m] = -A_raw[n, :]
                A_ub[row_base + 1, idx_tn] = -1.0
                b_ub[row_base + 1] = -M_target[n]
                row_base += 2
        
        # Convert to CSR if sparse (more efficient for solving)
        if use_sparse:
            A_ub = A_ub.tocsr()
        
        # Equality constraint: sum(w_j) = 1
        if use_sparse:
            # Use COO format for efficient construction, then convert to CSR
            A_eq = sparse.coo_matrix((np.ones(m), (np.zeros(m), np.arange(m))), shape=(1, n_vars))
            A_eq = A_eq.tocsr()
        else:
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds
        bounds = [(0, None)] * n_vars
        if objective_form == "A" and tau is not None:
            bounds[idx_t_pdf] = (0, tau)
        
        # Solve LP
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=solver
        )
        
        if result.success:
            break
        else:
            last_error = result.message if hasattr(result, 'message') else "LP solver failed"
    
    # Process result
    if result is None or not result.success:
        if objective_form == "A" and pdf_tolerance is not None:
            raise RuntimeError("LP infeasible even after relaxing pdf_tolerance")
        else:
            raise RuntimeError(f"LP solve failed: {last_error}")
    
    w = result.x[:m]
    t_pdf = result.x[idx_t_pdf]
    t_raw = result.x[[idx_t1, idx_t2, idx_t3, idx_t4]]
    
    # Compute actual raw moments from solution
    raw_mix = A_raw[:, :] @ w  # shape (5,)
    
    # Diagnostics
    diagnostics = {
        "n_dict": m,
        "t_pdf": float(t_pdf),
        "raw_target": M_target[1:5].tolist(),  # M_1..M_4
        "raw_mix": raw_mix[1:5].tolist(),
        "raw_abs_err": np.abs(raw_mix[1:5] - M_target[1:5]).tolist(),
        "n_nonzero": int(np.sum(w > 1e-10)),
        "selected_indices": np.argsort(w)[::-1].tolist(),
    }
    
    return {
        "w": w,
        "t_pdf": float(t_pdf),
        "t_raw": t_raw,
        "objective": float(result.fun),
        "status": 0 if result.success else -1,
        "message": result.message if hasattr(result, 'message') else "",
        "diagnostics": diagnostics,
    }


def fit_gmm_lp_simple(
    z: np.ndarray,
    f: np.ndarray,
    K: int,
    L: int,
    lp_params: Dict,
    objective_mode: str = "pdf",
) -> Tuple[Dict, Dict]:
    """
    Fit GMM using simple LP method (no greedy selection, no QP).
    
    Creates a dictionary of K*L Gaussian bases by dividing [μ-3σ, μ+3σ]
    into K segments and generating L different sigma values for each segment.
    Then solves LP to find optimal weights for all bases.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values at grid points, shape (N,)
    K : int
        Number of segments (and mean locations)
    L : int
        Number of sigma levels per segment
    lp_params : dict
        LP solver parameters:
        - solver: str - LP solver method (default: "highs")
        - sigma_min_scale: float - minimum sigma scale (default: 0.1)
        - sigma_max_scale: float - maximum sigma scale (default: 3.0)
        - lambda_mean: float - weight for mean error (for "moments" mode, default: 1.0)
        - lambda_variance: float - weight for variance error (for "moments" mode, default: 1.0)
        - lambda_skewness: float - weight for skewness error (for "moments" mode, default: 1.0)
        - lambda_kurtosis: float - weight for kurtosis error (for "moments" mode, default: 1.0)
        - pdf_tolerance: float - maximum PDF error for "moments" mode (default: 1e-6)
        - lambda_pdf: float - weight for PDF error (for "raw_moments" mode Form B, default: 1.0)
        - lambda_raw: list[float] - weights for raw moment errors [λ_1, λ_2, λ_3, λ_4] (for "raw_moments" mode, default: [1.0, 1.0, 1.0, 1.0])
        - objective_form: str - objective form "A" or "B" (for "raw_moments" mode, default: "A")
        - pdf_tolerance: float | None - maximum PDF error for "raw_moments" mode Form A (default: None)
    objective_mode : str
        Objective function mode: "pdf" (default), "moments", or "raw_moments"
        - "pdf": Minimize PDF error only (L∞ norm)
        - "moments": Minimize moment relative errors with PDF constraint (iterative LP)
        - "raw_moments": Minimize raw moment errors (M1-M4) with PDF constraint (fully linear LP)
    
    Returns:
    --------
    tuple
        (result_dict, timing_dict) where result_dict contains:
        - "weights": np.ndarray (K*L,) - mixing weights π_k (may be sparse)
        - "mus": np.ndarray (K*L,) - means μ_k
        - "sigmas": np.ndarray (K*L,) - standard deviations σ_k
        - "lp_objective": float - final LP objective value
        - "diagnostics": dict - additional diagnostic information
    """
    z = np.asarray(z)
    f = np.asarray(f)
    
    if K < 1 or L < 1:
        raise ValueError("K and L must be >= 1")
    if objective_mode not in ["pdf", "moments", "raw_moments"]:
        raise ValueError(f"objective_mode must be 'pdf', 'moments', or 'raw_moments', got '{objective_mode}'")
    
    # Initialize timing dictionary
    timing = {
        "dict_generation": 0.0,
        "basis_computation": 0.0,
        "lp_solving": 0.0,
        "total": 0.0
    }
    
    total_start = time.time()
    
    # Normalize PDF
    f_norm = normalize_pdf_on_grid(z, f)
    
    # Build dictionary (measure time)
    dict_start = time.time()
    dictionary = build_gaussian_dictionary_simple(
        z, f_norm,
        K=K,
        L=L,
        sigma_min_scale=lp_params.get("sigma_min_scale", 0.1),
        sigma_max_scale=lp_params.get("sigma_max_scale", 3.0),
    )
    timing["dict_generation"] = time.time() - dict_start
    
    # Compute basis matrices (measure time)
    basis_start = time.time()
    basis = compute_basis_matrices(z, dictionary["mus"], dictionary["sigmas"])
    Phi_pdf = basis["Phi_pdf"]
    # Phi_cdf is computed but not used in pdf mode (PDF only)
    timing["basis_computation"] = time.time() - basis_start
    
    # Solve LP (measure time)
    lp_start = time.time()
    solver = lp_params.get("solver", "highs")
    
    if objective_mode == "pdf":
        # Use PDF error minimization only (CDF is not considered)
        sol = solve_lp_pdf_linf(Phi_pdf, f_norm, solver)
    elif objective_mode == "moments":
        # Use moment error minimization with PDF constraint
        # Compute target moments from true PDF
        stats = compute_pdf_statistics(z, f_norm)
        target_mean = stats["mean"]
        target_variance = stats["std"]**2
        target_skewness = stats["skewness"]
        target_kurtosis = stats["kurtosis"]
        
        lambda_mean = lp_params.get("lambda_mean", 1.0)
        lambda_variance = lp_params.get("lambda_variance", 1.0)
        lambda_skewness = lp_params.get("lambda_skewness", 1.0)
        lambda_kurtosis = lp_params.get("lambda_kurtosis", 1.0)
        pdf_tolerance = lp_params.get("pdf_tolerance", 1e-6)
        
        max_moment_iter = lp_params.get("max_moment_iter", 5)
        moment_tolerance = lp_params.get("moment_tolerance", 1e-6)
        
        sol = solve_lp_pdf_moments_linf(
            Phi_pdf,
            dictionary["mus"],
            dictionary["sigmas"],
            f_norm,
            target_mean,
            target_variance,
            target_skewness,
            target_kurtosis,
            lambda_pdf=1.0,  # Not used in objective
            lambda_mean=lambda_mean,
            lambda_variance=lambda_variance,
            lambda_skewness=lambda_skewness,
            lambda_kurtosis=lambda_kurtosis,
            solver=solver,
            pdf_tolerance=pdf_tolerance,
            max_moment_iter=max_moment_iter,
            moment_tolerance=moment_tolerance
        )
    elif objective_mode == "raw_moments":
        # Use raw moment error minimization with PDF constraint (fully linear)
        pdf_tolerance = lp_params.get("pdf_tolerance", None)
        lambda_pdf = lp_params.get("lambda_pdf", 1.0)
        lambda_raw_val = lp_params.get("lambda_raw", [1.0, 1.0, 1.0, 1.0])
        # Handle both list and single float values
        if isinstance(lambda_raw_val, (int, float)):
            lambda_raw = tuple([float(lambda_raw_val)] * 4)
        elif isinstance(lambda_raw_val, (list, tuple)):
            lambda_raw = tuple([float(x) for x in lambda_raw_val[:4]])
            if len(lambda_raw) < 4:
                lambda_raw = lambda_raw + tuple([1.0] * (4 - len(lambda_raw)))
        else:
            lambda_raw = (1.0, 1.0, 1.0, 1.0)
        objective_form = lp_params.get("objective_form", "A")
        
        sol = solve_lp_pdf_rawmoments_linf(
            Phi_pdf=Phi_pdf,
            mus=dictionary["mus"],
            sigmas=dictionary["sigmas"],
            z=z,
            f=f_norm,
            pdf_tolerance=pdf_tolerance,
            lambda_pdf=lambda_pdf,
            lambda_raw=lambda_raw,
            solver=solver,
            objective_form=objective_form,
        )
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")
    
    timing["lp_solving"] = time.time() - lp_start
    
    if sol["status"] != 0:
        # Build detailed error message
        error_msg = f"LP solve failed: {sol['message']}\n\n"
        error_msg += "Problem Configuration:\n"
        error_msg += f"  Objective mode: {objective_mode}\n"
        error_msg += f"  Number of components (K): {K}\n"
        error_msg += f"  Dictionary size (L): {L}\n"
        error_msg += f"  Number of grid points: {len(z)}\n"
        error_msg += f"  Number of basis functions: {len(dictionary['mus'])}\n"
        error_msg += f"  Solver: {solver}\n"
        
        if objective_mode == "moments":
            error_msg += "\nMoment Constraints:\n"
            error_msg += f"  PDF tolerance: {lp_params.get('pdf_tolerance', 1e-6)}\n"
            error_msg += f"  Target mean: {target_mean:.6f}\n"
            error_msg += f"  Target variance: {target_variance:.6f}\n"
            error_msg += f"  Target skewness: {target_skewness:.6f}\n"
            error_msg += f"  Target kurtosis: {target_kurtosis:.6f}\n"
            error_msg += f"  Max moment iterations: {max_moment_iter}\n"
            error_msg += f"  Moment tolerance: {moment_tolerance}\n"
            error_msg += "\nPossible Solutions:\n"
            error_msg += "  1. Increase 'pdf_tolerance' in lp_params (e.g., 0.04 or 0.05)\n"
            error_msg += "  2. Increase 'L' (dictionary size) to provide more basis functions\n"
            error_msg += "  3. Increase 'K' (number of components)\n"
            error_msg += "  4. Check if target moments are achievable with the given dictionary\n"
        else:  # pdf mode
            error_msg += "\nPDF Constraints:\n"
            error_msg += f"  PDF error minimization mode\n"
            error_msg += "\nPossible Solutions:\n"
            error_msg += "  1. Increase 'L' (dictionary size) to provide more basis functions\n"
            error_msg += "  2. Increase 'K' (number of components)\n"
            error_msg += "  3. Check grid resolution and range\n"
        
        raise RuntimeError(error_msg)
    
    # Extract results
    weights = sol["w"]
    mus_all = dictionary["mus"]
    sigmas_all = dictionary["sigmas"]
    
    timing["total"] = time.time() - total_start
    
    diagnostics = {
        "t_pdf": sol.get("t_pdf", 0.0),
        "n_bases": len(weights),
        "n_nonzero": np.sum(weights > 1e-10),  # Count non-zero weights
        "L": L,  # Store L for display
        "objective_mode": objective_mode
    }
    
    # Add mode-specific diagnostics
    if objective_mode == "pdf":
        # PDF only mode - no CDF diagnostics
        pass
    elif objective_mode == "moments":
        diagnostics["moment_errors"] = sol.get("moment_errors", {})
        diagnostics["t_mean"] = sol.get("t_mean", 0.0)
        diagnostics["t_var"] = sol.get("t_var", 0.0)
        diagnostics["t_skew"] = sol.get("t_skew", 0.0)
        diagnostics["t_kurt"] = sol.get("t_kurt", 0.0)
    elif objective_mode == "raw_moments":
        # Raw moments mode - diagnostics already in sol["diagnostics"]
        diagnostics.update(sol.get("diagnostics", {}))
    
    return {
        "weights": weights,
        "mus": mus_all,
        "sigmas": sigmas_all,
        "lp_objective": sol["objective"],
        "diagnostics": diagnostics
    }, timing


