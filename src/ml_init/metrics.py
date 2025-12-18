"""Evaluation metrics for MDN initialization."""
import numpy as np


def compute_pdf_linf_error(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
) -> float:
    """
    Compute L∞ error between true and estimated PDFs.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f_true : np.ndarray
        True PDF values, shape (N,)
    f_hat : np.ndarray
        Estimated PDF values, shape (N,)
    
    Returns:
    --------
    error : float
        L∞ error: max_i |f_true(z_i) - f_hat(z_i)|
    """
    return float(np.max(np.abs(f_true - f_hat)))


def compute_cdf_linf_error(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
) -> float:
    """
    Compute L∞ error between true and estimated CDFs.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f_true : np.ndarray
        True PDF values, shape (N,)
    f_hat : np.ndarray
        Estimated PDF values, shape (N,)
    
    Returns:
    --------
    error : float
        L∞ error: max_i |F_true(z_i) - F_hat(z_i)|
    """
    N = len(z)
    if N == 1:
        # Single point: CDF is just the PDF value
        return float(np.abs(f_true[0] - f_hat[0]))
    
    w = np.full(N, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2  # Trapezoidal rule
    
    F_true = np.cumsum(f_true * w)
    F_hat = np.cumsum(f_hat * w)
    
    return float(np.max(np.abs(F_true - F_hat)))


def compute_quantile_error(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    quantiles: list[float],
) -> list[float]:
    """
    Compute quantile errors between true and estimated distributions.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f_true : np.ndarray
        True PDF values, shape (N,)
    f_hat : np.ndarray
        Estimated PDF values, shape (N,)
    quantiles : list[float]
        Quantile levels (e.g., [0.5, 0.9, 0.99])
    
    Returns:
    --------
    errors : list[float]
        Quantile errors: |q_p(true) - q_p(hat)| for each p in quantiles
    """
    N = len(z)
    if N == 1:
        # Single point: all quantiles map to the same point
        errors = [0.0] * len(quantiles)
        return errors
    
    w = np.full(N, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    F_true = np.cumsum(f_true * w)
    F_hat = np.cumsum(f_hat * w)
    
    errors = []
    for p in quantiles:
        # Find quantile: smallest z where CDF >= p
        idx_true = np.searchsorted(F_true, p)
        idx_hat = np.searchsorted(F_hat, p)
        
        q_true = z[min(idx_true, N - 1)]
        q_hat = z[min(idx_hat, N - 1)]
        
        errors.append(float(np.abs(q_true - q_hat)))
    
    return errors


def compute_cross_entropy(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute cross-entropy: -∫ f_true(z) log f_hat(z) dz
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f_true : np.ndarray
        True PDF values (normalized), shape (N,)
    f_hat : np.ndarray
        Estimated PDF values (normalized), shape (N,)
    epsilon : float
        Small value to avoid log(0)
    
    Returns:
    --------
    ce : float
        Cross-entropy loss
    """
    N = len(z)
    if N == 1:
        # Single point: use point value as weight
        f_hat_safe = max(f_hat[0], epsilon)
        ce = -f_true[0] * np.log(f_hat_safe)
        return float(ce)
    
    w = np.full(N, z[1] - z[0])
    w[0] = w[-1] = (z[1] - z[0]) / 2
    
    # Ensure non-negative
    f_hat_safe = np.maximum(f_hat, epsilon)
    
    # Compute: -∑ f_true * log(f_hat) * w
    ce = -np.sum(f_true * np.log(f_hat_safe) * w)
    
    return float(ce)

