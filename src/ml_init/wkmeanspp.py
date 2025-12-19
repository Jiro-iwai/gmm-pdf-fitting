"""Weighted k-means++ initialization for GMM."""
import numpy as np


def weighted_kmeanspp(
    z: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    K: int,
    reg_var: float = 1e-6,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted k-means++ initialization for GMM.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values (normalized), shape (N,)
    w : np.ndarray
        Integration weights (trapezoidal rule), shape (N,)
    K : int
        Number of clusters (GMM components)
    reg_var : float
        Minimum variance (regularization)
    max_iter : int
        Maximum iterations for Lloyd's algorithm
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    pi : np.ndarray
        Mixing weights, shape (K,)
    mu : np.ndarray
        Component means, shape (K,)
    var : np.ndarray
        Component variances, shape (K,)
    """
    N = len(z)
    
    # Compute weights: omega_i = max(f_i, 0) * w_i
    omega = np.maximum(f, 0.0) * w
    omega_sum = np.sum(omega)
    
    if omega_sum <= 0:
        # Fallback: uniform weights
        omega = np.ones(N) * (z[1] - z[0]) if N > 1 else np.ones(N)
        omega_sum = np.sum(omega)
    
    # Normalize to probabilities
    p = omega / omega_sum
    
    # Initialize centers using weighted k-means++
    centers = _weighted_kmeanspp_init(z, p, K)
    
    # Lloyd's algorithm
    centers = _lloyd_iteration(z, p, centers, max_iter, tol)
    
    # Compute GMM parameters
    pi, mu, var = _compute_gmm_params(z, p, centers, reg_var)
    
    # Sort by mu (ascending)
    idx = np.argsort(mu)
    pi = pi[idx]
    mu = mu[idx]
    var = var[idx]
    
    return pi, mu, var


def _weighted_kmeanspp_init(
    z: np.ndarray,
    p: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Initialize centers using weighted k-means++.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    p : np.ndarray
        Normalized weights (probabilities), shape (N,)
    K : int
        Number of centers
    
    Returns:
    --------
    centers : np.ndarray
        Initial centers, shape (K,)
    """
    N = len(z)
    centers = np.zeros(K)
    
    # First center: sample according to p
    centers[0] = np.random.choice(N, p=p)
    
    # Subsequent centers: weighted by p * D^2
    for k in range(1, K):
        # Compute minimum distance to existing centers
        D = np.min(np.abs(z[:, None] - centers[:k]), axis=1)
        
        # Selection probability: p_i * D_i^2
        prob = p * D**2
        prob_sum = np.sum(prob)
        
        if prob_sum <= 0:
            # Fallback: choose point with maximum p
            centers[k] = np.argmax(p)
        else:
            prob = prob / prob_sum
            centers[k] = np.random.choice(N, p=prob)
    
    # Convert indices to actual z values
    return z[centers.astype(int)]


def _lloyd_iteration(
    z: np.ndarray,
    p: np.ndarray,
    centers: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Lloyd's algorithm for k-means.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    p : np.ndarray
        Normalized weights, shape (N,)
    centers : np.ndarray
        Initial centers, shape (K,)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    centers : np.ndarray
        Converged centers, shape (K,)
    """
    K = len(centers)
    N = len(z)
    
    for _ in range(max_iter):
        # Assignment: find nearest center
        distances = np.abs(z[:, None] - centers[None, :])  # (N, K)
        assignments = np.argmin(distances, axis=1)  # (N,)
        
        # Update centers: weighted mean
        new_centers = np.zeros(K)
        for k in range(K):
            mask = assignments == k
            if np.sum(mask) == 0:
                # Empty cluster: reinitialize
                D = np.min(np.abs(z[:, None] - centers[None, :]), axis=1)
                prob = p * D**2
                prob_sum = np.sum(prob)
                if prob_sum > 0:
                    prob = prob / prob_sum
                    new_centers[k] = z[np.random.choice(N, p=prob)]
                else:
                    new_centers[k] = z[np.argmax(p)]
            else:
                # Weighted mean
                p_k = p[mask]
                z_k = z[mask]
                new_centers[k] = np.sum(p_k * z_k) / np.sum(p_k)
        
        # Check convergence
        if np.max(np.abs(new_centers - centers)) < tol:
            break
        
        centers = new_centers
    
    return centers


def _compute_gmm_params(
    z: np.ndarray,
    p: np.ndarray,
    centers: np.ndarray,
    reg_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GMM parameters from k-means centers.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    p : np.ndarray
        Normalized weights, shape (N,)
    centers : np.ndarray
        Cluster centers, shape (K,)
    reg_var : float
        Minimum variance
    
    Returns:
    --------
    pi : np.ndarray
        Mixing weights, shape (K,)
    mu : np.ndarray
        Component means, shape (K,)
    var : np.ndarray
        Component variances, shape (K,)
    """
    K = len(centers)
    N = len(z)
    
    # Assignment
    distances = np.abs(z[:, None] - centers[None, :])
    assignments = np.argmin(distances, axis=1)
    
    # Compute parameters for each cluster
    pi = np.zeros(K)
    mu = np.zeros(K)
    var = np.zeros(K)
    
    for k in range(K):
        mask = assignments == k
        if np.sum(mask) == 0:
            # Empty cluster: use default values
            pi[k] = 1.0 / K
            mu[k] = centers[k]
            var[k] = reg_var
        else:
            # Mixing weight
            pi[k] = np.sum(p[mask])
            
            # Mean
            p_k = p[mask]
            z_k = z[mask]
            mu[k] = np.sum(p_k * z_k) / np.sum(p_k)
            
            # Variance (weighted variance + floor)
            var[k] = np.sum(p_k * (z_k - mu[k])**2) / np.sum(p_k) + reg_var
    
    # Normalize pi (should already be normalized, but ensure)
    pi = pi / np.sum(pi)
    
    return pi, mu, var

