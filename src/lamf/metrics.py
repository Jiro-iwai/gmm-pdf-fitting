"""Evaluation metrics for LAMF.

This module provides metrics for evaluating GMM fitting quality:
- Cross-Entropy Loss
- PDF L∞ Error
- CDF L∞ Error  
- Moment Errors (M1-M4)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


# ==============================================================================
# PDF Computation
# ==============================================================================

def compute_gmm_pdf(
    z: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute GMM PDF values at grid points.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,) or (batch_size, N)
    pi : torch.Tensor
        Mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Means, shape (batch_size, K)
    sigma : torch.Tensor
        Standard deviations, shape (batch_size, K)
    
    Returns:
    --------
    f : torch.Tensor
        PDF values, shape (batch_size, N)
    """
    # Ensure z is 2D
    if z.dim() == 1:
        z = z.unsqueeze(0).expand(pi.shape[0], -1)  # (batch_size, N)
    
    batch_size, N = z.shape
    K = pi.shape[1]
    
    # Expand for broadcasting
    # z: (batch_size, N, 1)
    # mu, sigma: (batch_size, 1, K)
    z_exp = z.unsqueeze(-1)
    mu_exp = mu.unsqueeze(1)
    sigma_exp = sigma.unsqueeze(1)
    
    # Compute N(z; mu_k, sigma_k^2) for each component
    inv_sqrt_2pi = 1.0 / np.sqrt(2 * np.pi)
    normal_pdf = (
        inv_sqrt_2pi / (sigma_exp + 1e-12)
        * torch.exp(-0.5 * ((z_exp - mu_exp) / (sigma_exp + 1e-12)) ** 2)
    )  # (batch_size, N, K)
    
    # Weighted sum
    pi_exp = pi.unsqueeze(1)  # (batch_size, 1, K)
    f = (normal_pdf * pi_exp).sum(dim=-1)  # (batch_size, N)
    
    return f


def compute_log_gmm_pdf(
    z: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute log GMM PDF using log-sum-exp for numerical stability.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,) or (batch_size, N)
    pi : torch.Tensor
        Mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Means, shape (batch_size, K)
    sigma : torch.Tensor
        Standard deviations, shape (batch_size, K)
    eps : float
        Small value for numerical stability
    
    Returns:
    --------
    log_f : torch.Tensor
        Log PDF values, shape (batch_size, N)
    """
    # Ensure z is 2D
    if z.dim() == 1:
        z = z.unsqueeze(0).expand(pi.shape[0], -1)
    
    batch_size, N = z.shape
    K = pi.shape[1]
    
    # Expand for broadcasting
    z_exp = z.unsqueeze(-1)  # (batch_size, N, 1)
    mu_exp = mu.unsqueeze(1)  # (batch_size, 1, K)
    sigma_exp = sigma.unsqueeze(1)  # (batch_size, 1, K)
    
    # Log of normal PDF
    log_2pi = np.log(2 * np.pi)
    log_normal = (
        -0.5 * log_2pi
        - torch.log(sigma_exp + eps)
        - 0.5 * ((z_exp - mu_exp) / (sigma_exp + eps)) ** 2
    )  # (batch_size, N, K)
    
    # Log of mixing weights
    log_pi = torch.log(pi + eps).unsqueeze(1)  # (batch_size, 1, K)
    
    # Log-sum-exp
    log_weighted = log_normal + log_pi  # (batch_size, N, K)
    log_f = torch.logsumexp(log_weighted, dim=-1)  # (batch_size, N)
    
    return log_f


# ==============================================================================
# Loss Functions
# ==============================================================================

def compute_cross_entropy_loss(
    z: torch.Tensor,
    w: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute cross-entropy loss: -∑_i w_i * log(f_hat(z_i))
    
    This is the main training loss for LAMF.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    w : torch.Tensor
        Target PDF mass values, shape (batch_size, N)
    pi : torch.Tensor
        Predicted mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Predicted means, shape (batch_size, K)
    sigma : torch.Tensor
        Predicted standard deviations, shape (batch_size, K)
    eps : float
        Small value for numerical stability
    
    Returns:
    --------
    loss : torch.Tensor
        Cross-entropy loss, shape (batch_size,)
    """
    # Compute log PDF
    log_f_hat = compute_log_gmm_pdf(z, pi, mu, sigma, eps)  # (batch_size, N)
    
    # Cross-entropy: -∑_i w_i * log(f_hat(z_i))
    ce = -(w * log_f_hat).sum(dim=-1)  # (batch_size,)
    
    return ce


def compute_moment_loss(
    z: torch.Tensor,
    w: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    max_order: int = 4,
) -> torch.Tensor:
    """
    Compute moment matching loss: ∑_{n=1}^{max_order} (M_n^target - M_n^pred)^2
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    w : torch.Tensor
        Target PDF mass values, shape (batch_size, N)
    pi : torch.Tensor
        Predicted mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Predicted means, shape (batch_size, K)
    sigma : torch.Tensor
        Predicted standard deviations, shape (batch_size, K)
    max_order : int
        Maximum moment order (1 to 4)
    
    Returns:
    --------
    loss : torch.Tensor
        Moment loss, shape (batch_size,)
    """
    batch_size = w.shape[0]
    
    # Compute target raw moments from discrete distribution
    # M_n = ∑_i w_i * z_i^n
    z_exp = z.unsqueeze(0)  # (1, N)
    M_target = []
    for n in range(1, max_order + 1):
        M_n = (w * (z_exp ** n)).sum(dim=-1)  # (batch_size,)
        M_target.append(M_n)
    
    # Compute predicted raw moments from GMM (analytical formulas)
    # M_1 = ∑_k π_k μ_k
    # M_2 = ∑_k π_k (σ_k^2 + μ_k^2)
    # M_3 = ∑_k π_k (3σ_k^2 μ_k + μ_k^3)
    # M_4 = ∑_k π_k (3σ_k^4 + 6σ_k^2 μ_k^2 + μ_k^4)
    
    M_pred = []
    
    # M_1
    M1 = (pi * mu).sum(dim=-1)  # (batch_size,)
    M_pred.append(M1)
    
    if max_order >= 2:
        # M_2
        M2 = (pi * (sigma**2 + mu**2)).sum(dim=-1)
        M_pred.append(M2)
    
    if max_order >= 3:
        # M_3
        M3 = (pi * (3 * sigma**2 * mu + mu**3)).sum(dim=-1)
        M_pred.append(M3)
    
    if max_order >= 4:
        # M_4
        M4 = (pi * (3 * sigma**4 + 6 * sigma**2 * mu**2 + mu**4)).sum(dim=-1)
        M_pred.append(M4)
    
    # Sum of squared errors
    loss = torch.zeros(batch_size, device=w.device, dtype=w.dtype)
    for M_t, M_p in zip(M_target, M_pred):
        loss = loss + (M_t - M_p) ** 2
    
    return loss


def compute_pdf_l2_loss(
    z: torch.Tensor,
    f_true: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute PDF L2 loss: ∑_i (f_true(z_i) - f_hat(z_i))^2
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N)
    pi : torch.Tensor
        Predicted mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Predicted means, shape (batch_size, K)
    sigma : torch.Tensor
        Predicted standard deviations, shape (batch_size, K)
    
    Returns:
    --------
    loss : torch.Tensor
        PDF L2 loss, shape (batch_size,)
    """
    f_hat = compute_gmm_pdf(z, pi, mu, sigma)  # (batch_size, N)
    
    # L2 loss: mean squared error over grid points
    l2_loss = ((f_true - f_hat) ** 2).mean(dim=-1)  # (batch_size,)
    
    return l2_loss


def compute_pdf_linf_loss(
    z: torch.Tensor,
    f_true: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """
    Compute soft L∞ loss using LogSumExp approximation.
    
    L_linf ≈ (1/α) * log(Σ exp(α * |f_true - f_hat|))
    
    As α → ∞, this converges to max(|f_true - f_hat|).
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N)
    pi : torch.Tensor
        Predicted mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Predicted means, shape (batch_size, K)
    sigma : torch.Tensor
        Predicted standard deviations, shape (batch_size, K)
    alpha : float
        Softmax temperature. Higher = closer to true L∞.
        Recommended: 10-50
    
    Returns:
    --------
    loss : torch.Tensor
        Soft L∞ loss, shape (batch_size,)
    """
    f_hat = compute_gmm_pdf(z, pi, mu, sigma)  # (batch_size, N)
    
    # Absolute error at each grid point
    abs_error = torch.abs(f_true - f_hat)  # (batch_size, N)
    
    # Soft L∞ using LogSumExp: (1/α) * logsumexp(α * |error|)
    soft_linf = torch.logsumexp(alpha * abs_error, dim=-1) / alpha  # (batch_size,)
    
    return soft_linf


def compute_pdf_topk_loss(
    z: torch.Tensor,
    f_true: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Compute Top-k loss: average of k largest absolute errors.
    
    This focuses on the worst errors without being as extreme as L∞.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N)
    pi : torch.Tensor
        Predicted mixing weights, shape (batch_size, K)
    mu : torch.Tensor
        Predicted means, shape (batch_size, K)
    sigma : torch.Tensor
        Predicted standard deviations, shape (batch_size, K)
    k : int
        Number of top errors to average
    
    Returns:
    --------
    loss : torch.Tensor
        Top-k loss, shape (batch_size,)
    """
    f_hat = compute_gmm_pdf(z, pi, mu, sigma)  # (batch_size, N)
    
    # Absolute error at each grid point
    abs_error = torch.abs(f_true - f_hat)  # (batch_size, N)
    
    # Get top-k errors
    N = abs_error.shape[-1]
    k = min(k, N)
    topk_errors, _ = torch.topk(abs_error, k, dim=-1)  # (batch_size, k)
    
    # Average of top-k errors
    topk_loss = topk_errors.mean(dim=-1)  # (batch_size,)
    
    return topk_loss


def compute_pdf_huber_linf_loss(
    z: torch.Tensor,
    f_true: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    delta: float = 0.01,
    alpha: float = 20.0,
) -> torch.Tensor:
    """
    Compute Huber-style soft L∞ loss.
    
    Applies Huber transformation before soft L∞ to emphasize large errors.
    For |error| < delta: 0.5 * error^2 / delta
    For |error| >= delta: |error| - 0.5 * delta
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N)
    pi, mu, sigma : torch.Tensor
        GMM parameters
    delta : float
        Huber threshold. Errors above this are penalized linearly.
    alpha : float
        Softmax temperature for L∞ approximation.
    
    Returns:
    --------
    loss : torch.Tensor
        Huber L∞ loss, shape (batch_size,)
    """
    f_hat = compute_gmm_pdf(z, pi, mu, sigma)  # (batch_size, N)
    
    # Absolute error
    abs_error = torch.abs(f_true - f_hat)  # (batch_size, N)
    
    # Huber transformation
    huber_error = torch.where(
        abs_error < delta,
        0.5 * abs_error ** 2 / delta,
        abs_error - 0.5 * delta
    )
    
    # Soft L∞ using LogSumExp
    soft_linf = torch.logsumexp(alpha * huber_error, dim=-1) / alpha
    
    return soft_linf


def compute_deep_supervision_loss(
    z: torch.Tensor,
    w: torch.Tensor,
    intermediate_params: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lambda_mom: float = 0.0,
    lambda_pdf: float = 0.0,
    lambda_linf: float = 0.0,
    lambda_topk: float = 0.0,
    f_true: torch.Tensor = None,
    eta_schedule: str = "linear",
    linf_alpha: float = 20.0,
    topk_k: int = 10,
) -> torch.Tensor:
    """
    Compute deep supervision loss over all intermediate outputs.
    
    Parameters:
    -----------
    z : torch.Tensor
        Grid points, shape (N,)
    w : torch.Tensor
        Target PDF mass values, shape (batch_size, N)
    intermediate_params : list
        List of (pi, mu, sigma) at each iteration (T+1 total, including init)
    lambda_mom : float
        Weight for moment loss
    lambda_pdf : float
        Weight for PDF L2 loss (0 = CE only, 1 = PDF L2 only)
    lambda_linf : float
        Weight for soft L∞ loss (additional penalty for max errors)
    lambda_topk : float
        Weight for top-k loss (additional penalty for k largest errors)
    f_true : torch.Tensor
        True PDF values, shape (batch_size, N). Required if lambda_pdf > 0.
    eta_schedule : str
        Weighting schedule: "uniform", "linear" (later heavier), "final_only"
    linf_alpha : float
        Temperature for soft L∞ (higher = closer to true max)
    topk_k : int
        Number of top errors to average for top-k loss
    
    Returns:
    --------
    loss : torch.Tensor
        Total loss, shape ()
    """
    T = len(intermediate_params)
    
    # Compute weights
    if eta_schedule == "uniform":
        eta = torch.ones(T) / T
    elif eta_schedule == "linear":
        # η_t = t / sum(1..T) -> later iterations weighted more
        eta = torch.arange(1, T + 1, dtype=torch.float32)
        eta = eta / eta.sum()
    elif eta_schedule == "final_only":
        eta = torch.zeros(T)
        eta[-1] = 1.0
    else:
        raise ValueError(f"Unknown eta_schedule: {eta_schedule}")
    
    eta = eta.to(w.device)
    
    total_loss = torch.tensor(0.0, device=w.device, dtype=w.dtype)
    
    for t, (pi, mu, sigma) in enumerate(intermediate_params):
        # Cross-entropy loss (weighted by 1 - lambda_pdf)
        ce_loss = compute_cross_entropy_loss(z, w, pi, mu, sigma)
        
        # PDF L2 loss (weighted by lambda_pdf)
        if lambda_pdf > 0 and f_true is not None:
            pdf_loss = compute_pdf_l2_loss(z, f_true, pi, mu, sigma)
            # Combined loss: (1 - lambda_pdf) * CE + lambda_pdf * PDF_L2
            loss_t = (1.0 - lambda_pdf) * ce_loss + lambda_pdf * pdf_loss
        else:
            loss_t = ce_loss
        
        # Optional moment loss (additional)
        if lambda_mom > 0:
            mom_loss = compute_moment_loss(z, w, pi, mu, sigma)
            loss_t = loss_t + lambda_mom * mom_loss
        
        # Soft L∞ loss (additional penalty for max errors)
        if lambda_linf > 0 and f_true is not None:
            linf_loss = compute_pdf_linf_loss(z, f_true, pi, mu, sigma, alpha=linf_alpha)
            loss_t = loss_t + lambda_linf * linf_loss
        
        # Top-k loss (additional penalty for k largest errors)
        if lambda_topk > 0 and f_true is not None:
            topk_loss = compute_pdf_topk_loss(z, f_true, pi, mu, sigma, k=topk_k)
            loss_t = loss_t + lambda_topk * topk_loss
        
        # Weighted sum
        total_loss = total_loss + eta[t] * loss_t.mean()
    
    return total_loss


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def compute_pdf_linf_error(
    z: Union[torch.Tensor, np.ndarray],
    f_true: Union[torch.Tensor, np.ndarray],
    f_hat: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, float]:
    """
    Compute PDF L∞ error: max_i |f_true(z_i) - f_hat(z_i)|
    
    Parameters:
    -----------
    z : array-like
        Grid points, shape (N,)
    f_true : array-like
        True PDF values, shape (N,) or (batch_size, N)
    f_hat : array-like
        Estimated PDF values, shape (N,) or (batch_size, N)
    
    Returns:
    --------
    error : float or Tensor
        L∞ error
    """
    if isinstance(f_true, np.ndarray):
        return float(np.max(np.abs(f_true - f_hat)))
    else:
        if f_true.dim() == 1:
            return torch.max(torch.abs(f_true - f_hat))
        else:
            return torch.max(torch.abs(f_true - f_hat), dim=-1)[0]


def compute_cdf_linf_error(
    z: Union[torch.Tensor, np.ndarray],
    f_true: Union[torch.Tensor, np.ndarray],
    f_hat: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, float]:
    """
    Compute CDF L∞ error: max_i |F_true(z_i) - F_hat(z_i)|
    
    Parameters:
    -----------
    z : array-like
        Grid points, shape (N,)
    f_true : array-like
        True PDF values, shape (N,) or (batch_size, N)
    f_hat : array-like
        Estimated PDF values, shape (N,) or (batch_size, N)
    
    Returns:
    --------
    error : float or Tensor
        CDF L∞ error
    """
    if isinstance(f_true, np.ndarray):
        N = len(z)
        if N == 1:
            return float(np.abs(f_true[0] - f_hat[0]))
        
        dz = z[1] - z[0]
        w = np.full(N, dz)
        w[0] = w[-1] = dz / 2  # Trapezoidal rule
        
        F_true = np.cumsum(f_true * w)
        F_hat = np.cumsum(f_hat * w)
        
        return float(np.max(np.abs(F_true - F_hat)))
    else:
        N = z.shape[0]
        if N == 1:
            return torch.abs(f_true - f_hat).squeeze()
        
        dz = z[1] - z[0]
        w = torch.full((N,), dz, device=z.device, dtype=z.dtype)
        w[0] = w[-1] = dz / 2
        
        if f_true.dim() == 1:
            F_true = torch.cumsum(f_true * w, dim=0)
            F_hat = torch.cumsum(f_hat * w, dim=0)
            return torch.max(torch.abs(F_true - F_hat))
        else:
            F_true = torch.cumsum(f_true * w.unsqueeze(0), dim=-1)
            F_hat = torch.cumsum(f_hat * w.unsqueeze(0), dim=-1)
            return torch.max(torch.abs(F_true - F_hat), dim=-1)[0]


def compute_moment_errors(
    z: Union[torch.Tensor, np.ndarray],
    w_true: Union[torch.Tensor, np.ndarray],
    pi: Union[torch.Tensor, np.ndarray],
    mu: Union[torch.Tensor, np.ndarray],
    sigma: Union[torch.Tensor, np.ndarray],
    max_order: int = 4,
) -> dict:
    """
    Compute moment errors between target distribution and GMM.
    
    Parameters:
    -----------
    z : array-like
        Grid points, shape (N,)
    w_true : array-like
        Target PDF mass values, shape (N,)
    pi : array-like
        Predicted mixing weights, shape (K,)
    mu : array-like
        Predicted means, shape (K,)
    sigma : array-like
        Predicted standard deviations, shape (K,)
    max_order : int
        Maximum moment order
    
    Returns:
    --------
    errors : dict
        Dictionary with keys 'M1', 'M2', 'M3', 'M4' and absolute errors
    """
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
        w_true = w_true.cpu().numpy()
        pi = pi.cpu().numpy()
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
    
    errors = {}
    
    for n in range(1, max_order + 1):
        # Target moment
        M_target = np.sum(w_true * z**n)
        
        # Predicted moment (analytical)
        if n == 1:
            M_pred = np.sum(pi * mu)
        elif n == 2:
            M_pred = np.sum(pi * (sigma**2 + mu**2))
        elif n == 3:
            M_pred = np.sum(pi * (3 * sigma**2 * mu + mu**3))
        elif n == 4:
            M_pred = np.sum(pi * (3 * sigma**4 + 6 * sigma**2 * mu**2 + mu**4))
        else:
            raise ValueError(f"Moment order {n} not supported")
        
        errors[f'M{n}_target'] = float(M_target)
        errors[f'M{n}_pred'] = float(M_pred)
        errors[f'M{n}_error'] = float(np.abs(M_target - M_pred))
    
    return errors


def evaluate_gmm_fit(
    z: Union[torch.Tensor, np.ndarray],
    f_true: Union[torch.Tensor, np.ndarray],
    pi: Union[torch.Tensor, np.ndarray],
    mu: Union[torch.Tensor, np.ndarray],
    sigma: Union[torch.Tensor, np.ndarray],
) -> dict:
    """
    Comprehensive evaluation of GMM fit quality.
    
    Parameters:
    -----------
    z : array-like
        Grid points, shape (N,)
    f_true : array-like
        True PDF values (normalized), shape (N,)
    pi : array-like
        Predicted mixing weights, shape (K,)
    mu : array-like
        Predicted means, shape (K,)
    sigma : array-like
        Predicted standard deviations, shape (K,)
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all evaluation metrics
    """
    # Convert to numpy for consistent processing
    if isinstance(z, torch.Tensor):
        z_np = z.cpu().numpy()
        f_true_np = f_true.cpu().numpy()
        pi_np = pi.cpu().numpy()
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
    else:
        z_np = z
        f_true_np = f_true
        pi_np = pi
        mu_np = mu
        sigma_np = sigma
    
    # Compute predicted PDF
    N = len(z_np)
    K = len(pi_np)
    f_hat_np = np.zeros(N)
    
    inv_sqrt_2pi = 1.0 / np.sqrt(2 * np.pi)
    for k in range(K):
        f_hat_np += pi_np[k] * inv_sqrt_2pi / sigma_np[k] * np.exp(
            -0.5 * ((z_np - mu_np[k]) / sigma_np[k]) ** 2
        )
    
    # Compute w (mass values) for moment calculation
    dz = z_np[1] - z_np[0] if N > 1 else 1.0
    w_true = f_true_np * dz
    w_true = w_true / w_true.sum()  # Normalize
    
    # Compute metrics
    metrics = {
        'pdf_linf': compute_pdf_linf_error(z_np, f_true_np, f_hat_np),
        'cdf_linf': compute_cdf_linf_error(z_np, f_true_np, f_hat_np),
    }
    
    # Cross-entropy
    eps = 1e-12
    f_hat_safe = np.maximum(f_hat_np, eps)
    ce = -np.sum(w_true * np.log(f_hat_safe))
    metrics['cross_entropy'] = float(ce)
    
    # Moment errors
    moment_errors = compute_moment_errors(z_np, w_true, pi_np, mu_np, sigma_np)
    metrics.update(moment_errors)
    
    return metrics

