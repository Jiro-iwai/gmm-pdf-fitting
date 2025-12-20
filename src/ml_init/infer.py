"""MDN inference API for GMM initialization."""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

from src.ml_init.model import MDNModel, log_gmm_pdf
from src.ml_init.eval import load_model_and_metadata


class MDNInitError(RuntimeError):
    """Raised when MDN-based initialization fails (load/version/device/numerics)."""
    pass


def _compute_m1(z: np.ndarray, f: np.ndarray) -> float:
    """
    Compute first moment (mean) of PDF.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values, shape (N,)
    
    Returns:
    --------
    m1 : float
        First moment E[Z] = ∫ z f(z) dz
    """
    return float(np.trapezoid(z * f, z))


def _shift_pdf_to_relative(
    z_orig: np.ndarray,
    f_orig: np.ndarray,
    m1: float,
    z_target: np.ndarray,
) -> np.ndarray:
    """
    Shift PDF to relative coordinates (centered at M1=0) and interpolate to target grid.
    
    Parameters:
    -----------
    z_orig : np.ndarray
        Original grid points, shape (N,)
    f_orig : np.ndarray
        Original PDF values, shape (N,)
    m1 : float
        First moment of original PDF
    z_target : np.ndarray
        Target grid points, shape (N_target,)
    
    Returns:
    --------
    f_shifted : np.ndarray
        Shifted and interpolated PDF, shape (N_target,)
    """
    # Shift grid to relative coordinates
    z_shifted = z_orig - m1
    
    # Interpolate to target grid
    f_shifted = np.interp(z_target, z_shifted, f_orig, left=0.0, right=0.0)
    
    # Ensure non-negative
    f_shifted = np.maximum(f_shifted, 0.0)
    
    # Re-normalize
    integral = np.trapezoid(f_shifted, z_target)
    if integral > 0:
        f_shifted = f_shifted / integral
    else:
        # Fallback: uniform distribution
        f_shifted = np.ones_like(f_shifted) / (z_target[-1] - z_target[0])
    
    return f_shifted


# Module-level cache for loaded models
_model_cache: dict[str, Tuple[MDNModel, dict, np.ndarray]] = {}


def _get_cached_model(
    model_path: Path,
    device: str,
) -> Tuple[MDNModel, dict, np.ndarray]:
    """
    Get cached model or load and cache it.
    
    Parameters:
    -----------
    model_path : Path
        Path to model checkpoint
    device : str
        Device ("cpu", "cuda", or "auto")
    
    Returns:
    --------
    model : MDNModel
        Loaded model (moved to device)
    metadata : dict
        Model metadata
    z : np.ndarray
        Grid points
    """
    model_path_str = str(model_path.resolve())
    
    # Check cache
    if model_path_str in _model_cache:
        model, metadata, z = _model_cache[model_path_str]
        # Move model to requested device
        device_obj = torch.device(device)
        if next(model.parameters()).device != device_obj:
            model = model.to(device_obj)
        return model, metadata, z
    
    # Load and cache
    model, metadata, z = load_model_and_metadata(model_path)
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Cache the model (on CPU to save GPU memory)
    model_cpu = model.cpu()
    _model_cache[model_path_str] = (model_cpu, metadata, z)
    
    # Return model on requested device
    return model_cpu.to(device_obj), metadata, z


def clear_model_cache():
    """Clear the model cache."""
    global _model_cache
    _model_cache.clear()


def get_cache_size() -> int:
    """Get the number of cached models."""
    return len(_model_cache)


def _resample_to_fixed_grid(
    z_orig: np.ndarray,
    f_orig: np.ndarray,
    z_target: np.ndarray,
) -> np.ndarray:
    """
    Resample PDF from original grid to target grid using linear interpolation.
    
    Parameters:
    -----------
    z_orig : np.ndarray
        Original grid points, shape (N_orig,)
    f_orig : np.ndarray
        Original PDF values, shape (N_orig,)
    z_target : np.ndarray
        Target grid points, shape (N_target,)
    
    Returns:
    --------
    f_target : np.ndarray
        Resampled PDF values, shape (N_target,)
    """
    # Ensure non-negative
    f_orig = np.maximum(f_orig, 0.0)
    
    # Linear interpolation
    interp_func = interp1d(
        z_orig,
        f_orig,
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    
    f_target = interp_func(z_target)
    
    # Ensure non-negative
    f_target = np.maximum(f_target, 0.0)
    
    # Normalize
    w = np.full(len(z_target), z_target[1] - z_target[0])
    w[0] = w[-1] = (z_target[1] - z_target[0]) / 2
    integral = np.sum(f_target * w)
    
    if integral > 0:
        f_target = f_target / integral
    else:
        # Fallback: uniform distribution
        f_target = np.ones_like(f_target) / np.sum(w)
    
    return f_target


def _sort_components(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort GMM components by mu (ascending), then by -pi (descending), then by sigma (ascending).
    
    Parameters:
    -----------
    pi : np.ndarray
        Mixing weights, shape (K,)
    mu : np.ndarray
        Component means, shape (K,)
    sigma : np.ndarray
        Component standard deviations, shape (K,)
    
    Returns:
    --------
    pi_sorted : np.ndarray
    mu_sorted : np.ndarray
    sigma_sorted : np.ndarray
    """
    # Lexicographic sort: (mu, -pi, sigma)
    idx = np.lexsort((sigma, -pi, mu))
    
    return pi[idx], mu[idx], sigma[idx]


def mdn_predict_init(
    z: np.ndarray,
    f: np.ndarray,
    K: int,
    model_path: str | Path,
    device: str = "auto",
    reg_var: float = 1e-6,
) -> dict:
    """
    Predict GMM initialization parameters using MDN.
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values (normalized), shape (N,)
    K : int
        Number of GMM components
    model_path : str | Path
        Path to model checkpoint (.pt file)
    device : str
        Device ("cpu", "cuda", or "auto")
    reg_var : float
        Minimum variance (regularization)
    
    Returns:
    --------
    result : dict
        {
            "pi": np.ndarray shape (K,),
            "mu": np.ndarray shape (K,),
            "var": np.ndarray shape (K,)
        }
    
    Raises:
    -------
    MDNInitError
        If model loading, version mismatch, or inference fails
    """
    model_path = Path(model_path)
    
    # Check file exists
    if not model_path.exists():
        raise MDNInitError(
            f"MDN init failed: model file not found. path={model_path}"
        )
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and metadata (cached)
        model, metadata, z_model = _get_cached_model(model_path, device)
        model.eval()
        
        # Check K matches
        if metadata["K_model"] != K:
            raise MDNInitError(
                f"MDN init failed: K mismatch. "
                f"expected K={metadata['K_model']} got K={K} "
                f"path={model_path}"
            )
        
        # Check coordinate mode (default to "absolute" for backward compatibility)
        coordinate_mode = metadata.get("coordinate_mode", "absolute")
        
        # Resample to model's grid if needed
        if len(z) != len(z_model) or not np.allclose(z, z_model):
            f_resampled = _resample_to_fixed_grid(z, f, z_model)
        else:
            f_resampled = f.copy()
        
        # Ensure non-negative and normalized
        f_resampled = np.maximum(f_resampled, 0.0)
        w = np.full(len(z_model), z_model[1] - z_model[0])
        w[0] = w[-1] = (z_model[1] - z_model[0]) / 2
        integral = np.sum(f_resampled * w)
        if integral > 0:
            f_resampled = f_resampled / integral
        else:
            raise MDNInitError(
                f"MDN init failed: PDF integral is non-positive. "
                f"path={model_path}"
            )
        
        # Apply coordinate transformation for relative mode
        m1_offset = 0.0
        if coordinate_mode == "relative":
            # Compute M1 of input PDF
            m1_offset = _compute_m1(z_model, f_resampled)
            # Shift PDF to relative coordinates (M1 → 0)
            f_resampled = _shift_pdf_to_relative(z_model, f_resampled, m1_offset, z_model)
        
        # Convert to tensor
        z_torch = torch.from_numpy(z_model).float().to(device)
        f_torch = torch.from_numpy(f_resampled).float().unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            alpha, mu, beta = model(f_torch)
            pi = torch.softmax(alpha, dim=-1)
            sigma = torch.nn.functional.softplus(beta) + model.sigma_min
        
        # Convert to numpy
        pi_np = pi[0].cpu().numpy()
        mu_np = mu[0].cpu().numpy()
        sigma_np = sigma[0].cpu().numpy()
        
        # Inverse transform for relative mode: shift mu back to absolute coordinates
        if coordinate_mode == "relative":
            mu_np = mu_np + m1_offset
        
        # Check for NaN/Inf
        if not (np.all(np.isfinite(pi_np)) and 
                np.all(np.isfinite(mu_np)) and 
                np.all(np.isfinite(sigma_np))):
            raise MDNInitError(
                f"MDN init failed: non-finite output detected. "
                f"path={model_path}"
            )
        
        # Compute PDF to check validity
        log_f_hat = log_gmm_pdf(z_torch, pi, mu, sigma)
        f_hat = torch.exp(log_f_hat[0]).cpu().numpy()
        
        if not np.all(np.isfinite(f_hat)):
            raise MDNInitError(
                f"MDN init failed: non-finite PDF output. "
                f"path={model_path}"
            )
        
        if np.any(f_hat < 0):
            raise MDNInitError(
                f"MDN init failed: negative PDF values detected. "
                f"path={model_path}"
            )
        
        # Sort components
        pi_sorted, mu_sorted, sigma_sorted = _sort_components(
            pi_np, mu_np, sigma_np
        )
        
        # Convert to variance
        var_sorted = sigma_sorted ** 2
        
        # Apply variance floor
        var_sorted = np.maximum(var_sorted, reg_var)
        
        # Normalize pi (should already be normalized, but ensure)
        pi_sorted = pi_sorted / np.sum(pi_sorted)
        
        return {
            "pi": pi_sorted,
            "mu": mu_sorted,
            "var": var_sorted,
        }
        
    except MDNInitError:
        raise
    except FileNotFoundError as e:
        raise MDNInitError(
            f"MDN init failed: file not found. "
            f"path={model_path}, error={str(e)}"
        ) from e
    except json.JSONDecodeError as e:
        raise MDNInitError(
            f"MDN init failed: invalid metadata JSON. "
            f"path={model_path}, error={str(e)}"
        ) from e
    except Exception as e:
        raise MDNInitError(
            f"MDN init failed: unexpected error. "
            f"path={model_path}, error={str(e)}"
        ) from e

