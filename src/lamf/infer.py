"""Inference API for LAMF model.

This module provides the main inference function for LAMF (Learned Accelerated
Mixture Fitter) that can be used to fit GMM to input PDF.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import torch

from .model import LAMFFitter

logger = logging.getLogger(__name__)

# Global cache for loaded models
_model_cache: dict[str, tuple[LAMFFitter, dict]] = {}


class LAMFInitError(Exception):
    """Exception raised when LAMF initialization fails."""
    pass


def _get_cached_model(
    model_path: Union[str, Path],
    device: torch.device,
) -> tuple[LAMFFitter, dict]:
    """
    Load model from cache or disk.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to model checkpoint directory or .pt file
    device : torch.device
        Device to load model on
    
    Returns:
    --------
    model : LAMFFitter
        Loaded model
    metadata : dict
        Model metadata
    """
    model_path = Path(model_path)
    cache_key = f"{model_path}_{device}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Determine paths
    if model_path.is_dir():
        checkpoint_path = model_path / "lamf_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_path / "best_model.pt"
        metadata_path = model_path / "metadata.json"
    else:
        checkpoint_path = model_path
        metadata_path = model_path.parent / "metadata.json"
    
    # Check files exist
    if not checkpoint_path.exists():
        raise LAMFInitError(f"Model file not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise LAMFInitError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = LAMFFitter(
        N=metadata['N'],
        K=metadata['K'],
        T=metadata['T'],
        init_hidden_dim=metadata.get('init_hidden_dim', 256),
        init_num_layers=metadata.get('init_num_layers', 3),
        refine_hidden_dim=metadata.get('refine_hidden_dim', 128),
        refine_num_layers=metadata.get('refine_num_layers', 2),
        sigma_min=metadata.get('sigma_min', 1e-3),
        sigma_max=metadata.get('sigma_max', 5.0),
        pi_min=metadata.get('pi_min', 0.0),
        corr_scale=metadata.get('corr_scale', 0.5),
        dropout=metadata.get('dropout', 0.1),
        share_refine_weights=metadata.get('share_refine_weights', True),
    )
    
    # Load weights
    if str(checkpoint_path).endswith('best_model.pt'):
        # Load from training checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load state dict directly
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    # Cache
    _model_cache[cache_key] = (model, metadata)
    
    logger.info(f"Loaded LAMF model from {checkpoint_path}")
    logger.info(f"Model: N={metadata['N']}, K={metadata['K']}, T={metadata['T']}")
    
    return model, metadata


def _compute_probability_mass(
    z: np.ndarray,
    f: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert PDF to probability mass (w).
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    f : np.ndarray
        PDF values, shape (N,)
    normalize : bool
        If True, normalize w to sum to 1
    
    Returns:
    --------
    w : np.ndarray
        Probability mass, shape (N,)
    """
    N = len(z)
    if N == 1:
        return np.array([1.0])
    
    dz = z[1] - z[0]
    weights = np.full(N, dz)
    weights[0] = weights[-1] = dz / 2  # Trapezoidal rule
    
    w = f * weights
    
    if normalize:
        total = w.sum()
        if total > 1e-12:
            w = w / total
        else:
            # Uniform if PDF is near zero
            w = np.ones(N) / N
    
    return w


def _interpolate_to_model_grid(
    z_input: np.ndarray,
    f_input: np.ndarray,
    z_model: np.ndarray,
) -> np.ndarray:
    """
    Interpolate input PDF to model grid.
    
    Parameters:
    -----------
    z_input : np.ndarray
        Input grid points, shape (N_input,)
    f_input : np.ndarray
        Input PDF values, shape (N_input,)
    z_model : np.ndarray
        Model grid points, shape (N_model,)
    
    Returns:
    --------
    f_interp : np.ndarray
        Interpolated PDF values, shape (N_model,)
    """
    return np.interp(z_model, z_input, f_input, left=0.0, right=0.0)


def fit_gmm1d_to_pdf_lamf(
    z: np.ndarray,
    pdf: np.ndarray,
    K: int = 5,
    model_path: Optional[str] = None,
    device: str = "auto",
    T: Optional[int] = None,
    fallback_to_em: bool = True,
    em_fallback_kwargs: Optional[dict] = None,
    validate_result: bool = True,
    ce_threshold: float = 10.0,
) -> dict:
    """
    Fit GMM to 1D PDF using LAMF.
    
    This is the main inference API for LAMF. It takes a PDF on a grid and
    returns GMM parameters (pi, mu, var).
    
    Parameters:
    -----------
    z : np.ndarray
        Grid points, shape (N,)
    pdf : np.ndarray
        PDF values, shape (N,)
    K : int
        Number of GMM components (must match model)
    model_path : str, optional
        Path to LAMF model checkpoint. If None, uses default.
    device : str
        Device: "auto", "cuda", "cpu"
    T : int, optional
        Number of refinement iterations (uses model default if None)
    fallback_to_em : bool
        If True, fall back to EM if LAMF fails
    em_fallback_kwargs : dict, optional
        Kwargs to pass to EM fallback
    validate_result : bool
        If True, validate LAMF output and fallback if invalid
    ce_threshold : float
        Cross-entropy threshold for fallback
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'pi': Mixing weights, shape (K,)
        - 'mu': Means, shape (K,)
        - 'var': Variances, shape (K,)
        - 'method': "lamf" or "em" (if fallback used)
        - 'iterations': T (LAMF iterations)
    
    Raises:
    -------
    LAMFInitError
        If LAMF fails and fallback is disabled
    """
    # Device setup
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    
    # Default model path
    if model_path is None:
        # Use default LAMF checkpoint
        default_paths = [
            Path(__file__).parent.parent.parent / "lamf" / "checkpoints",
            Path("lamf/checkpoints"),
            Path("./lamf/checkpoints"),
        ]
        for p in default_paths:
            if p.exists():
                model_path = str(p)
                break
        else:
            raise LAMFInitError(
                "No default LAMF model found. Please specify model_path."
            )
    
    try:
        # Load model
        model, metadata = _get_cached_model(model_path, device_obj)
        
        # Validate K
        if K != metadata['K']:
            raise LAMFInitError(
                f"K mismatch: requested K={K}, model K={metadata['K']}"
            )
        
        # Override T if specified
        if T is not None and T != model.T:
            logger.info(f"Overriding T: model default {model.T} -> {T}")
            model.T = T
        
        # Prepare input
        z_model = np.linspace(
            metadata['z_min'],
            metadata['z_max'],
            metadata['N'],
        ).astype(np.float32)
        
        # Interpolate to model grid if needed
        if len(z) != metadata['N'] or not np.allclose(z, z_model):
            logger.debug(f"Interpolating input grid to model grid")
            f_interp = _interpolate_to_model_grid(z, pdf, z_model)
        else:
            f_interp = pdf.astype(np.float32)
        
        # Convert to probability mass
        w = _compute_probability_mass(z_model, f_interp)
        
        # Relative coordinate shift (if using relative mode)
        # Compute M1 (first moment) of input
        M1_input = np.sum(w * z_model)
        
        # Shift to relative coordinates (center at 0)
        z_relative = z_model - M1_input
        
        # Convert to tensors
        z_tensor = torch.from_numpy(z_relative).to(device_obj)
        w_tensor = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(device_obj)
        
        # Run LAMF
        with torch.no_grad():
            result = model(z_tensor, w_tensor)
        
        # Extract parameters
        pi = result['pi'][0].cpu().numpy()
        mu = result['mu'][0].cpu().numpy()
        sigma = result['sigma'][0].cpu().numpy()
        
        # Shift mu back to absolute coordinates
        mu = mu + M1_input
        
        # Validate
        if validate_result:
            # Check for NaN/Inf
            if np.any(~np.isfinite(pi)) or np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
                raise LAMFInitError("LAMF output contains NaN/Inf")
            
            # Check constraints
            if np.any(pi < 0) or np.any(sigma <= 0):
                raise LAMFInitError("LAMF output violates constraints")
            
            # Check cross-entropy (optional)
            # TODO: Add CE validation if needed
        
        var = sigma ** 2
        
        return {
            'pi': pi,
            'mu': mu,
            'var': var,
            'method': 'lamf',
            'iterations': model.T,
        }
    
    except (LAMFInitError, Exception) as e:
        logger.warning(f"LAMF failed: {e}")
        
        if not fallback_to_em:
            raise LAMFInitError(f"LAMF failed and fallback disabled: {e}")
        
        # Fallback to EM
        logger.info("Falling back to EM method")
        
        try:
            from src.gmm_fitting.em_method import fit_gmm_em
            
            fallback_kwargs = em_fallback_kwargs or {}
            em_result = fit_gmm_em(
                z=z,
                f=pdf,
                K=K,
                **fallback_kwargs,
            )
            
            return {
                'pi': em_result['pi'],
                'mu': em_result['mu'],
                'var': em_result['var'],
                'method': 'em_fallback',
                'iterations': em_result.get('iterations', -1),
            }
        except Exception as em_error:
            raise LAMFInitError(
                f"LAMF failed ({e}) and EM fallback also failed ({em_error})"
            )


def clear_model_cache() -> None:
    """Clear the model cache to free memory."""
    global _model_cache
    _model_cache.clear()
    logger.info("LAMF model cache cleared")

