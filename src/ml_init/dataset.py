"""Dataset generation for MDN training."""
import numpy as np
from pathlib import Path
from typing import Optional
import json

from gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid


# Parameter ranges (hardcoded for reproducibility)
MU_RANGE = (-3.0, 3.0)
SIGMA_RANGE = (0.3, 2.0)
RHO_RANGE = (-0.99, 0.99)


def generate_dataset(
    output_dir: Path,
    n_train: int = 80000,
    n_val: int = 10000,
    n_test: int = 10000,
    seed_train: int = 0,
    seed_val: int = 1,
    seed_test: int = 2,
    z_min: float = -8.0,
    z_max: float = 8.0,
    n_points: int = 64,
) -> None:
    """
    Generate training dataset for MDN.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory for .npz files
    n_train : int
        Number of training samples
    n_val : int
        Number of validation samples
    n_test : int
        Number of test samples
    seed_train : int
        Random seed for training set
    seed_val : int
        Random seed for validation set
    seed_test : int
        Random seed for test set
    z_min : float
        Minimum z value for grid
    z_max : float
        Maximum z value for grid
    n_points : int
        Number of grid points
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed grid
    z = np.linspace(z_min, z_max, n_points)
    
    # Generate each split
    splits = [
        ("train", n_train, seed_train),
        ("val", n_val, seed_val),
        ("test", n_test, seed_test),
    ]
    
    for split_name, n_samples, seed in splits:
        z_array, f_array, params_array = _generate_split(
            z, n_samples, seed
        )
        
        # Save as .npz
        output_path = output_dir / f"{split_name}.npz"
        np.savez(
            output_path,
            z=z_array,
            f=f_array,
            params=params_array,
        )
        
        print(f"Generated {split_name}: {n_samples} samples -> {output_path}")


def _generate_split(
    z: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one split of the dataset.
    
    Parameters:
    -----------
    z : np.ndarray
        Fixed grid points, shape (N,)
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed
    
    Returns:
    --------
    z_array : np.ndarray
        Grid points (same for all samples), shape (N,)
    f_array : np.ndarray
        PDF values, shape (n_samples, N)
    params_array : np.ndarray
        Parameters (mu_x, sigma_x, mu_y, sigma_y, rho), shape (n_samples, 5)
    """
    np.random.seed(seed)
    
    N = len(z)
    f_array = np.zeros((n_samples, N))
    params_array = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        # Sample parameters
        mu_x = np.random.uniform(*MU_RANGE)
        sigma_x = np.random.uniform(*SIGMA_RANGE)
        mu_y = np.random.uniform(*MU_RANGE)
        sigma_y = np.random.uniform(*SIGMA_RANGE)
        rho = np.random.uniform(*RHO_RANGE)
        
        # Clamp rho to avoid numerical issues
        rho = np.clip(rho, -0.99, 0.99)
        
        # Generate PDF
        f = max_pdf_bivariate_normal(z, mu_x, sigma_x, mu_y, sigma_y, rho)
        
        # Ensure non-negative
        f = np.maximum(f, 0.0)
        
        # Normalize (handle single point case)
        if len(z) == 1:
            # Single point: PDF is just 1.0 (normalized)
            f = np.array([1.0])
        else:
            f = normalize_pdf_on_grid(z, f)
        
        f_array[i] = f
        params_array[i] = [mu_x, sigma_x, mu_y, sigma_y, rho]
    
    return z, f_array, params_array

