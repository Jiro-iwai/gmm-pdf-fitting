"""Dataset generation for MDN training."""
import numpy as np
from pathlib import Path
from typing import Optional
import json

from src.gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid


# Parameter ranges (hardcoded for reproducibility)
# V2: Extended ranges to cover more edge cases
MU_RANGE = (-3.0, 3.0)
SIGMA_RANGE = (0.15, 2.5)  # Extended: was (0.3, 2.0), now covers smaller variances
RHO_RANGE = (-0.99, 0.99)

# V2: Sampling strategy options
SAMPLING_STRATEGY = "uniform"  # "uniform" or "improved"


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


def _sample_sigma_log_uniform(sigma_min: float, sigma_max: float, rng: np.random.Generator) -> float:
    """Sample sigma from log-uniform distribution (more small values)."""
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    return float(np.exp(rng.uniform(log_min, log_max)))


def _sample_params_improved(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    """
    Improved parameter sampling strategy.
    
    - 50% standard cases: symmetric parameters
    - 30% asymmetric cases: different X/Y parameters
    - 20% edge cases: extreme correlations or variance ratios
    """
    case_type = rng.random()
    
    if case_type < 0.5:
        # Standard symmetric case
        mu = rng.uniform(*MU_RANGE)
        sigma = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        rho = rng.uniform(*RHO_RANGE)
        return mu, sigma, mu, sigma, rho
    
    elif case_type < 0.8:
        # Asymmetric case: different X/Y parameters
        mu_x = rng.uniform(*MU_RANGE)
        mu_y = rng.uniform(*MU_RANGE)
        sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        rho = rng.uniform(*RHO_RANGE)
        return mu_x, sigma_x, mu_y, sigma_y, rho
    
    else:
        # Edge case: extreme parameters
        edge_type = rng.integers(0, 3)
        
        if edge_type == 0:
            # Extreme correlation
            mu_x = rng.uniform(*MU_RANGE)
            mu_y = rng.uniform(*MU_RANGE)
            sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
            sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
            # High correlation (positive or negative)
            if rng.random() < 0.5:
                rho = rng.uniform(0.85, 0.99)
            else:
                rho = rng.uniform(-0.99, -0.85)
            return mu_x, sigma_x, mu_y, sigma_y, rho
        
        elif edge_type == 1:
            # Extreme variance ratio
            mu_x = rng.uniform(*MU_RANGE)
            mu_y = rng.uniform(*MU_RANGE)
            # One sigma very small, one larger
            if rng.random() < 0.5:
                sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.3, rng)
                sigma_y = _sample_sigma_log_uniform(0.8, SIGMA_RANGE[1], rng)
            else:
                sigma_x = _sample_sigma_log_uniform(0.8, SIGMA_RANGE[1], rng)
                sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.3, rng)
            rho = rng.uniform(*RHO_RANGE)
            return mu_x, sigma_x, mu_y, sigma_y, rho
        
        else:
            # Very small variance (both)
            mu_x = rng.uniform(-1.0, 1.0)  # Narrower mean range for small variance
            mu_y = rng.uniform(-1.0, 1.0)
            sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.25, rng)
            sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.25, rng)
            rho = rng.uniform(*RHO_RANGE)
            return mu_x, sigma_x, mu_y, sigma_y, rho


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
    rng = np.random.default_rng(seed)
    
    N = len(z)
    f_array = np.zeros((n_samples, N))
    params_array = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        if SAMPLING_STRATEGY == "improved":
            mu_x, sigma_x, mu_y, sigma_y, rho = _sample_params_improved(rng)
        else:
            # Legacy uniform sampling
            mu_x = rng.uniform(*MU_RANGE)
            sigma_x = rng.uniform(*SIGMA_RANGE)
            mu_y = rng.uniform(*MU_RANGE)
            sigma_y = rng.uniform(*SIGMA_RANGE)
            rho = rng.uniform(*RHO_RANGE)
        
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MDN training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset files")
    parser.add_argument("--n_train", type=int, default=80000, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=10000, help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=10000, help="Number of test samples")
    parser.add_argument("--seed_train", type=int, default=0, help="Random seed for training set")
    parser.add_argument("--seed_val", type=int, default=1, help="Random seed for validation set")
    parser.add_argument("--seed_test", type=int, default=2, help="Random seed for test set")
    parser.add_argument("--z_min", type=float, default=-8.0, help="Minimum z value")
    parser.add_argument("--z_max", type=float, default=8.0, help="Maximum z value")
    parser.add_argument("--n_points", type=int, default=64, help="Number of grid points")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_dataset(
        output_dir=output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed_train=args.seed_train,
        seed_val=args.seed_val,
        seed_test=args.seed_test,
        z_min=args.z_min,
        z_max=args.z_max,
        n_points=args.n_points,
    )

